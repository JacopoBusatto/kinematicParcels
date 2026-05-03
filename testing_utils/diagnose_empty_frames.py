"""
Diagnostic script - Step 1: count active particles per timestep.

Loads the zarr output and counts how many particles have a valid (non-NaN)
position at each calendar timestamp, using three independent methods:

  A. Raw zarr arrays (no xarray, no post-processing)
     → ground truth: what was actually written to disk

  B. load_trajectory_table (full postprocessing pipeline)
     → what the animation actually sees after sanitization

  C. Timestamp precision audit
     → checks for near-duplicate timestamps (microsecond-level artifacts)
        that would cause the animation's exact-equality filter to silently
        skip particles on sparse frames

Interpretation:
  - If A shows a dip  → data problem (integration or velocity field dropped particles)
  - If B dips but A doesn't → postprocessing pipeline is discarding something
  - If C shows split timestamps → animation timestamp-equality bug

Usage:
    cd <repo root>
    python testing_utils/diagnose_empty_frames.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from kinematicparcels.postprocessing.io import load_trajectory_table

# ---------------------------------------------------------------------------
# Configuration — edit as needed
# ---------------------------------------------------------------------------
ZARR_PATH = Path("./outputs/sicily/ex20_PAGES26_LKM.zarr")
OUTPUT_PLOT = Path("./outputs/sicily/postprocessing/diagnose_empty_frames.png")
EPOCH = pd.Timestamp("2026-04-01 00:00")   # Parcels time reference for this run

TRUNCATE_STAGNANT = True
STAGNANT_TOL = 1.0e-6
STAGNANT_MIN_CONSECUTIVE = 3
# ---------------------------------------------------------------------------


def _raw_count_per_timestamp(zarr_path: Path) -> pd.Series:
    """
    Method A: open zarr directly and count non-NaN lon values per timestamp.

    Returns a Series indexed by Timestamp, values = particle count.
    """
    print("  [A] Reading raw zarr arrays …")
    z = zarr.open(str(zarr_path), mode="r")
    lon = z["lon"][:]        # (n_traj, n_obs)  float32
    t_raw = z["time"][:]     # (n_traj, n_obs)  float64  seconds from EPOCH

    valid_lon = ~np.isnan(lon)
    valid_time = ~np.isnan(t_raw)

    # Flatten; keep only cells with a valid time entry
    flat_mask = valid_time.ravel()
    t_flat = t_raw.ravel()[flat_mask]
    lon_valid_flat = valid_lon.ravel()[flat_mask]

    df = pd.DataFrame({"time_s": t_flat, "valid": lon_valid_flat.astype(int)})
    df["datetime"] = EPOCH + pd.to_timedelta(df["time_s"], unit="s")

    count = df.groupby("datetime")["valid"].sum().rename("raw_active_count")
    total = df.groupby("datetime").size().rename("raw_total_count")
    result = pd.concat([count, total], axis=1)
    result.index.name = "time"
    return result


def _pipeline_count_per_timestamp(zarr_path: Path) -> pd.Series:
    """
    Method B: load via load_trajectory_table (same path as the animation).

    Returns a Series indexed by Timestamp, values = row count (= active particles).
    """
    print("  [B] Loading via load_trajectory_table …")
    df = load_trajectory_table(
        zarr_path,
        truncate_stagnant=TRUNCATE_STAGNANT,
        stagnant_tol=STAGNANT_TOL,
        stagnant_min_consecutive=STAGNANT_MIN_CONSECUTIVE,
    )
    df["time"] = pd.to_datetime(df["time"])
    count = df.dropna(subset=["lon", "lat"]).groupby("time").size().rename("pipeline_count")
    count.index.name = "time"
    return count


def _audit_timestamp_precision(zarr_path: Path) -> pd.DataFrame:
    """
    Method C: detect near-duplicate timestamps.

    For each nominal hour bucket, check how many distinct float64 time values
    (in seconds) exist.  More than one value per bucket means the exact-equality
    filter in the animation will silently skip particles.

    Returns a DataFrame with columns:
        hour_bucket, n_distinct_values, values_sample
    Only buckets with n_distinct_values > 1 are returned.
    """
    print("  [C] Auditing timestamp precision …")
    z = zarr.open(str(zarr_path), mode="r")
    t_raw = z["time"][:]
    t_flat = t_raw.ravel()
    t_flat = t_flat[~np.isnan(t_flat)]

    df = pd.DataFrame({"time_s": t_flat})
    df["datetime"] = EPOCH + pd.to_timedelta(df["time_s"], unit="s")
    df["hour_bucket"] = df["datetime"].dt.floor("h")

    groups = df.groupby("hour_bucket")["time_s"].apply(lambda x: sorted(x.unique()))
    result = groups.reset_index()
    result.columns = ["hour_bucket", "distinct_values"]
    result["n_distinct"] = result["distinct_values"].apply(len)

    splits = result[result["n_distinct"] > 1].copy()
    if splits.empty:
        print("     → No timestamp precision splits found (all timestamps land exactly on the hour).")
    else:
        print(f"     → Found {len(splits)} hour buckets with multiple distinct time values!")
        print(splits[["hour_bucket", "n_distinct"]].to_string(index=False))

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not ZARR_PATH.exists():
        raise FileNotFoundError(f"Zarr not found: {ZARR_PATH}")

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Active particle count per timestep ===")
    raw = _raw_count_per_timestamp(ZARR_PATH)
    pipe = _pipeline_count_per_timestamp(ZARR_PATH)
    splits = _audit_timestamp_precision(ZARR_PATH)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"  Raw zarr : {len(raw)} timesteps, "
          f"max active = {raw['raw_active_count'].max()}, "
          f"min active = {raw['raw_active_count'].min()}")
    print(f"  Pipeline : {len(pipe)} timesteps, "
          f"max = {pipe.max()}, "
          f"min = {pipe.min()}")

    # Find frames where pipeline count drops significantly compared to raw
    threshold = 0.1  # flag frames with < 10% of the max
    max_active = pipe.max()
    sparse_frames = pipe[pipe < threshold * max_active]
    if sparse_frames.empty:
        print(f"  No sparse frames found (threshold < {int(threshold*100)}% of max).")
    else:
        print(f"\n  !! {len(sparse_frames)} SPARSE FRAMES (< {int(threshold*100)}% of max {int(max_active)}) in pipeline data:")
        print(sparse_frames.to_string())

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # Panel A: raw count
    ax = axes[0]
    ax.plot(raw.index, raw["raw_active_count"], color="steelblue", lw=1.2)
    ax.set_ylabel("Active particles\n(raw zarr)")
    ax.set_title("A  —  Non-NaN particle count per timestamp (raw zarr)")
    ax.grid(True, alpha=0.4)

    # Panel B: pipeline count
    ax = axes[1]
    ax.plot(pipe.index, pipe.values, color="darkorange", lw=1.2)
    if not sparse_frames.empty:
        ax.scatter(sparse_frames.index, sparse_frames.values,
                   color="red", s=30, zorder=5, label="sparse frames")
        ax.legend()
    ax.set_ylabel("Active particles\n(pipeline / animation)")
    ax.set_title("B  —  Particle count after load_trajectory_table (animation view)")
    ax.grid(True, alpha=0.4)

    # Panel C: timestamp split count per frame
    ax = axes[2]
    if not splits.empty:
        ax.bar(splits["hour_bucket"], splits["n_distinct"],
               color="crimson", width=pd.Timedelta(hours=0.8))
        ax.set_title("C  —  Timestamp precision splits (>1 distinct float per hour → animation bug)")
    else:
        ax.text(0.5, 0.5, "No timestamp precision splits detected",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title("C  —  Timestamp precision audit")
    ax.set_ylabel("Distinct float values\nper hour bucket")
    ax.grid(True, alpha=0.4)

    fig.autofmt_xdate()
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nPlot saved to: {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    main()
