from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from kinematicparcels.postprocessing.plotting import plot_trajectories_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one trajectory from a trajectory-table Parquet file."
    )
    parser.add_argument("parquet_file", type=Path)
    parser.add_argument("trajectory_id", help='Trajectory ID, e.g. "77_m1"')
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_file)

    required = {"trajectory", "obs", "lon", "lat"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Parquet file is missing required columns: {sorted(missing)}"
        )

    # Convert to strings so IDs such as 77 and 77_m1 can be selected uniformly.
    trajectory_ids = df["trajectory"].astype(str)
    selected = df.loc[trajectory_ids == str(args.trajectory_id)].copy()

    if selected.empty:
        similar = sorted(
            identifier
            for identifier in trajectory_ids.unique()
            if str(args.trajectory_id) in identifier
        )

        message = f"Trajectory {args.trajectory_id!r} was not found."
        if similar:
            message += f" Similar IDs: {similar[:20]}"
        raise ValueError(message)

    selected = (
        selected.dropna(subset=["lon", "lat"])
        .sort_values("obs")
        .reset_index(drop=True)
    )

    if selected.empty:
        raise ValueError(
            f"Trajectory {args.trajectory_id!r} has no finite positions."
        )

    if args.output is None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.trajectory_id))
        output_path = args.parquet_file.parent / f"trajectory_{safe_id}.png"
    else:
        output_path = args.output

    plot_trajectories_map(
        selected,
        outpath=output_path,
        projection="PlateCarree",
        title=f"Trajectory {args.trajectory_id}",
        show_start=True,
        show_end=True,
        linewidth=1.5,
        alpha=0.9,
        color_by=None,
        max_group_member=None,
    )

    print(f"Rows plotted: {len(selected)}")
    print(f"Time range: {selected['time'].min()} -> {selected['time'].max()}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()