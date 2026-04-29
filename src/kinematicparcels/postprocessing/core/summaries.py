from __future__ import annotations

import numpy as np
import pandas as pd


def _scalarize_identifier(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return _scalarize_identifier(value.item() if value.ndim == 0 else value.reshape(-1)[0])
        return tuple(_scalarize_identifier(v) for v in value.tolist())

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _scalarize_identifier(value[0])
        return tuple(_scalarize_identifier(v) for v in value)

    return value


def build_particle_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a per-particle summary table from a cleaned trajectory table.

    Parameters
    ----------
    df
        Clean trajectory table with at least:
        trajectory, obs, time, lon, lat
        optionally z.

    Returns
    -------
    pd.DataFrame
        One row per trajectory with columns:

        trajectory
        n_obs

        time0
        lon0
        lat0
        z0 (optional)

        timef
        lonf
        latf
        zf (optional)

        lifetime_seconds
    """

    required = ["trajectory", "obs", "time", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input dataframe missing required columns: {missing}"
        )

    if df.empty:
        return pd.DataFrame()

    has_z = "z" in df.columns

    summaries = []
    group_cols = ["trajectory"]
    has_group_member = "group_member" in df.columns
    if has_group_member:
        group_cols.append("group_member")
    metadata_cols = [col for col in ("circle_id", "group_id", "group_size") if col in df.columns]

    for group_key, g in df.groupby(group_cols, sort=False):

        g = g.sort_values("obs")

        first = g.iloc[0]
        last = g.iloc[-1]

        if has_group_member:
            traj, group_member = group_key
        else:
            traj = group_key
            group_member = None

        row = {
            "trajectory": _scalarize_identifier(traj),
            "n_obs": len(g),

            "time0": first["time"],
            "lon0": first["lon"],
            "lat0": first["lat"],

            "timef": last["time"],
            "lonf": last["lon"],
            "latf": last["lat"],
        }

        if has_group_member:
            row["group_member"] = _scalarize_identifier(group_member)

        for col in metadata_cols:
            row[col] = first[col]

        if has_z:
            row["z0"] = first["z"]
            row["zf"] = last["z"]

        lifetime = last["time"] - first["time"]
        row["lifetime_seconds"] = lifetime.total_seconds()

        summaries.append(row)

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df.sort_values(group_cols).reset_index(drop=True)

    return summary_df