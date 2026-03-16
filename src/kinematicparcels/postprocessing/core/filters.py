from __future__ import annotations

from typing import Iterable

import pandas as pd


def filter_by_bbox(
    df: pd.DataFrame,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pd.DataFrame:
    """
    Filter rows within a longitude/latitude bounding box.
    """
    required = [lon_col, lat_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    mask = (
        (df[lon_col] >= lon_min)
        & (df[lon_col] <= lon_max)
        & (df[lat_col] >= lat_min)
        & (df[lat_col] <= lat_max)
    )
    return df.loc[mask].copy()


def filter_by_time_range(
    df: pd.DataFrame,
    *,
    time_start: str | pd.Timestamp | None = None,
    time_end: str | pd.Timestamp | None = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Filter rows within a time range [time_start, time_end].
    """
    if time_col not in df.columns:
        raise KeyError(f"Input dataframe missing required column: '{time_col}'")

    out = df.copy()

    if time_start is not None:
        time_start = pd.to_datetime(time_start)
        out = out.loc[out[time_col] >= time_start]

    if time_end is not None:
        time_end = pd.to_datetime(time_end)
        out = out.loc[out[time_col] <= time_end]

    return out.copy()


def filter_by_trajectories(
    df: pd.DataFrame,
    trajectories: Iterable,
    *,
    trajectory_col: str = "trajectory",
) -> pd.DataFrame:
    """
    Keep only rows belonging to the selected trajectories.
    """
    if trajectory_col not in df.columns:
        raise KeyError(f"Input dataframe missing required column: '{trajectory_col}'")

    traj_set = set(trajectories)
    return df.loc[df[trajectory_col].isin(traj_set)].copy()


def filter_by_z_range(
    df: pd.DataFrame,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
    z_col: str = "z",
) -> pd.DataFrame:
    """
    Filter rows within a vertical range [z_min, z_max].
    """
    if z_col not in df.columns:
        raise KeyError(f"Input dataframe missing required column: '{z_col}'")

    out = df.copy()

    if z_min is not None:
        out = out.loc[out[z_col] >= z_min]

    if z_max is not None:
        out = out.loc[out[z_col] <= z_max]

    return out.copy()