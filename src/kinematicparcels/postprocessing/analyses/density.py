from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..core.gridding import RegularGrid


def compute_time_density(
    df: pd.DataFrame,
    *,
    grid: RegularGrid,
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    normalize_active: bool = True,
    normalize_total: bool = True,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Compute time-dependent particle density on a regular lon/lat grid.

    Parameters
    ----------
    df
        Canonical trajectory table.
    grid
        Regular analysis grid.
    lon_col, lat_col, time_col
        Column names used for spatial and temporal coordinates.
    normalize_active
        If True, compute particle_fraction_active:
        particle_count / N_active(time)
    normalize_total
        If True, compute particle_fraction_total:
        particle_count / N_total

    Returns
    -------
    tuple[pd.DataFrame, xr.Dataset]
        - aggregated density table
        - xarray Dataset with dimensions (time, lat, lon)
    """
    required = [lon_col, lat_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    if df.empty:
        empty_ds = xr.Dataset(
            coords={
                "time": np.array([], dtype="datetime64[ns]"),
                "lat": grid.lat_centers,
                "lon": grid.lon_centers,
            }
        )
        return pd.DataFrame(), empty_ds

    # Ensure datetime-like time column
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col])

    # Assign grid bins
    binned = grid.assign_bins(
        work,
        lon_col=lon_col,
        lat_col=lat_col,
        drop_outside=True,
    )

    if binned.empty:
        empty_ds = xr.Dataset(
            coords={
                "time": np.array([], dtype="datetime64[ns]"),
                "lat": grid.lat_centers,
                "lon": grid.lon_centers,
            }
        )
        return pd.DataFrame(), empty_ds

    # Count particles per (time, lon_bin, lat_bin)
    grouped = (
        binned.groupby(
            [time_col, "lon_bin", "lat_bin"],
            sort=True,
            observed=False,
        )
        .size()
        .reset_index(name="particle_count")
    )

    # Reconstruct exact pixel centers from integer bins
    grouped["lon_center"] = grid.lon_min + (grouped["lon_bin"] + 0.5) * grid.dlon
    grouped["lat_center"] = grid.lat_min + (grouped["lat_bin"] + 0.5) * grid.dlat

    # Active particle count at each time
    active_counts = (
        work.groupby(time_col, sort=True, observed=False)
        .size()
        .rename("n_active")
        .reset_index()
    )

    grouped = grouped.merge(active_counts, on=time_col, how="left")

    # Total number of trajectories in the dataset
    # Use trajectory column if available, otherwise fallback to row-based unique count at t0 is impossible.
    if "trajectory" in work.columns:
        n_total = work["trajectory"].nunique()
    else:
        n_total = np.nan

    if normalize_active:
        grouped["particle_fraction_active"] = (
            grouped["particle_count"] / grouped["n_active"]
        )

    if normalize_total and not np.isnan(n_total) and n_total > 0:
        grouped["particle_fraction_total"] = grouped["particle_count"] / n_total

    # Build xarray Dataset
    time_values = np.sort(grouped[time_col].unique())
    nt = len(time_values)

    count_data = np.full((nt, grid.nlat, grid.nlon), np.nan, dtype=float)
    active_frac_data = np.full((nt, grid.nlat, grid.nlon), np.nan, dtype=float)
    total_frac_data = np.full((nt, grid.nlat, grid.nlon), np.nan, dtype=float)

    time_index = {t: i for i, t in enumerate(time_values)}

    for row in grouped.itertuples(index=False):
        t_idx = time_index[getattr(row, time_col)]
        j = int(row.lat_bin)
        i = int(row.lon_bin)

        count_data[t_idx, j, i] = float(row.particle_count)

        if normalize_active and hasattr(row, "particle_fraction_active"):
            active_frac_data[t_idx, j, i] = float(row.particle_fraction_active)

        if normalize_total and hasattr(row, "particle_fraction_total"):
            total_frac_data[t_idx, j, i] = float(row.particle_fraction_total)

    data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray]] = {
        "particle_count": (("time", "lat", "lon"), count_data),
    }

    if normalize_active:
        data_vars["particle_fraction_active"] = (
            ("time", "lat", "lon"),
            active_frac_data,
        )

    if normalize_total and not np.isnan(n_total) and n_total > 0:
        data_vars["particle_fraction_total"] = (
            ("time", "lat", "lon"),
            total_frac_data,
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time_values,
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs={
            "grid_type": "regular_lonlat",
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "dlon": grid.dlon,
            "dlat": grid.dlat,
        },
    )

    return grouped, ds