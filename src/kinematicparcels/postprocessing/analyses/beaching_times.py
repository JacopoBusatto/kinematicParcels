from __future__ import annotations

import pandas as pd
import xarray as xr

from ..core import RegularGrid


def compute_beaching_times(
    summary_df: pd.DataFrame,
    *,
    grid: RegularGrid,
    lon_col: str = "lon0",
    lat_col: str = "lat0",
    value_col: str = "lifetime_seconds",
    agg: str = "min",
    output_col: str = "beaching_time_seconds",
) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Compute beaching times on the release grid from particle summary.

    Parameters
    ----------
    summary_df
        Particle summary table.
    grid
        Release grid.
    lon_col, lat_col
        Initial-position columns used for mapping to the release grid.
    value_col
        Summary variable to map (default: lifetime_seconds).
    agg
        Aggregation over particles starting in the same pixel.
        Default is 'min'.
    output_col
        Output variable name.

    Returns
    -------
    tuple[pd.DataFrame, xr.Dataset]
        Aggregated grid table and 2D xarray dataset.
    """
    required = [lon_col, lat_col, value_col]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    grid_table = grid.aggregate(
        summary_df,
        value_col=value_col,
        agg=agg,
        lon_col=lon_col,
        lat_col=lat_col,
        output_col=output_col,
    )

    ds = grid.to_xarray(
        grid_table,
        value_col=output_col,
        dataset_name=output_col,
    )

    return grid_table, ds