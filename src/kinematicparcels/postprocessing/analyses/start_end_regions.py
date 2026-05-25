from __future__ import annotations

import pandas as pd
import xarray as xr

from ..core import RegularGrid, classify_region_points
from ..core.regions import build_region_manager
from kinematicparcels.regions import RegionManager


def classify_start_end_regions(
    summary_df: pd.DataFrame,
    *,
    region_manager: RegionManager,
    how_many: str = "priority_max",
    priority_level: int | None = None,
    priority_mode: str = "exact",
    input_lon_mode: str = "-180_180",
    start_lon_col: str = "lon0",
    start_lat_col: str = "lat0",
    end_lon_col: str = "lonf",
    end_lat_col: str = "latf",
) -> pd.DataFrame:
    """
    Classify start and end positions into geographic regions.

    Returns the input summary dataframe enriched with:
    - start_region
    - start_numericLabel
    - end_region
    - end_numericLabel
    """
    required = [start_lon_col, start_lat_col, end_lon_col, end_lat_col]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise KeyError(f"Input summary dataframe missing required columns: {missing}")

    out = classify_region_points(
        summary_df,
        region_manager=region_manager,
        how_many=how_many,
        priority_level=priority_level,
        priority_mode=priority_mode,
        input_lon_mode=input_lon_mode,
        lon_col=start_lon_col,
        lat_col=start_lat_col,
        region_col="start_region",
        numeric_col="start_numericLabel",
        priority_col="start_priority",
    )
    out = classify_region_points(
        out,
        region_manager=region_manager,
        how_many=how_many,
        priority_level=priority_level,
        priority_mode=priority_mode,
        input_lon_mode=input_lon_mode,
        lon_col=end_lon_col,
        lat_col=end_lat_col,
        region_col="end_region",
        numeric_col="end_numericLabel",
        priority_col="end_priority",
    )
    return out


def _aggregate_region_grid(
    classified_summary_df: pd.DataFrame,
    *,
    grid: RegularGrid,
    value_col: str,
    priority_col: str,
    lon_col: str,
    lat_col: str,
    output_col: str,
) -> pd.DataFrame:
    binned = grid.assign_bins(
        classified_summary_df,
        lon_col=lon_col,
        lat_col=lat_col,
        drop_outside=True,
    )

    if binned.empty:
        return pd.DataFrame(
            columns=["lon_bin", "lat_bin", "lon_center", "lat_center", output_col]
        )

    group_cols = ["lon_bin", "lat_bin", "lon_center", "lat_center"]

    if priority_col in binned.columns:
        chosen = (
            binned.sort_values(
                group_cols + [priority_col, value_col],
                ascending=[True, True, True, True, False, True],
                na_position="last",
            )
            .drop_duplicates(subset=group_cols, keep="first")
            .reset_index(drop=True)
        )
        return chosen[group_cols + [value_col]].rename(columns={value_col: output_col})

    return grid.aggregate(
        classified_summary_df,
        value_col=value_col,
        agg="min",
        lon_col=lon_col,
        lat_col=lat_col,
        output_col=output_col,
    )


def compute_start_end_region_maps(
    classified_summary_df: pd.DataFrame,
    *,
    grid: RegularGrid,
    lon_col: str = "lon0",
    lat_col: str = "lat0",
) -> tuple[pd.DataFrame, xr.Dataset, pd.DataFrame, xr.Dataset]:
    """
    Build start and end region maps on the release grid.

    Returns:
    - start grid table
    - start dataset
    - end grid table
    - end dataset
    """
    required = [lon_col, lat_col, "start_numericLabel", "end_numericLabel"]
    missing = [c for c in required if c not in classified_summary_df.columns]
    if missing:
        raise KeyError(f"Input classified summary missing required columns: {missing}")

    start_grid_table = _aggregate_region_grid(
        classified_summary_df,
        grid=grid,
        value_col="start_numericLabel",
        priority_col="start_priority",
        lon_col=lon_col,
        lat_col=lat_col,
        output_col="start_numericLabel",
    )
    start_ds = grid.to_xarray(
        start_grid_table,
        value_col="start_numericLabel",
        dataset_name="start_numericLabel",
    )

    end_grid_table = _aggregate_region_grid(
        classified_summary_df,
        grid=grid,
        value_col="end_numericLabel",
        priority_col="end_priority",
        lon_col=lon_col,
        lat_col=lat_col,
        output_col="end_numericLabel",
    )
    end_ds = grid.to_xarray(
        end_grid_table,
        value_col="end_numericLabel",
        dataset_name="end_numericLabel",
    )

    return start_grid_table, start_ds, end_grid_table, end_ds