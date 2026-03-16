from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..core import RegularGrid
from kinematicparcels.utilities.geographicalRegions import ALL_REGIONS, RegionManager


def _normalize_region_result(result) -> tuple[str | None, float]:
    """
    Normalize RegionManager.find_regions output to:
    (label, numeric_label)

    Supported outputs:
    - None
    - dict with keys 'label' and 'numericLabel'
    - list of dicts
    - object-like result with .label and .numericLabel
    """
    if result is None:
        return None, np.nan

    if isinstance(result, list):
        if len(result) == 0:
            return None, np.nan
        result = result[0]

    if isinstance(result, dict):
        label = result.get("label", None)
        numeric_label = result.get("numericLabel", np.nan)
        return label, float(numeric_label) if numeric_label is not None else np.nan

    label = getattr(result, "label", None)
    numeric_label = getattr(result, "numericLabel", np.nan)

    return label, float(numeric_label) if numeric_label is not None else np.nan


def build_region_manager(
    *,
    region_labels: tuple[str, ...] | None = None,
) -> RegionManager:
    """
    Build a RegionManager from ALL_REGIONS, optionally filtering by label.
    """
    if region_labels is None:
        selected = ALL_REGIONS
    else:
        wanted = set(region_labels)
        selected = [r for r in ALL_REGIONS if r.label in wanted]

    if len(selected) == 0:
        raise ValueError("No regions selected for start/end region analysis.")

    return RegionManager(selected)


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

    out = summary_df.copy()

    start_regions: list[str | None] = []
    start_numeric: list[float] = []
    end_regions: list[str | None] = []
    end_numeric: list[float] = []

    for row in out.itertuples(index=False):
        start_result = region_manager.find_regions(
            getattr(row, start_lon_col),
            getattr(row, start_lat_col),
            howMany=how_many,
            priority_level=priority_level,
            priority_mode=priority_mode,
            input_lon_mode=input_lon_mode,
        )
        end_result = region_manager.find_regions(
            getattr(row, end_lon_col),
            getattr(row, end_lat_col),
            howMany=how_many,
            priority_level=priority_level,
            priority_mode=priority_mode,
            input_lon_mode=input_lon_mode,
        )

        s_label, s_num = _normalize_region_result(start_result)
        e_label, e_num = _normalize_region_result(end_result)

        start_regions.append(s_label)
        start_numeric.append(s_num)
        end_regions.append(e_label)
        end_numeric.append(e_num)

    out["start_region"] = start_regions
    out["start_numericLabel"] = start_numeric
    out["end_region"] = end_regions
    out["end_numericLabel"] = end_numeric

    return out


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

    start_grid_table = grid.aggregate(
        classified_summary_df,
        value_col="start_numericLabel",
        agg="min",
        lon_col=lon_col,
        lat_col=lat_col,
        output_col="start_numericLabel",
    )
    start_ds = grid.to_xarray(
        start_grid_table,
        value_col="start_numericLabel",
        dataset_name="start_numericLabel",
    )

    end_grid_table = grid.aggregate(
        classified_summary_df,
        value_col="end_numericLabel",
        agg="min",
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