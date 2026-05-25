from __future__ import annotations

import numpy as np
import pandas as pd

from kinematicparcels.regions import ALL_REGIONS, RegionManager


_PRIORITY_BY_LABEL = {r.label: float(r.priority) for r in ALL_REGIONS}


def _normalize_region_result(result) -> tuple[str | None, float, float]:
    """
    Normalize RegionManager.find_regions output to:
    (label, numeric_label, priority)

    Supported outputs:
    - None
    - dict with keys 'label' and 'numericLabel'
    - list of dicts
    - object-like result with .label and .numericLabel
    """
    if result is None:
        return None, np.nan, np.nan

    if isinstance(result, list):
        if len(result) == 0:
            return None, np.nan, np.nan
        result = result[0]

    if isinstance(result, dict):
        label = result.get("label", None)
        numeric_label = result.get("numericLabel", np.nan)
        priority = result.get("priority", _PRIORITY_BY_LABEL.get(label, np.nan))
        return (
            label,
            float(numeric_label) if numeric_label is not None else np.nan,
            float(priority) if priority is not None else np.nan,
        )

    label = getattr(result, "label", None)
    numeric_label = getattr(result, "numericLabel", np.nan)
    priority = getattr(result, "priority", _PRIORITY_BY_LABEL.get(label, np.nan))

    return (
        label,
        float(numeric_label) if numeric_label is not None else np.nan,
        float(priority) if priority is not None else np.nan,
    )


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
        raise ValueError("No regions selected for region analysis.")

    return RegionManager(selected)


def classify_region_points(
    points_df: pd.DataFrame,
    *,
    region_manager: RegionManager,
    how_many: str = "priority_max",
    priority_level: int | None = None,
    priority_mode: str = "exact",
    input_lon_mode: str = "-180_180",
    lon_col: str = "lon",
    lat_col: str = "lat",
    region_col: str = "region",
    numeric_col: str = "numericLabel",
    priority_col: str = "priority",
) -> pd.DataFrame:
    """
    Classify arbitrary lon/lat rows into geographic regions.
    """
    required = [lon_col, lat_col]
    missing = [c for c in required if c not in points_df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    out = points_df.copy()

    labels: list[str | None] = []
    numeric_labels: list[float] = []
    priorities: list[float] = []

    for row in out.itertuples(index=False):
        result = region_manager.find_regions(
            getattr(row, lon_col),
            getattr(row, lat_col),
            howMany=how_many,
            priority_level=priority_level,
            priority_mode=priority_mode,
            input_lon_mode=input_lon_mode,
        )
        label, numeric_label, priority = _normalize_region_result(result)
        labels.append(label)
        numeric_labels.append(numeric_label)
        priorities.append(priority)

    out[region_col] = labels
    out[numeric_col] = numeric_labels
    out[priority_col] = priorities
    return out


__all__ = [
    "build_region_manager",
    "classify_region_points",
]