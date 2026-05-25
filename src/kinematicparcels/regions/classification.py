from __future__ import annotations

import pandas as pd

from .core import RegionManager


def classify_trajectories(
    df,
    region_manager: RegionManager,
    id_col='id',
    x_col='X',
    y_col='Y',
    howMany="first",
    priority_level=None,
    priority_mode="exact",
    input_lon_mode="-180_180",
):
    """
    Classify start and end regions for each trajectory in a DataFrame.
    """
    results = []

    grouped = df.groupby(id_col)

    for traj_id, group in grouped:
        group = group.sort_index()

        start_x, start_y = group.iloc[0][x_col], group.iloc[0][y_col]
        end_x, end_y = group.iloc[-1][x_col], group.iloc[-1][y_col]

        start_region = region_manager.find_regions(
            float(start_x),
            float(start_y),
            howMany=howMany,
            priority_level=priority_level,
            priority_mode=priority_mode,
            input_lon_mode=input_lon_mode,
        )

        end_region = region_manager.find_regions(
            float(end_x),
            float(end_y),
            howMany=howMany,
            priority_level=priority_level,
            priority_mode=priority_mode,
            input_lon_mode=input_lon_mode,
        )

        start_label = start_region["label"] if start_region else None
        start_numeric = start_region["numericLabel"] if start_region else None
        end_label = end_region["label"] if end_region else None
        end_numeric = end_region["numericLabel"] if end_region else None

        results.append({
            id_col: traj_id,
            'start_region': start_label,
            'start_numericLabel': start_numeric,
            'end_region': end_label,
            'end_numericLabel': end_numeric,
        })

    return pd.DataFrame(results)


def classify_full_trajectory(
    df,
    region_manager: RegionManager,
    id_col='id',
    x_col='X',
    y_col='Y',
    howMany="first",
    priority_level=None,
    priority_mode="exact",
    input_lon_mode="-180_180",
):
    """
    Classify the region of each point in a trajectory.
    """
    results = []

    grouped = df.groupby(id_col)

    for traj_id, group in grouped:
        group = group.sort_index()

        for idx, row in group.iterrows():
            point_x, point_y = row[x_col], row[y_col]

            region = region_manager.find_regions(
                float(point_x),
                float(point_y),
                howMany=howMany,
                priority_level=priority_level,
                priority_mode=priority_mode,
                input_lon_mode=input_lon_mode,
            )

            if region:
                region_label = region["label"]
                numeric_label = region["numericLabel"]
            else:
                region_label = None
                numeric_label = None

            item = {
                id_col: traj_id,
                'point_id': idx,
                'region_label': region_label,
                'numericLabel': numeric_label,
            }

            if "age" in row.index:
                item["age"] = row["age"]

            results.append(item)

    return pd.DataFrame(results)


__all__ = [
    "classify_trajectories",
    "classify_full_trajectory",
]
