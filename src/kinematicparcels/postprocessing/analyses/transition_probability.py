from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config.models import TransitionProbabilityConfig
from ..core.regions import build_region_manager, classify_region_points


def _particle_group_cols(df: pd.DataFrame) -> list[str]:
    cols = ["trajectory"]
    if "group_member" in df.columns:
        cols.append("group_member")
    return cols


def _apply_isolated_region_filter(labels: pd.Series) -> pd.Series:
    if len(labels) < 3:
        return labels

    original = labels.tolist()
    filtered = list(original)

    for idx in range(1, len(original) - 1):
        prev_label = original[idx - 1]
        curr_label = original[idx]
        next_label = original[idx + 1]

        if (
            prev_label is not None
            and curr_label is not None
            and next_label is not None
            and curr_label != prev_label
            and prev_label == next_label
        ):
            filtered[idx] = prev_label

    return pd.Series(filtered, index=labels.index, dtype=labels.dtype)


def _resolve_selected_regions(cfg: TransitionProbabilityConfig):
    region_manager = build_region_manager(region_labels=cfg.region_labels)
    selected_by_label = {region.label: region for region in region_manager.get_regions()}
    missing = [label for label in cfg.region_labels if label not in selected_by_label]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Unknown region labels requested in transition_probability.region_labels: "
            f"{missing_str}"
        )

    ordered_regions = [selected_by_label[label] for label in cfg.region_labels]
    priorities = {region.priority for region in ordered_regions}
    if len(priorities) > 1:
        warnings.warn(
            "Selected transition-probability regions do not share the same priority level; "
            "overlaps may produce ambiguous classifications.",
            UserWarning,
            stacklevel=2,
        )

    return region_manager, ordered_regions


def compute_transition_probability(
    df: pd.DataFrame,
    *,
    cfg: TransitionProbabilityConfig,
    region_manager=None,
) -> pd.DataFrame:
    """
    Compute the time-dependent transition probability matrix between regions.
    """
    required = ["trajectory", "obs", "time", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    if df.empty:
        columns = ["time"] + [
            f"p_{origin}__{target}"
            for origin in cfg.region_labels
            for target in cfg.region_labels
        ]
        return pd.DataFrame(columns=columns)

    if region_manager is None:
        region_manager, ordered_regions = _resolve_selected_regions(cfg)
    else:
        selected_by_label = {region.label: region for region in region_manager.get_regions()}
        missing = [label for label in cfg.region_labels if label not in selected_by_label]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "Unknown region labels requested in transition_probability.region_labels: "
                f"{missing_str}"
            )
        ordered_regions = [selected_by_label[label] for label in cfg.region_labels]
        priorities = {region.priority for region in ordered_regions}
        if len(priorities) > 1:
            warnings.warn(
                "Selected transition-probability regions do not share the same priority level; "
                "overlaps may produce ambiguous classifications.",
                UserWarning,
                stacklevel=2,
            )

    region_order = [region.label for region in ordered_regions]

    work = df.copy()
    if cfg.max_group_member is not None and "group_member" in work.columns:
        work = work.loc[work["group_member"] <= cfg.max_group_member].copy()

    if work.empty:
        columns = ["time"] + [
            f"p_{origin}__{target}"
            for origin in region_order
            for target in region_order
        ]
        return pd.DataFrame(columns=columns)

    particle_cols = _particle_group_cols(work)
    work = work.sort_values(particle_cols + ["obs"]).reset_index(drop=True)
    work = work.loc[work["obs"] % cfg.time_frequency == 0].copy()

    if work.empty:
        columns = ["time"] + [
            f"p_{origin}__{target}"
            for origin in region_order
            for target in region_order
        ]
        return pd.DataFrame(columns=columns)

    work = classify_region_points(
        work,
        region_manager=region_manager,
        how_many=cfg.how_many,
        priority_level=cfg.priority_level,
        priority_mode=cfg.priority_mode,
        input_lon_mode=cfg.input_lon_mode,
        lon_col="lon",
        lat_col="lat",
        region_col="current_region",
        numeric_col="current_numericLabel",
        priority_col="current_priority",
    )

    if cfg.filter_isolated:
        work["current_region"] = (
            work.groupby(particle_cols, sort=False)["current_region"]
            .transform(_apply_isolated_region_filter)
        )

    origin_regions = (
        work.sort_values(particle_cols + ["obs"])
        .drop_duplicates(subset=particle_cols, keep="first")
        [particle_cols + ["current_region"]]
        .rename(columns={"current_region": "start_region"})
        .reset_index(drop=True)
    )

    work = work.merge(origin_regions, on=particle_cols, how="left")
    work = work.loc[work["start_region"].isin(region_order)].copy()

    if work.empty:
        columns = ["time"] + [
            f"p_{origin}__{target}"
            for origin in region_order
            for target in region_order
        ]
        return pd.DataFrame(columns=columns)

    sampled_times = pd.Index(pd.unique(work["time"]), name="time")
    denominators = origin_regions[origin_regions["start_region"].isin(region_order)]["start_region"].value_counts()

    counts = (
        work.loc[work["current_region"].isin(region_order)]
        .groupby(["time", "start_region", "current_region"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )

    if counts.empty:
        pivot = pd.DataFrame(index=sampled_times)
    else:
        pivot = counts.pivot_table(
            index="time",
            columns=["start_region", "current_region"],
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reindex(index=sampled_times, fill_value=0)

    ordered_columns = pd.MultiIndex.from_product(
        [region_order, region_order],
        names=["start_region", "current_region"],
    )
    pivot = pivot.reindex(columns=ordered_columns, fill_value=0)

    probability = pd.DataFrame(index=sampled_times)
    for origin in region_order:
        denominator = int(denominators.get(origin, 0))
        for target in region_order:
            column_name = f"p_{origin}__{target}"
            if denominator == 0:
                probability[column_name] = np.nan
            else:
                probability[column_name] = pivot[(origin, target)].astype(float) / float(denominator)

    probability = probability.reset_index()
    return probability