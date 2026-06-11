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


def _empty_transition_table(region_order: list[str]) -> pd.DataFrame:
    columns = ["age_days", "represented_fraction_total"] + [
        f"n_{origin}"
        for origin in region_order
    ] + [
        f"p_{origin}__{target}"
        for origin in region_order
        for target in region_order
    ]
    return pd.DataFrame(columns=columns)


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
        return _empty_transition_table(list(cfg.region_labels))

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
        return _empty_transition_table(region_order)

    particle_cols = _particle_group_cols(work)
    work = work.sort_values(particle_cols + ["obs"]).reset_index(drop=True)

    particle_meta = (
        work.groupby(particle_cols, sort=False)
        .agg(first_time=("time", "min"), last_time=("time", "max"))
        .reset_index()
    )
    particle_meta["lifetime"] = particle_meta["last_time"] - particle_meta["first_time"]

    min_life = pd.Timedelta(days=cfg.min_life_days)
    eligible_particles = particle_meta.loc[
        particle_meta["lifetime"] >= min_life,
        particle_cols + ["first_time", "lifetime"],
    ]
    work = work.merge(eligible_particles, on=particle_cols, how="inner")

    if work.empty:
        return _empty_transition_table(region_order)

    work["age"] = work["time"] - work["first_time"]

    if cfg.trimming_age_days is not None:
        trimming_age = pd.Timedelta(days=cfg.trimming_age_days)
        work = work.loc[work["age"] <= trimming_age].copy()

    if work.empty:
        return _empty_transition_table(region_order)

    work = work.loc[work["obs"] % cfg.time_step_stride == 0].copy()

    if work.empty:
        return _empty_transition_table(region_order)

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
        return _empty_transition_table(region_order)

    sampled_ages = pd.Index(sorted(pd.unique(work["age"])), name="age")
    denominators = origin_regions[origin_regions["start_region"].isin(region_order)]["start_region"].value_counts()

    counts = (
        work.loc[work["current_region"].isin(region_order)]
        .groupby(["age", "start_region", "current_region"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )

    if counts.empty:
        pivot = pd.DataFrame(index=sampled_ages)
    else:
        pivot = counts.pivot_table(
            index="age",
            columns=["start_region", "current_region"],
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reindex(index=sampled_ages, fill_value=0)

    ordered_columns = pd.MultiIndex.from_product(
        [region_order, region_order],
        names=["start_region", "current_region"],
    )
    pivot = pivot.reindex(columns=ordered_columns, fill_value=0)

    probability = pd.DataFrame(index=sampled_ages)
    represented_fraction_total = pd.Series(0.0, index=sampled_ages, dtype=float)
    total_denominator = int(sum(int(denominators.get(origin, 0)) for origin in region_order))

    for origin in region_order:
        denominator = int(denominators.get(origin, 0))
        origin_represented_fraction = pd.Series(0.0, index=sampled_ages, dtype=float)
        for target in region_order:
            column_name = f"p_{origin}__{target}"
            if denominator == 0:
                probability[column_name] = np.nan
            else:
                probability[column_name] = pivot[(origin, target)].astype(float) / float(denominator)
                origin_represented_fraction = origin_represented_fraction + probability[column_name].fillna(0.0)

        probability[f"n_{origin}"] = denominator
        if denominator > 0:
            represented_fraction_total = represented_fraction_total + float(denominator) * origin_represented_fraction

    if total_denominator == 0:
        probability["represented_fraction_total"] = np.nan
    else:
        probability["represented_fraction_total"] = represented_fraction_total / float(total_denominator)

    probability = probability.reset_index()
    probability["age_days"] = probability["age"].dt.total_seconds() / 86400.0
    probability = probability.drop(columns=["age"])
    ordered_probability_columns = [
        f"p_{origin}__{target}"
        for origin in region_order
        for target in region_order
    ]
    ordered_count_columns = [f"n_{origin}" for origin in region_order]
    probability = probability[
        ["age_days", "represented_fraction_total"]
        + ordered_count_columns
        + ordered_probability_columns
    ]
    return probability