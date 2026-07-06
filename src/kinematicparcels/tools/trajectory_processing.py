from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, *args, **kwargs):
        return iterable

from kinematicparcels.regions import ALL_REGIONS, RegionManager


@dataclass(frozen=True)
class ResampleConfig:
    frequency: str | None = None
    interpolate: str = "time"
    reference_time: pd.Timestamp | None = None
    shared_time: bool = False
    shift_start_to_reference: bool = False
    min_duration_days: float | None = None


@dataclass(frozen=True)
class RegionSelectionConfig:
    names_or_labels: tuple[str, ...] = ()
    selection_mode: str = "from_first_entry"
    input_lon_mode: str = "-180_180"


def resolve_resample_config(config: dict[str, Any]) -> ResampleConfig:
    raw = config.get("processing", {}).get("resample", {})
    enabled = bool(raw.get("enabled", raw.get("frequency") is not None))
    frequency = raw.get("frequency") if enabled else None
    if isinstance(frequency, str):
        frequency = frequency.lower()

    shared_time = bool(raw.get("shared_time", False)) if enabled else False
    shift_start_to_reference = bool(raw.get("shift_start_to_reference", False)) if enabled else False
    reference_time_raw = raw.get("reference_time") if enabled else None

    align_raw = raw.get("align_start", {}) or {}
    align_start_enabled = enabled and bool(align_raw.get("enabled", False))
    if align_start_enabled:
        if reference_time_raw is None:
            reference_time_raw = align_raw.get("start_time")
        if "shared_time" not in raw:
            shared_time = True
        if "shift_start_to_reference" not in raw:
            shift_start_to_reference = True

    reference_time: pd.Timestamp | None = None
    if reference_time_raw is not None:
        reference_time = pd.to_datetime(reference_time_raw, utc=True, errors="raise").tz_convert(None)

    if (shared_time or shift_start_to_reference) and reference_time is None:
        raise ValueError(
            "processing.resample.reference_time is required when shared_time or shift_start_to_reference is enabled"
        )
    if shared_time and frequency is None:
        raise ValueError("processing.resample.frequency is required when shared_time is true")

    min_duration_raw = raw.get("min_duration_days", None)
    min_duration_days: float | None = None
    if min_duration_raw is not None:
        min_duration_days = float(min_duration_raw)
        if not np.isfinite(min_duration_days) or min_duration_days < 0:
            raise ValueError("processing.resample.min_duration_days must be a finite number >= 0 or null")

    return ResampleConfig(
        frequency=frequency,
        interpolate=str(raw.get("interpolate", "time")),
        reference_time=reference_time,
        shared_time=shared_time,
        shift_start_to_reference=shift_start_to_reference,
        min_duration_days=min_duration_days,
    )


def resolve_region_selection_config(config: dict[str, Any]) -> RegionSelectionConfig:
    raw = config.get("processing", {}).get("regions", {})
    names_or_labels = tuple(raw.get("names_or_labels", []) or [])
    selection_mode = str(raw.get("selection_mode", "from_first_entry"))
    if selection_mode not in {"from_first_entry", "full_if_enters", "initial_inside"}:
        raise ValueError(
            "processing.regions.selection_mode must be one of: "
            "from_first_entry, full_if_enters, initial_inside"
        )

    input_lon_mode = str(raw.get("input_lon_mode", "-180_180"))
    if input_lon_mode in {"-180180", "180180"}:
        input_lon_mode = "-180_180"

    return RegionSelectionConfig(
        names_or_labels=names_or_labels,
        selection_mode=selection_mode,
        input_lon_mode=input_lon_mode,
    )


def normalize_trajectories(
    trajectories: list[pd.DataFrame],
    *,
    show_progress: bool = True,
) -> list[pd.DataFrame]:
    normalized: list[pd.DataFrame] = []
    trajectory_iterator = (
        tqdm(enumerate(trajectories), total=len(trajectories), desc="Normalizing trajectories", unit="traj")
        if show_progress and len(trajectories) > 1
        else enumerate(trajectories)
    )
    for trajectory_index, trajectory in trajectory_iterator:
        if trajectory.empty:
            continue
        current = trajectory.sort_values("time", kind="stable").reset_index(drop=True).copy()
        current["trajectory"] = trajectory_index
        current["obs"] = np.arange(len(current), dtype=np.int32)
        normalized.append(current)

    return normalized


def build_selected_region_manager(region_names_or_labels: tuple[str, ...]) -> RegionManager | None:
    if not region_names_or_labels:
        return None

    requested = set(region_names_or_labels)
    selected_regions = [
        region
        for region in ALL_REGIONS
        if region.label in requested or region.name in requested
    ]

    found_tokens = {
        token
        for region in selected_regions
        for token in (region.label, region.name)
        if isinstance(token, str)
    }
    missing = sorted(requested - found_tokens)
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown region labels/names requested in processing.regions: {missing_str}")

    return RegionManager(selected_regions)


def point_in_regions(
    lon: float,
    lat: float,
    *,
    region_manager: RegionManager,
    input_lon_mode: str,
) -> bool:
    return bool(
        region_manager.find_regions(
            lon,
            lat,
            howMany="first",
            input_lon_mode=input_lon_mode,
        )
    )


def apply_region_selection(
    trajectories: list[pd.DataFrame],
    *,
    config: RegionSelectionConfig,
) -> list[pd.DataFrame]:
    region_manager = build_selected_region_manager(config.names_or_labels)
    if region_manager is None:
        return trajectories

    selected_trajectories: list[pd.DataFrame] = []
    trajectory_iterator = (
        tqdm(trajectories, desc="Applying region selection", unit="traj")
        if len(trajectories) > 1
        else trajectories
    )

    for trajectory in trajectory_iterator:
        if trajectory.empty:
            continue

        mask_values = [
            point_in_regions(
                float(row.lon),
                float(row.lat),
                region_manager=region_manager,
                input_lon_mode=config.input_lon_mode,
            )
            for row in trajectory.itertuples(index=False)
        ]
        mask = np.asarray(mask_values, dtype=bool)

        if config.selection_mode == "initial_inside":
            if mask.size > 0 and mask[0]:
                selected_trajectories.append(trajectory.reset_index(drop=True))
            continue

        if not mask.any():
            continue

        if config.selection_mode == "full_if_enters":
            selected_trajectories.append(trajectory.reset_index(drop=True))
            continue

        first_hit_position = int(np.flatnonzero(mask)[0])
        selected_trajectories.append(trajectory.iloc[first_hit_position:].reset_index(drop=True))

    return selected_trajectories


def collapse_duplicate_times(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("time", kind="stable").reset_index(drop=True)
    if not ordered["time"].duplicated().any():
        return ordered

    return ordered.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)


def unwrap_longitudes(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=float)
    if values.size <= 1:
        return values.copy()

    unwrapped = values.copy()
    valid_idx = np.flatnonzero(np.isfinite(values))
    if valid_idx.size <= 1:
        return unwrapped

    split_points = np.flatnonzero(np.diff(valid_idx) > 1) + 1
    for chunk in np.split(valid_idx, split_points):
        if chunk.size <= 1:
            continue
        unwrapped[chunk] = np.rad2deg(np.unwrap(np.deg2rad(values[chunk])))

    return unwrapped


def wrap_longitudes(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def shift_trajectory_to_start(trajectory: pd.DataFrame, target_start: pd.Timestamp) -> pd.DataFrame:
    if trajectory.empty:
        return trajectory.copy()

    shifted = trajectory.copy()
    offset = target_start - shifted["time"].iloc[0]
    shifted.loc[:, "time"] = shifted["time"] + offset
    return shifted


def collapse_shared_time_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    if trajectory.empty:
        return trajectory.copy()

    valid_mask = trajectory["lon"].notna() & trajectory["lat"].notna()
    if not valid_mask.any():
        return trajectory.iloc[0:0].copy().reset_index(drop=True)

    return trajectory.loc[valid_mask].reset_index(drop=True).copy()


def trajectory_duration_days(trajectory: pd.DataFrame) -> float | None:
    if trajectory.empty or "time" not in trajectory.columns:
        return None

    time = pd.to_datetime(trajectory["time"], errors="coerce")
    valid = time.dropna()
    if valid.empty:
        return None

    duration_days = (valid.max() - valid.min()).total_seconds() / 86400.0
    if not np.isfinite(duration_days):
        return None
    return float(duration_days)


def filter_trajectories_by_min_duration(
    trajectories: list[pd.DataFrame],
    min_duration_days: float | None,
) -> tuple[list[pd.DataFrame], int]:
    if min_duration_days is None:
        return trajectories, 0
    if not np.isfinite(min_duration_days) or min_duration_days < 0:
        raise ValueError("min_duration_days must be a finite number >= 0 or null")

    kept: list[pd.DataFrame] = []
    dropped = 0
    for trajectory in trajectories:
        duration_days = trajectory_duration_days(trajectory)
        if duration_days is None or duration_days < min_duration_days:
            dropped += 1
            continue
        kept.append(trajectory)

    return kept, dropped


def resample_single_trajectory(
    df: pd.DataFrame,
    *,
    config: ResampleConfig,
    non_interpolated_columns: set[str],
    target_index: pd.DatetimeIndex | None = None,
    keep_full_target_index: bool = False,
) -> pd.DataFrame:
    if (not config.frequency and target_index is None) or df.empty:
        return df.reset_index(drop=True)

    ordered = collapse_duplicate_times(df).set_index("time").copy()
    min_time = ordered.index.min()
    max_time = ordered.index.max()
    if "lon" in ordered.columns:
        ordered.loc[:, "lon"] = unwrap_longitudes(ordered["lon"].to_numpy(dtype=float))

    if target_index is None:
        new_index = pd.date_range(min_time, max_time, freq=config.frequency)
        if len(new_index) == 0 or new_index[-1] != max_time:
            new_index = new_index.append(pd.DatetimeIndex([max_time]))
    else:
        if keep_full_target_index:
            new_index = target_index
        else:
            new_index = target_index[(target_index >= min_time) & (target_index <= max_time)]

    if len(new_index) == 0:
        return ordered.reset_index(drop=False).rename(columns={"index": "time"}).reset_index(drop=True)

    expanded = ordered.reindex(ordered.index.union(new_index)).sort_index()

    numeric_cols = expanded.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    interpolated_numeric_cols = [
        column for column in numeric_cols
        if column not in non_interpolated_columns
    ]
    if interpolated_numeric_cols:
        expanded.loc[:, interpolated_numeric_cols] = expanded.loc[:, interpolated_numeric_cols].interpolate(
            method=config.interpolate,
            limit_direction="both",
        )

    fill_only_cols = [column for column in expanded.columns if column not in interpolated_numeric_cols]
    if fill_only_cols:
        expanded.loc[:, fill_only_cols] = expanded.loc[:, fill_only_cols].ffill().bfill()

    result = expanded.loc[new_index].reset_index().rename(columns={"index": "time"})
    if keep_full_target_index:
        active_mask = (result["time"] >= min_time) & (result["time"] <= max_time)
        inactive_mask = ~active_mask
        value_columns = [column for column in result.columns if column != "time"]
        if value_columns:
            result.loc[inactive_mask, value_columns] = np.nan
    if "lon" in result.columns:
        result.loc[:, "lon"] = wrap_longitudes(result["lon"].to_numpy(dtype=float))
    return result.reset_index(drop=True)


def apply_resampling(
    trajectories: list[pd.DataFrame],
    *,
    config: ResampleConfig,
    non_interpolated_columns: set[str],
    show_progress: bool = True,
) -> list[pd.DataFrame]:
    working_trajectories = trajectories
    if config.shift_start_to_reference:
        if config.reference_time is None:
            raise ValueError("reference_time must be set when shift_start_to_reference is enabled")
        working_trajectories = [
            shift_trajectory_to_start(trajectory, config.reference_time)
            for trajectory in trajectories
        ]

    if config.shared_time:
        if config.reference_time is None:
            raise ValueError("reference_time must be set when shared_time is enabled")
        if not config.frequency:
            raise ValueError("processing.resample.frequency is required when shared_time is true")

        non_empty = [trajectory for trajectory in working_trajectories if not trajectory.empty]
        if not non_empty:
            return working_trajectories

        min_start_time = min(trajectory["time"].iloc[0] for trajectory in non_empty)
        max_end_time = max(trajectory["time"].iloc[-1] for trajectory in non_empty)
        common_index = pd.date_range(config.reference_time, max_end_time, freq=config.frequency)
        if len(common_index) == 0:
            common_index = pd.DatetimeIndex([config.reference_time])
        else:
            common_index = common_index[common_index >= min_start_time]
            if len(common_index) == 0:
                common_index = pd.DatetimeIndex([max_end_time])

        trajectory_iterator = (
            tqdm(working_trajectories, desc="Resampling trajectories", unit="traj")
            if show_progress and len(working_trajectories) > 1
            else working_trajectories
        )
        return [
            collapse_shared_time_trajectory(
                resample_single_trajectory(
                    trajectory,
                    config=config,
                    non_interpolated_columns=non_interpolated_columns,
                    target_index=common_index,
                    keep_full_target_index=True,
                )
            )
            for trajectory in trajectory_iterator
        ]

    if not config.frequency:
        return [trajectory.reset_index(drop=True) for trajectory in working_trajectories]

    if config.reference_time is not None:
        resampled: list[pd.DataFrame] = []
        trajectory_iterator = (
            tqdm(working_trajectories, desc="Resampling trajectories", unit="traj")
            if show_progress and len(working_trajectories) > 1
            else working_trajectories
        )
        for trajectory in trajectory_iterator:
            if trajectory.empty:
                resampled.append(trajectory.reset_index(drop=True))
                continue
            local_index = pd.date_range(config.reference_time, trajectory["time"].iloc[-1], freq=config.frequency)
            resampled.append(
                resample_single_trajectory(
                    trajectory,
                    config=config,
                    non_interpolated_columns=non_interpolated_columns,
                    target_index=local_index,
                )
            )
        return resampled

    if not show_progress or len(working_trajectories) <= 1:
        return [
            resample_single_trajectory(
                trajectory,
                config=config,
                non_interpolated_columns=non_interpolated_columns,
            )
            for trajectory in working_trajectories
        ]

    return [
        resample_single_trajectory(
            trajectory,
            config=config,
            non_interpolated_columns=non_interpolated_columns,
        )
        for trajectory in tqdm(working_trajectories, desc="Resampling trajectories", unit="traj")
    ]
