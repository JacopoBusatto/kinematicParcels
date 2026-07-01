from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, *args, **kwargs):
        return iterable

from kinematicparcels.tools.argo_to_zarr import (
    _apply_region_filter,
    _apply_resampling,
    _resolve_input_files,
    _resolve_output_path,
    _resolve_region_filter_config,
    _resolve_resample_config,
    load_config,
)
from kinematicparcels.tools.zarr_writer import (
    DEFAULT_TRAJECTORY_DATASET_ATTRS,
    build_dataset_from_trajectories,
    build_zarr_encoding,
)


DEFAULT_COLUMNS = {
    "platform_code": "ID",
    "time": "time",
    "lat": "latitude",
    "lon": "longitude",
    "drogue_lost_time": "drogue_lost_date",
    "drogue_length": "DrogueLength",
}

DEFAULT_DATASET_ATTRS = {
    **DEFAULT_TRAJECTORY_DATASET_ATTRS,
    "source": "Drifter CSV conversion",
}

TRAJECTORY_LEVEL_COLUMNS = {"platform_code"}


@dataclass(frozen=True)
class DrogueConfig:
    clip_after_loss: bool = True
    minimum_length_m: float | None = None


@dataclass(frozen=True)
class SegmentConfig:
    mode: str = "ignore"
    step_hours: float = 6.0
    tolerance_minutes: float = 30.0


@dataclass
class DrifterConversionSummary:
    input_files: int = 0
    buffered_platforms: int = 0
    buffered_observations: int = 0
    segment_mode: str = ""
    segmented_trajectories: int = 0
    segmented_platforms: int = 0
    segmented_observations: int = 0
    region_names_or_labels: tuple[str, ...] = ()
    region_input_trajectories: int = 0
    region_input_platforms: int = 0
    region_kept_trajectories: int = 0
    region_kept_platforms: int = 0
    resample_frequency: str | None = None
    resample_dropped_empty: int = 0
    final_trajectories: int = 0
    final_platforms: int = 0
    final_observations: int = 0
    output_path: str = ""
    output_counts: dict[str, int] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert drifter CSV files into a Parcels-compatible trajectory Zarr dataset."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the drifter conversion YAML configuration file.",
    )
    return parser


def _resolve_columns(config: dict[str, Any]) -> dict[str, str]:
    columns = dict(DEFAULT_COLUMNS)
    columns.update(config.get("columns", {}))
    return columns


def _resolve_drogue_config(config: dict[str, Any]) -> DrogueConfig:
    raw = config.get("processing", {}).get("drogue", {})
    minimum_length_raw = raw.get("minimum_length_m", raw.get("min_length_m"))
    minimum_length_m: float | None = None
    if minimum_length_raw is not None:
        minimum_length_m = float(minimum_length_raw)
        if minimum_length_m < 0.0:
            raise ValueError("processing.drogue.minimum_length_m must be >= 0 when provided")

    return DrogueConfig(
        clip_after_loss=bool(raw.get("clip_after_loss", True)),
        minimum_length_m=minimum_length_m,
    )


def _resolve_segment_config(config: dict[str, Any]) -> SegmentConfig:
    raw = config.get("processing", {}).get("segment", {})
    mode = raw.get("mode", "ignore")
    aliases = {
        "split": "split_as_new",
        "separate": "split_as_new",
        "irregular": "ignore",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"ignore", "longest", "split_as_new"}:
        raise ValueError("processing.segment.mode must be one of: ignore, longest, split_as_new")

    step_hours = float(
        raw.get(
            "step_hours",
            raw.get("time_step_hours", raw.get("expected_step_hours", raw.get("resolution_hours", 6.0))),
        )
    )
    tolerance_minutes = float(raw.get("tolerance_minutes", 30.0))
    if step_hours <= 0.0:
        raise ValueError("processing.segment.step_hours must be > 0")
    if tolerance_minutes < 0.0:
        raise ValueError("processing.segment.tolerance_minutes must be >= 0")

    return SegmentConfig(
        mode=mode,
        step_hours=step_hours,
        tolerance_minutes=tolerance_minutes,
    )


def _coerce_time_series(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    return times.dt.tz_convert(None)


def _parse_drogue_length_m(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _resolve_csv_read_columns(
    csv_path: Path,
    *,
    columns: dict[str, str],
    drogue_config: DrogueConfig,
) -> list[str]:
    available_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    required = [
        columns["platform_code"],
        columns["time"],
        columns["lat"],
        columns["lon"],
    ]
    missing = [column for column in required if column not in available_columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required drifter columns in {csv_path}: {missing_str}")

    read_columns = list(required)
    loss_column = columns.get("drogue_lost_time")
    length_column = columns.get("drogue_length")

    if drogue_config.clip_after_loss:
        if not isinstance(loss_column, str) or loss_column not in available_columns:
            raise KeyError(
                f"Missing required drifter drogue loss column in {csv_path}: {loss_column}"
            )
        read_columns.append(loss_column)
    elif isinstance(loss_column, str) and loss_column in available_columns:
        read_columns.append(loss_column)

    if drogue_config.minimum_length_m is not None:
        if not isinstance(length_column, str) or length_column not in available_columns:
            raise KeyError(
                f"Missing required drifter drogue length column in {csv_path}: {length_column}"
            )
        read_columns.append(length_column)
    elif isinstance(length_column, str) and length_column in available_columns:
        read_columns.append(length_column)

    return list(dict.fromkeys(read_columns))


def _read_drifter_points(
    csv_path: Path,
    *,
    columns: dict[str, str],
    drogue_config: DrogueConfig,
) -> pd.DataFrame:
    read_columns = _resolve_csv_read_columns(
        csv_path,
        columns=columns,
        drogue_config=drogue_config,
    )
    df = pd.read_csv(csv_path, usecols=read_columns)

    time_col = columns["time"]
    lat_col = columns["lat"]
    lon_col = columns["lon"]
    platform_col = columns["platform_code"]
    loss_col = columns.get("drogue_lost_time")
    length_col = columns.get("drogue_length")

    df = df.copy()
    df[time_col] = _coerce_time_series(df[time_col])
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    platform_codes = pd.to_numeric(df[platform_col], errors="coerce")

    if isinstance(loss_col, str) and loss_col in df.columns:
        df[loss_col] = _coerce_time_series(df[loss_col])
    if isinstance(length_col, str) and length_col in df.columns:
        df[length_col] = _parse_drogue_length_m(df[length_col])

    valid_required = (
        platform_codes.notna()
        & df[time_col].notna()
        & df[lat_col].notna()
        & df[lon_col].notna()
    )
    filtered = df.loc[valid_required].copy()
    if filtered.empty:
        raise ValueError(f"No valid drifter rows were found in {csv_path}")

    filtered_platform_codes = pd.to_numeric(filtered[platform_col], errors="coerce")
    if not np.allclose(
        filtered_platform_codes.to_numpy(dtype=float),
        np.round(filtered_platform_codes.to_numpy(dtype=float)),
    ):
        raise ValueError(f"Non-integer platform_code values found in {csv_path}")

    filtered[platform_col] = filtered_platform_codes.to_numpy(dtype=np.int64)

    rename_map = {
        platform_col: "platform_code",
        time_col: "time",
        lat_col: "lat",
        lon_col: "lon",
    }
    if isinstance(loss_col, str) and loss_col in filtered.columns:
        rename_map[loss_col] = "drogue_lost_time"
    if isinstance(length_col, str) and length_col in filtered.columns:
        rename_map[length_col] = "drogue_length_m"

    filtered = filtered.rename(columns=rename_map)
    filtered["z"] = 0.0

    keep_columns = ["platform_code", "time", "lat", "lon", "z"]
    if "drogue_lost_time" in filtered.columns:
        keep_columns.append("drogue_lost_time")
    if "drogue_length_m" in filtered.columns:
        keep_columns.append("drogue_length_m")

    return filtered[keep_columns].sort_values(["platform_code", "time", "lat", "lon"], kind="stable").reset_index(drop=True)


def _merge_platform_points(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0].sort_values(["platform_code", "time", "lat", "lon"], kind="stable").reset_index(drop=True)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["platform_code", "time", "lat", "lon"], kind="stable")
    merged = merged.drop_duplicates(subset=["platform_code", "time", "lat", "lon"], keep="last")
    return merged.reset_index(drop=True)


def _trajectory_platform_code(trajectory: pd.DataFrame) -> int | None:
    if trajectory.empty or "platform_code" not in trajectory.columns:
        return None
    value = pd.to_numeric(pd.Series([trajectory["platform_code"].iloc[0]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return int(value)


def _count_platforms(trajectories: list[pd.DataFrame]) -> int:
    platform_codes = {
        platform_code
        for trajectory in trajectories
        if (platform_code := _trajectory_platform_code(trajectory)) is not None
    }
    return len(platform_codes)


def _count_observations(trajectories: list[pd.DataFrame]) -> int:
    return int(sum(len(trajectory) for trajectory in trajectories))


def _normalize_trajectories(trajectories: list[pd.DataFrame]) -> list[pd.DataFrame]:
    normalized: list[pd.DataFrame] = []
    trajectory_iterator = (
        tqdm(enumerate(trajectories), total=len(trajectories), desc="Normalizing trajectories", unit="traj")
        if len(trajectories) > 1
        else enumerate(trajectories)
    )
    for trajectory_index, trajectory in trajectory_iterator:
        current = trajectory.sort_values("time", kind="stable").reset_index(drop=True).copy()
        current["trajectory"] = trajectory_index
        current["obs"] = np.arange(len(current), dtype=np.int32)
        normalized.append(current)

    return normalized


def _resolve_platform_drogue_length_m(df: pd.DataFrame) -> float | None:
    if "drogue_length_m" not in df.columns:
        return None

    valid = df["drogue_length_m"].dropna()
    if valid.empty:
        return None
    return float(valid.iloc[0])


def _clip_after_drogue_loss(df: pd.DataFrame, *, enabled: bool) -> pd.DataFrame:
    if not enabled or "drogue_lost_time" not in df.columns:
        return df.copy()

    loss_times = df["drogue_lost_time"].dropna()
    if loss_times.empty:
        return df.copy()

    loss_time = pd.Timestamp(loss_times.iloc[0])
    clipped = df.loc[df["time"] < loss_time].copy()
    return clipped.reset_index(drop=True)


def _segment_duration_days(df: pd.DataFrame) -> float:
    if len(df) <= 1:
        return 0.0
    delta = df["time"].iloc[-1] - df["time"].iloc[0]
    return float(delta.total_seconds() / 86400.0)


def _build_segment_break_mask(df: pd.DataFrame, *, config: SegmentConfig) -> pd.Series:
    ordered = df.sort_values("time", kind="stable").reset_index(drop=True)
    deltas_hours = ordered["time"].diff().dt.total_seconds().div(3600.0)
    lower_bound = config.step_hours - (config.tolerance_minutes / 60.0)
    upper_bound = config.step_hours + (config.tolerance_minutes / 60.0)

    irregular = ~deltas_hours.between(lower_bound, upper_bound, inclusive="both")
    irregular = irregular.fillna(False)
    irregular.iloc[0] = False
    return irregular.astype(bool)


def _apply_segment_policy(df: pd.DataFrame, *, config: SegmentConfig) -> list[pd.DataFrame]:
    if df.empty:
        return []

    ordered = df.sort_values("time", kind="stable").reset_index(drop=True)
    if config.mode == "ignore":
        return [ordered]

    break_mask = _build_segment_break_mask(ordered, config=config)
    labels = pd.Series(break_mask, index=ordered.index).cumsum().astype(int)
    segments = [segment.reset_index(drop=True) for _, segment in ordered.groupby(labels, sort=True)]

    if config.mode == "longest":
        selected = max(
            segments,
            key=lambda segment: (len(segment), _segment_duration_days(segment)),
        ).copy()
        return [selected]

    return [segment.copy() for segment in segments]


def _convert_drifter_to_processed_trajectories(
    config: dict[str, Any],
) -> tuple[list[pd.DataFrame], DrifterConversionSummary]:
    columns = _resolve_columns(config)
    files = _resolve_input_files(config)
    drogue_config = _resolve_drogue_config(config)
    segment_config = _resolve_segment_config(config)
    region_config = _resolve_region_filter_config(config)
    resample_config = _resolve_resample_config(config)
    summary = DrifterConversionSummary(
        input_files=len(files),
        segment_mode=segment_config.mode,
        region_names_or_labels=region_config.names_or_labels,
        resample_frequency=resample_config.frequency,
    )

    print(f"Resolved {len(files)} drifter CSV file(s)")
    file_iterator = tqdm(files, desc="Reading drifter CSV files", unit="file") if len(files) > 1 else files
    platform_buffers: dict[int, list[pd.DataFrame]] = {}
    for csv_path in file_iterator:
        drifter_points = _read_drifter_points(
            csv_path,
            columns=columns,
            drogue_config=drogue_config,
        )

        for platform_code, platform_df in drifter_points.groupby("platform_code", sort=False):
            code = int(platform_code)
            platform_buffers.setdefault(code, []).append(platform_df.reset_index(drop=True))

    merged_platform_buffers = {
        platform_code: _merge_platform_points(frames)
        for platform_code, frames in platform_buffers.items()
    }

    print(f"Buffered drifter fixes for {len(merged_platform_buffers)} platform(s)")
    summary.buffered_platforms = len(merged_platform_buffers)
    summary.buffered_observations = int(sum(len(platform_df) for platform_df in merged_platform_buffers.values()))

    print(f"Applying segmentation policy: {segment_config.mode}")
    trajectories: list[pd.DataFrame] = []
    platform_iterator = (
        tqdm(
            merged_platform_buffers.items(),
            total=len(merged_platform_buffers),
            desc="Segmenting drifters",
            unit="platform",
        )
        if len(merged_platform_buffers) > 1
        else merged_platform_buffers.items()
    )
    for _, platform_df in platform_iterator:
        if drogue_config.minimum_length_m is not None:
            platform_length_m = _resolve_platform_drogue_length_m(platform_df)
            if platform_length_m is None or platform_length_m < drogue_config.minimum_length_m:
                continue

        clipped = _clip_after_drogue_loss(platform_df, enabled=drogue_config.clip_after_loss)
        if clipped.empty:
            continue

        prepared = clipped.drop(columns=["drogue_lost_time", "drogue_length_m"], errors="ignore")
        trajectories.extend(_apply_segment_policy(prepared, config=segment_config))

    print(f"Built {len(trajectories)} trajectory segment(s) after segmentation")
    summary.segmented_trajectories = len(trajectories)
    summary.segmented_platforms = _count_platforms(trajectories)
    summary.segmented_observations = _count_observations(trajectories)

    if region_config.names_or_labels:
        print(f"Filtering trajectories by region(s): {', '.join(region_config.names_or_labels)}")
    summary.region_input_trajectories = len(trajectories)
    summary.region_input_platforms = _count_platforms(trajectories)
    trajectories = _apply_region_filter(trajectories, config=region_config)

    print(f"Kept {len(trajectories)} trajectory segment(s) after region filtering")
    summary.region_kept_trajectories = len(trajectories)
    summary.region_kept_platforms = _count_platforms(trajectories)

    if resample_config.frequency:
        print(f"Resampling trajectories at frequency: {resample_config.frequency}")
    if resample_config.shared_time and resample_config.reference_time is not None:
        print(f"Using shared time grid from reference time: {resample_config.reference_time.isoformat()}")
    elif resample_config.reference_time is not None:
        print(f"Anchoring resampling to reference time: {resample_config.reference_time.isoformat()}")
    if resample_config.shift_start_to_reference and resample_config.reference_time is not None:
        print(f"Shifting trajectory starts to reference time: {resample_config.reference_time.isoformat()}")
    trajectories = _apply_resampling(trajectories, config=resample_config)

    pre_drop_count = len(trajectories)
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    dropped_empty = pre_drop_count - len(trajectories)
    if dropped_empty > 0:
        print(f"Dropped {dropped_empty} empty trajectory segment(s) after resampling")
    summary.resample_dropped_empty = int(dropped_empty)
    summary.final_trajectories = len(trajectories)
    summary.final_platforms = _count_platforms(trajectories)
    summary.final_observations = _count_observations(trajectories)

    return trajectories, summary


def convert_drifter_to_dataframe(config: dict[str, Any]) -> list[pd.DataFrame]:
    trajectories, _ = _convert_drifter_to_processed_trajectories(config)
    return _normalize_trajectories(trajectories)


def _print_conversion_summary(summary: DrifterConversionSummary) -> None:
    print("")
    print("Drifter conversion summary")
    print(f"  input files: {summary.input_files}")
    print(
        "  buffered fixes: "
        f"{summary.buffered_observations} observation(s) across {summary.buffered_platforms} platform(s)"
    )
    print(f"  segmentation mode: {summary.segment_mode}")
    print(
        "  after segmentation: "
        f"{summary.segmented_trajectories} trajectory segment(s), "
        f"{summary.segmented_platforms} platform(s), "
        f"{summary.segmented_observations} observation(s)"
    )

    if summary.region_names_or_labels:
        print(f"  region filter: {', '.join(summary.region_names_or_labels)}")
    else:
        print("  region filter: disabled")
    print(
        "  trajectories entering region filter: "
        f"{summary.region_input_trajectories} ({summary.region_input_platforms} platform(s))"
    )
    print(
        "  trajectories kept after region filter: "
        f"{summary.region_kept_trajectories} ({summary.region_kept_platforms} platform(s))"
    )

    if summary.resample_frequency:
        print(f"  resampling frequency: {summary.resample_frequency}")
    else:
        print("  resampling: disabled")
    print(f"  empty trajectories dropped after resampling: {summary.resample_dropped_empty}")
    print(
        "  final output: "
        f"{summary.final_trajectories} trajectory segment(s), "
        f"{summary.final_platforms} platform(s), "
        f"{summary.final_observations} observation(s)"
    )
    if summary.output_path:
        print(f"  output path: {summary.output_path}")


def convert_drifter_to_zarr(config: dict[str, Any]) -> xr.Dataset:
    processed_trajectories, summary = _convert_drifter_to_processed_trajectories(config)
    trajectories = _normalize_trajectories(processed_trajectories)
    print("Building output dataset")
    ds = build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
        dataset_attrs=DEFAULT_DATASET_ATTRS,
    )
    encoding = build_zarr_encoding(ds)

    output_path = _resolve_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode="w", encoding=encoding)
    summary.output_path = str(output_path)
    summary.output_counts = {
        "trajectories": len(trajectories),
        "platforms": _count_platforms(trajectories),
        "observations": _count_observations(trajectories),
    }
    _print_conversion_summary(summary)
    print("Drifter conversion completed")
    return ds


def run_conversion(config_path: str | Path) -> xr.Dataset:
    config = load_config(config_path)
    return convert_drifter_to_zarr(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args.config)


if __name__ == "__main__":
    main()
