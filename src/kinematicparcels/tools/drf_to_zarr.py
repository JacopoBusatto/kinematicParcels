from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from kinematicparcels.tools.argo_to_zarr import (
    _apply_region_filter,
    _apply_resampling,
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

DEFAULT_DATASET_ATTRS = {
    **DEFAULT_TRAJECTORY_DATASET_ATTRS,
    "source": "DRF conversion",
}

TRAJECTORY_LEVEL_COLUMNS = {"platform_code"}


@dataclass(frozen=True)
class SegmentConfig:
    mode: str = "ignore"
    step_hours: float = 6.0
    tolerance_minutes: float = 30.0


@dataclass(frozen=True)
class AtSeaConfig:
    allowed_flags: tuple[int, ...] = (1,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert IOS DRF files into a Parcels-compatible trajectory Zarr dataset."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the DRF conversion YAML configuration file.",
    )
    return parser


def _resolve_input_files(config: dict[str, Any]) -> list[Path]:
    input_cfg = config.get("input", {})
    files: list[Path] = []

    for item in input_cfg.get("drf_files", []) or []:
        files.append(Path(item))

    drf_glob = input_cfg.get("drf_glob")
    if drf_glob:
        files.extend(Path(path) for path in glob.glob(drf_glob))

    input_dir = input_cfg.get("drf_dir")
    pattern = input_cfg.get("pattern", "*.drf")
    if input_dir:
        files.extend(sorted(Path(input_dir).glob(pattern)))

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError("No input DRF files were found from the provided configuration.")

    missing = [path for path in unique_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"The following DRF files were not found: {missing_str}")

    return unique_files


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


def _resolve_at_sea_config(config: dict[str, Any]) -> AtSeaConfig:
    raw = config.get("processing", {}).get("quality", {})
    flags_raw = raw.get("keep_at_sea_flags", [1])
    if isinstance(flags_raw, (int, float, str)):
        flags_raw = [flags_raw]

    parsed: list[int] = []
    for value in flags_raw:
        parsed_value = int(value)
        if parsed_value < 0:
            raise ValueError("processing.quality.keep_at_sea_flags must contain non-negative integers")
        parsed.append(parsed_value)

    if not parsed:
        raise ValueError("processing.quality.keep_at_sea_flags must contain at least one flag")

    return AtSeaConfig(allowed_flags=tuple(sorted(set(parsed))))


def _split_header_body(lines: list[str], *, file_path: Path) -> tuple[list[str], list[str]]:
    for idx, line in enumerate(lines):
        if line.strip().upper() == "*END OF HEADER":
            return lines[:idx], lines[idx + 1 :]
    raise ValueError(f"Could not find '*END OF HEADER' marker in DRF file: {file_path}")


def _parse_instrument_id(header_lines: list[str]) -> str | None:
    in_instrument = False
    for raw_line in header_lines:
        stripped = raw_line.strip()
        if stripped.upper().startswith("*INSTRUMENT"):
            in_instrument = True
            continue
        if in_instrument and stripped.startswith("*"):
            in_instrument = False
        if not in_instrument:
            continue

        match = re.match(r"^ID\s*:\s*(.+)$", stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _derive_platform_code(file_path: Path, instrument_id: str | None) -> int:
    candidates = []
    if instrument_id:
        candidates.append(instrument_id)
    candidates.append(file_path.stem)

    for token in candidates:
        direct = token.strip()
        if direct.isdigit():
            return int(direct)

        match = re.search(r"(\d{3,})", direct)
        if match:
            return int(match.group(1))

    raise ValueError(f"Unable to derive numeric platform_code from DRF file '{file_path}'")


def _parse_body_rows(body_lines: list[str], *, file_path: Path, platform_code: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            continue

        fields = stripped.split()
        if len(fields) < 6:
            continue

        date_token = fields[1]
        time_token = fields[2]
        timestamp = pd.to_datetime(f"{date_token} {time_token}", format="%Y/%m/%d %H:%M:%S", utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue

        lat = pd.to_numeric(fields[3], errors="coerce")
        lon = pd.to_numeric(fields[4], errors="coerce")
        at_sea = pd.to_numeric(fields[5], errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(at_sea):
            continue

        records.append(
            {
                "platform_code": platform_code,
                "time": timestamp.tz_convert(None),
                "lat": float(lat),
                "lon": float(lon),
                "at_sea_flag": int(at_sea),
                "z": 0.0,
            }
        )

    if not records:
        raise ValueError(f"No valid trajectory rows were parsed from DRF file: {file_path}")

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["platform_code", "time", "lat", "lon"], kind="stable").reset_index(drop=True)


def _read_drf_points(file_path: Path) -> pd.DataFrame:
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_lines, body_lines = _split_header_body(lines, file_path=file_path)
    instrument_id = _parse_instrument_id(header_lines)
    platform_code = _derive_platform_code(file_path, instrument_id)
    return _parse_body_rows(body_lines, file_path=file_path, platform_code=platform_code)


def _merge_platform_points(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0].sort_values(["platform_code", "time", "lat", "lon"], kind="stable").reset_index(drop=True)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["platform_code", "time", "lat", "lon"], kind="stable")
    merged = merged.drop_duplicates(subset=["platform_code", "time", "lat", "lon"], keep="last")
    return merged.reset_index(drop=True)


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


def _compute_trajectory_delta_seconds(trajectory: pd.DataFrame) -> np.ndarray:
    if len(trajectory) <= 1:
        return np.asarray([], dtype=float)
    return trajectory["time"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)


def _build_cadence_summary(trajectories: list[pd.DataFrame]) -> dict[str, Any]:
    all_deltas: list[np.ndarray] = []
    unique_resolution_counts: dict[int, int] = {}

    for trajectory in trajectories:
        deltas = _compute_trajectory_delta_seconds(trajectory)
        if deltas.size == 0:
            continue

        rounded = np.rint(deltas).astype(np.int64)
        all_deltas.append(rounded.astype(float))
        values, counts = np.unique(rounded, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            unique_resolution_counts[int(value)] = unique_resolution_counts.get(int(value), 0) + int(count)

    if not all_deltas:
        return {
            "n_trajectories": len(trajectories),
            "n_with_deltas": 0,
            "min_step_seconds": None,
            "median_step_seconds": None,
            "max_step_seconds": None,
            "mode_step_seconds": None,
            "common_steps_seconds": [],
            "step_histogram": {},
        }

    stacked = np.concatenate(all_deltas)
    values, counts = np.unique(stacked.astype(np.int64), return_counts=True)
    mode_step = int(values[np.argmax(counts)])

    sorted_hist = sorted(unique_resolution_counts.items(), key=lambda item: (-item[1], item[0]))
    common_steps = [step for step, _ in sorted_hist[:10]]

    return {
        "n_trajectories": len(trajectories),
        "n_with_deltas": len(all_deltas),
        "min_step_seconds": int(np.min(stacked)),
        "median_step_seconds": int(np.median(stacked)),
        "max_step_seconds": int(np.max(stacked)),
        "mode_step_seconds": mode_step,
        "common_steps_seconds": common_steps,
        "step_histogram": {str(step): int(count) for step, count in sorted_hist},
    }


def _print_cadence_summary(summary: dict[str, Any]) -> None:
    print("Cadence diagnostics (pre-resample):")
    print(f"  trajectories: {summary['n_trajectories']}")
    print(f"  trajectories with deltas: {summary['n_with_deltas']}")
    if summary["n_with_deltas"] == 0:
        return

    print(
        "  step seconds min/median/mode/max: "
        f"{summary['min_step_seconds']} / {summary['median_step_seconds']} / "
        f"{summary['mode_step_seconds']} / {summary['max_step_seconds']}"
    )
    common = ", ".join(str(step) for step in summary["common_steps_seconds"])
    print(f"  common step seconds: {common}")


def convert_drf_to_dataframe(config: dict[str, Any]) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    files = _resolve_input_files(config)
    segment_config = _resolve_segment_config(config)
    at_sea_config = _resolve_at_sea_config(config)
    region_config = _resolve_region_filter_config(config)
    resample_config = _resolve_resample_config(config)

    print(f"Resolved {len(files)} DRF file(s)")
    print(f"Keeping At_Sea flags: {', '.join(str(flag) for flag in at_sea_config.allowed_flags)}")
    file_iterator = tqdm(files, desc="Reading DRF files", unit="file") if len(files) > 1 else files
    platform_buffers: dict[int, list[pd.DataFrame]] = {}

    for drf_path in file_iterator:
        points = _read_drf_points(drf_path)
        points = points[points["at_sea_flag"].isin(at_sea_config.allowed_flags)].reset_index(drop=True)
        if points.empty:
            continue

        for platform_code, platform_df in points.groupby("platform_code", sort=False):
            code = int(platform_code)
            platform_buffers.setdefault(code, []).append(platform_df.reset_index(drop=True))

    merged_platform_buffers = {
        platform_code: _merge_platform_points(frames)
        for platform_code, frames in platform_buffers.items()
    }

    print(f"Buffered DRF fixes for {len(merged_platform_buffers)} platform(s)")
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
        prepared = platform_df.drop(columns=["at_sea_flag"], errors="ignore")
        trajectories.extend(_apply_segment_policy(prepared, config=segment_config))

    print(f"Built {len(trajectories)} trajectory segment(s) after segmentation")
    if region_config.names_or_labels:
        print(f"Filtering trajectories by region(s): {', '.join(region_config.names_or_labels)}")
    trajectories = _apply_region_filter(trajectories, config=region_config)
    print(f"Kept {len(trajectories)} trajectory segment(s) after region filtering")

    cadence_summary = _build_cadence_summary(trajectories)
    _print_cadence_summary(cadence_summary)

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

    if not normalized:
        raise ValueError("No trajectories were produced after DRF filtering and processing")

    return normalized, cadence_summary


def _build_cadence_dataset_attrs(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "cadence_n_trajectories": summary["n_trajectories"],
        "cadence_n_with_deltas": summary["n_with_deltas"],
        "cadence_min_step_seconds": summary["min_step_seconds"],
        "cadence_median_step_seconds": summary["median_step_seconds"],
        "cadence_max_step_seconds": summary["max_step_seconds"],
        "cadence_mode_step_seconds": summary["mode_step_seconds"],
        "cadence_common_steps_seconds": json.dumps(summary["common_steps_seconds"]),
        "cadence_step_histogram": json.dumps(summary["step_histogram"]),
    }


def convert_drf_to_zarr(config: dict[str, Any]) -> xr.Dataset:
    trajectories, cadence_summary = convert_drf_to_dataframe(config)
    dataset_attrs = dict(DEFAULT_DATASET_ATTRS)
    dataset_attrs.update(_build_cadence_dataset_attrs(cadence_summary))

    print("Building output dataset")
    ds = build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
        dataset_attrs=dataset_attrs,
    )
    encoding = build_zarr_encoding(ds)

    output_path = _resolve_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode="w", encoding=encoding)
    print("DRF conversion completed")
    return ds


def run_conversion(config_path: str | Path) -> xr.Dataset:
    config = load_config(config_path)
    return convert_drf_to_zarr(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args.config)


if __name__ == "__main__":
    main()
