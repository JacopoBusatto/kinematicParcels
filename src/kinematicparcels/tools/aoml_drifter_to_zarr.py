from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import glob
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

from kinematicparcels.tools.trajectory_processing import (
    apply_region_selection,
    apply_resampling,
    normalize_trajectories,
    resolve_region_selection_config,
    resolve_resample_config,
)
from kinematicparcels.tools.zarr_writer import (
    DEFAULT_TRAJECTORY_DATASET_ATTRS,
    build_dataset_from_trajectories,
    build_zarr_encoding,
)


DEFAULT_DATASET_ATTRS = {
    **DEFAULT_TRAJECTORY_DATASET_ATTRS,
    "source": "AOML GDP 6-hour NetCDF drifter conversion",
    "z_source": "fixed surface value",
    "z_approximation": "z is set to 0.0 for all drifter observations.",
}

TRAJECTORY_LEVEL_COLUMNS = {"platform_code"}
NON_INTERPOLATED_COLUMNS = {"platform_code", "z"}


@dataclass(frozen=True)
class DrogueConfig:
    clip_to_drogued_period: bool = True
    minimum_length_m: float | None = None


@dataclass
class AomlDrifterConversionSummary:
    input_files: int = 0
    raw_trajectories: int = 0
    valid_trajectories: int = 0
    skipped_short_drogue: int = 0
    skipped_empty_after_clip: int = 0
    raw_rows: int = 0
    valid_rows: int = 0
    region_names_or_labels: tuple[str, ...] = ()
    region_selection_mode: str = ""
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


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert AOML GDP 6-hour drifter NetCDF files into a Parcels-compatible trajectory Zarr dataset."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the AOML drifter conversion YAML configuration file.",
    )
    return parser


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_input_files(config: dict[str, Any]) -> list[Path]:
    input_cfg = config.get("input", {})
    files: list[Path] = []

    for key in ("netcdf_files", "drifter_files"):
        for item in _as_list(input_cfg.get(key)):
            files.append(Path(item))

    for key in ("netcdf_glob", "drifter_glob"):
        for pattern in _as_list(input_cfg.get(key)):
            files.extend(Path(path) for path in glob.glob(str(pattern)))

    input_dirs: list[Any] = []
    for key in ("netcdf_dir", "netcdf_dirs", "drifter_dir", "drifter_dirs"):
        input_dirs.extend(_as_list(input_cfg.get(key)))

    pattern = str(input_cfg.get("pattern", "drifter_6h_*.nc"))
    for input_dir in input_dirs:
        files.extend(sorted(Path(input_dir).glob(pattern)))

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError(
            "No AOML drifter NetCDF files were found from the provided configuration."
        )

    missing = [path for path in unique_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"The following AOML drifter NetCDF files were not found: {missing_str}")

    return unique_files


def _resolve_output_path(config: dict[str, Any]) -> Path:
    output_cfg = config.get("output", {})
    output_path = output_cfg.get("path")
    if not output_path:
        raise ValueError("The conversion config must define output.path.")
    return Path(output_path)


def _resolve_drogue_config(config: dict[str, Any]) -> DrogueConfig:
    raw = config.get("processing", {}).get("drogue", {})
    minimum_length_raw = raw.get("minimum_length_m", raw.get("min_length_m"))
    minimum_length_m: float | None = None
    if minimum_length_raw is not None:
        minimum_length_m = float(minimum_length_raw)
        if minimum_length_m < 0.0:
            raise ValueError("processing.drogue.minimum_length_m must be >= 0 when provided")

    clip = raw.get("clip_to_drogued_period", raw.get("clip_after_loss", True))
    return DrogueConfig(
        clip_to_drogued_period=bool(clip),
        minimum_length_m=minimum_length_m,
    )


def _decode_byte(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore")
    return str(value)


def _decode_string_value(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _decode_byte(arr.item()).strip()
    if arr.dtype.kind in {"S", "U"} and arr.dtype.itemsize in {1, 4}:
        return "".join(_decode_byte(item) for item in arr.ravel()).strip()
    return _decode_byte(arr.ravel()[0]).strip()


def _parse_numeric_token(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.number)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(value))
    if not match:
        return None
    numeric = float(match.group(1))
    return numeric if np.isfinite(numeric) else None


def _scalar_at(ds: xr.Dataset, name: str, traj_index: int) -> Any | None:
    if name not in ds:
        return None

    variable = ds[name]
    values = np.asarray(variable.values)
    if values.ndim == 0:
        return values.item()
    if "traj" in variable.dims:
        axis = variable.dims.index("traj")
        selected = np.take(values, traj_index, axis=axis)
        if np.asarray(selected).ndim == 0:
            return np.asarray(selected).item()
        return selected
    return values.ravel()[0] if values.size else None


def _series_at(ds: xr.Dataset, name: str, traj_index: int) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Missing required AOML drifter variable: {name}")

    variable = ds[name]
    values = np.asarray(variable.values)
    if values.ndim == 1:
        return values
    if "traj" in variable.dims:
        axis = variable.dims.index("traj")
        return np.asarray(np.take(values, traj_index, axis=axis)).ravel()
    return values.reshape(-1)


def _coerce_times(values: Any) -> pd.Series:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.Series(pd.to_datetime(arr.ravel(), errors="coerce")).dt.tz_localize(None)
    decoded = pd.to_datetime(pd.Series(arr.ravel()), utc=True, errors="coerce")
    return decoded.dt.tz_convert(None)


def _coerce_time_scalar(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.size == 0:
        return None
    time = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(time):
        return None
    timestamp = pd.Timestamp(time)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _platform_code(ds: xr.Dataset, *, file_path: Path, traj_index: int) -> int:
    raw_id = _scalar_at(ds, "ID", traj_index)
    if raw_id is not None:
        candidate = _decode_string_value(raw_id)
        match = re.search(r"\d+", candidate)
        if match:
            return int(match.group(0))

    match = re.search(r"drifter_6h_(\d+)$", file_path.stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"Unable to derive drifter ID from {file_path}")


def _drogue_length_m(ds: xr.Dataset, traj_index: int) -> float | None:
    if "DrogueLength" in ds.attrs:
        return _parse_numeric_token(ds.attrs.get("DrogueLength"))

    raw = _scalar_at(ds, "DrogueLength", traj_index)
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = _decode_string_value(raw)
    return _parse_numeric_token(raw)


def _valid_drogue_length(length_m: float | None, *, config: DrogueConfig) -> bool:
    if config.minimum_length_m is None:
        return True
    return length_m is not None and length_m >= config.minimum_length_m


def _read_file_trajectories(
    file_path: Path,
    *,
    drogue_config: DrogueConfig,
) -> tuple[list[pd.DataFrame], dict[str, int]]:
    with xr.open_dataset(
        file_path,
        decode_times=True,
        decode_timedelta=False,
        mask_and_scale=True,
        drop_variables=("WMO",),
    ) as ds:
        traj_count = int(ds.sizes.get("traj", 1))
        trajectories: list[pd.DataFrame] = []
        summary = {
            "raw_trajectories": traj_count,
            "valid_trajectories": 0,
            "skipped_short_drogue": 0,
            "skipped_empty_after_clip": 0,
            "raw_rows": 0,
            "valid_rows": 0,
        }

        for traj_index in range(traj_count):
            drogue_length_m = _drogue_length_m(ds, traj_index)
            if not _valid_drogue_length(drogue_length_m, config=drogue_config):
                summary["skipped_short_drogue"] += 1
                continue

            platform_code = _platform_code(ds, file_path=file_path, traj_index=traj_index)
            time = _coerce_times(_series_at(ds, "time", traj_index))
            lat = pd.to_numeric(pd.Series(_series_at(ds, "latitude", traj_index)), errors="coerce")
            lon = pd.to_numeric(pd.Series(_series_at(ds, "longitude", traj_index)), errors="coerce")

            n_rows = min(len(time), len(lat), len(lon))
            frame = pd.DataFrame(
                {
                    "platform_code": np.full(n_rows, platform_code, dtype=np.int64),
                    "time": time.iloc[:n_rows].reset_index(drop=True),
                    "lat": lat.iloc[:n_rows].reset_index(drop=True),
                    "lon": lon.iloc[:n_rows].reset_index(drop=True),
                    "z": np.zeros(n_rows, dtype=float),
                }
            )
            summary["raw_rows"] += int(n_rows)

            if drogue_config.clip_to_drogued_period:
                start_time = _coerce_time_scalar(_scalar_at(ds, "start_date", traj_index))
                drogue_lost_time = _coerce_time_scalar(_scalar_at(ds, "drogue_lost_date", traj_index))
                end_time = _coerce_time_scalar(_scalar_at(ds, "end_date", traj_index))

                if start_time is not None:
                    frame = frame.loc[frame["time"] >= start_time]
                if drogue_lost_time is not None:
                    frame = frame.loc[frame["time"] < drogue_lost_time]
                elif end_time is not None:
                    frame = frame.loc[frame["time"] <= end_time]

            valid = frame["time"].notna() & frame["lat"].notna() & frame["lon"].notna()
            frame = frame.loc[valid].sort_values(["time", "lat", "lon"], kind="stable").reset_index(drop=True)
            summary["valid_rows"] += int(len(frame))

            if frame.empty:
                summary["skipped_empty_after_clip"] += 1
                continue

            trajectories.append(frame)
            summary["valid_trajectories"] += 1

    return trajectories, summary


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


def _convert_aoml_drifter_to_processed_trajectories(
    config: dict[str, Any],
) -> tuple[list[pd.DataFrame], AomlDrifterConversionSummary]:
    files = _resolve_input_files(config)
    drogue_config = _resolve_drogue_config(config)
    region_config = resolve_region_selection_config(config)
    resample_config = resolve_resample_config(config)
    summary = AomlDrifterConversionSummary(
        input_files=len(files),
        region_names_or_labels=region_config.names_or_labels,
        region_selection_mode=region_config.selection_mode,
        resample_frequency=resample_config.frequency,
    )

    print(f"Resolved {len(files)} AOML drifter NetCDF file(s)")
    file_iterator = tqdm(files, desc="Reading AOML drifter files", unit="file") if len(files) > 1 else files

    trajectories: list[pd.DataFrame] = []
    for file_path in file_iterator:
        file_trajectories, file_summary = _read_file_trajectories(
            file_path,
            drogue_config=drogue_config,
        )
        trajectories.extend(file_trajectories)
        summary.raw_trajectories += file_summary["raw_trajectories"]
        summary.valid_trajectories += file_summary["valid_trajectories"]
        summary.skipped_short_drogue += file_summary["skipped_short_drogue"]
        summary.skipped_empty_after_clip += file_summary["skipped_empty_after_clip"]
        summary.raw_rows += file_summary["raw_rows"]
        summary.valid_rows += file_summary["valid_rows"]

    print(f"Buffered {len(trajectories)} AOML drifter trajectory/trajectories")
    print(f"Kept {summary.valid_rows}/{summary.raw_rows} rows with valid time/lat/lon after drogue clipping")
    if drogue_config.minimum_length_m is not None:
        print(
            "Drogue-length filter: "
            f"minimum={drogue_config.minimum_length_m:g} m, skipped={summary.skipped_short_drogue}"
        )
    if summary.skipped_empty_after_clip > 0:
        print(f"Skipped {summary.skipped_empty_after_clip} empty trajectory/trajectories after clipping")

    if region_config.names_or_labels:
        print(
            "Applying region selection "
            f"({region_config.selection_mode}) for: {', '.join(region_config.names_or_labels)}"
        )
    summary.region_input_trajectories = len(trajectories)
    summary.region_input_platforms = _count_platforms(trajectories)
    trajectories = apply_region_selection(trajectories, config=region_config)
    print(f"Kept {len(trajectories)} trajectory/trajectories after region selection")
    summary.region_kept_trajectories = len(trajectories)
    summary.region_kept_platforms = _count_platforms(trajectories)

    if resample_config.frequency:
        print(f"Resampling trajectories at frequency: {resample_config.frequency}")
    if resample_config.shared_time and resample_config.reference_time is not None:
        print(f"Using shared time grid from reference time: {resample_config.reference_time.isoformat()}")
    elif resample_config.reference_time is not None:
        print(f"Anchoring resampling to reference time: {resample_config.reference_time.isoformat()}")
    trajectories = apply_resampling(
        trajectories,
        config=resample_config,
        non_interpolated_columns=NON_INTERPOLATED_COLUMNS,
    )

    pre_drop_count = len(trajectories)
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    summary.resample_dropped_empty = int(pre_drop_count - len(trajectories))
    if summary.resample_dropped_empty > 0:
        print(f"Dropped {summary.resample_dropped_empty} empty trajectory/trajectories after resampling")

    summary.final_trajectories = len(trajectories)
    summary.final_platforms = _count_platforms(trajectories)
    summary.final_observations = _count_observations(trajectories)
    if not trajectories:
        raise ValueError("No trajectories were produced after AOML drifter filtering and processing")

    return trajectories, summary


def convert_aoml_drifter_to_dataframe(config: dict[str, Any]) -> list[pd.DataFrame]:
    trajectories, _ = _convert_aoml_drifter_to_processed_trajectories(config)
    normalized = normalize_trajectories(trajectories)
    if not normalized:
        raise ValueError("No trajectories were produced after AOML drifter filtering and processing")
    return normalized


def _build_aoml_drifter_dataset(trajectories: list[pd.DataFrame]) -> xr.Dataset:
    print("Building output dataset")
    return build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
        dataset_attrs=DEFAULT_DATASET_ATTRS,
    )


def _write_dataset_to_zarr(ds: xr.Dataset, output_path: Path) -> None:
    encoding = build_zarr_encoding(ds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode="w", encoding=encoding)


def _print_conversion_summary(summary: AomlDrifterConversionSummary) -> None:
    print("")
    print("AOML drifter conversion summary")
    print(f"  input files: {summary.input_files}")
    print(f"  raw trajectories: {summary.raw_trajectories}")
    print(f"  valid trajectories after drogue filtering/clipping: {summary.valid_trajectories}")
    print(f"  valid rows: {summary.valid_rows}/{summary.raw_rows}")
    print(f"  skipped by drogue length: {summary.skipped_short_drogue}")
    print(f"  skipped empty after clipping: {summary.skipped_empty_after_clip}")
    if summary.region_names_or_labels:
        regions = ", ".join(summary.region_names_or_labels)
        print(f"  region selection: {summary.region_selection_mode} over {regions}")
    else:
        print("  region selection: disabled")
    print(
        "  trajectories entering region stage: "
        f"{summary.region_input_trajectories} ({summary.region_input_platforms} platform(s))"
    )
    print(
        "  trajectories kept after region selection: "
        f"{summary.region_kept_trajectories} ({summary.region_kept_platforms} platform(s))"
    )
    if summary.resample_frequency:
        print(f"  resampling frequency: {summary.resample_frequency}")
    else:
        print("  resampling: disabled")
    print(f"  empty trajectories dropped after resampling: {summary.resample_dropped_empty}")
    print(
        "  final trajectories after resampling: "
        f"{summary.final_trajectories} ({summary.final_platforms} platform(s), "
        f"{summary.final_observations} observation(s))"
    )
    if summary.output_path:
        print(f"  output path: {summary.output_path}")


def convert_aoml_drifter_to_zarr(config: dict[str, Any]) -> xr.Dataset:
    trajectories, summary = _convert_aoml_drifter_to_processed_trajectories(config)
    normalized = normalize_trajectories(trajectories)
    if not normalized:
        raise ValueError("No trajectories were produced after AOML drifter filtering and processing")

    ds = _build_aoml_drifter_dataset(normalized)
    output_path = _resolve_output_path(config)
    _write_dataset_to_zarr(ds, output_path)
    summary.output_path = str(output_path)
    summary.output_counts = {
        "trajectories": len(normalized),
        "platforms": _count_platforms(normalized),
        "observations": _count_observations(normalized),
    }
    _print_conversion_summary(summary)
    print("AOML drifter conversion completed")
    return ds


def run_conversion(config_path: str | Path) -> xr.Dataset:
    config = load_config(config_path)
    return convert_aoml_drifter_to_zarr(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args.config)


if __name__ == "__main__":
    main()
