from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import glob
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, *args, **kwargs):
        return iterable

from kinematicparcels.tools.trajectory_processing import (
    apply_resampling,
    apply_region_selection,
    normalize_trajectories,
    RegionSelectionConfig,
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
    "source": "ARGO Rtraj NetCDF conversion",
    "z_source": "Argo REPRESENTATIVE_PARK_PRESSURE",
    "z_approximation": (
        "z is approximated from Argo REPRESENTATIVE_PARK_PRESSURE in dbar, "
        "with positive values and no pressure-to-geometric-depth conversion."
    ),
}

TRAJECTORY_LEVEL_COLUMNS = {"platform_code", "depth_bin", "depth_bin_interval"}
NON_INTERPOLATED_COLUMNS = {"platform_code", "z"}


@dataclass(frozen=True)
class ParkingDepthConfig:
    mode: str = "representative_park_pressure"
    fallback_value: float | None = None


@dataclass(frozen=True)
class DepthBin:
    label: str
    min_value: float
    max_value: float | None = None

    @property
    def interval_label(self) -> str:
        upper = "+inf" if self.max_value is None else f"{self.max_value:g}"
        return f"[{self.min_value:g}, {upper})"


@dataclass(frozen=True)
class DepthBinConfig:
    enabled: bool = False
    output_mode: str = "per_bin"
    bins: tuple[DepthBin, ...] = ()


@dataclass
class RtrajConversionSummary:
    input_files: int = 0
    raw_trajectories: int = 0
    raw_platforms: int = 0
    raw_rows: int = 0
    valid_rows: int = 0
    unmapped_z_rows: int = 0
    missing_z_rows: int = 0
    depth_bins_enabled: bool = False
    depth_bin_segments: int = 0
    depth_bin_counts: dict[str, dict[str, int]] = field(default_factory=dict)
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
    output_counts: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ARGO Rtraj NetCDF files into a Parcels-compatible trajectory Zarr dataset."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the ARGO Rtraj conversion YAML configuration file.",
    )
    return parser


def _resolve_input_files(config: dict[str, Any]) -> list[Path]:
    input_cfg = config.get("input", {})
    files: list[Path] = []

    for item in input_cfg.get("rtraj_files", []) or []:
        files.append(Path(item))

    rtraj_glob = input_cfg.get("rtraj_glob")
    if rtraj_glob:
        files.extend(Path(path) for path in glob.glob(rtraj_glob))

    input_dir = input_cfg.get("rtraj_dir")
    pattern = input_cfg.get("pattern", "*_Rtraj.nc")
    if input_dir:
        files.extend(sorted(Path(input_dir).glob(pattern)))

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError("No input ARGO Rtraj NetCDF files were found from the provided configuration.")

    missing = [path for path in unique_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"The following ARGO Rtraj files were not found: {missing_str}")

    return unique_files


def _resolve_output_path(config: dict[str, Any]) -> Path:
    output_cfg = config.get("output", {})
    output_path = output_cfg.get("path")
    if not output_path:
        raise ValueError("The conversion config must define output.path.")
    return Path(output_path)


def _resolve_parking_depth_config(config: dict[str, Any]) -> ParkingDepthConfig:
    raw = config.get("processing", {}).get("parking_depth", {})
    mode = str(raw.get("mode", "representative_park_pressure"))
    if mode != "representative_park_pressure":
        raise ValueError(
            "processing.parking_depth.mode must be 'representative_park_pressure' for Rtraj conversion"
        )

    fallback_raw = raw.get("fallback_value", None)
    fallback_value = None if fallback_raw is None else float(fallback_raw)

    return ParkingDepthConfig(mode=mode, fallback_value=fallback_value)


def _sanitize_depth_bin_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("processing.depth_bins.bins labels must not be empty")
    return sanitized


def _resolve_depth_bin_config(config: dict[str, Any]) -> DepthBinConfig:
    raw = config.get("processing", {}).get("depth_bins", {}) or {}
    enabled = bool(raw.get("enabled", False))
    output_mode = str(raw.get("output_mode", "per_bin"))
    if output_mode != "per_bin":
        raise ValueError("processing.depth_bins.output_mode currently supports only 'per_bin'")

    if not enabled:
        return DepthBinConfig(enabled=False, output_mode=output_mode)

    raw_bins = raw.get("bins", []) or []
    if not raw_bins:
        raise ValueError("processing.depth_bins.bins must contain at least one bin when depth bins are enabled")

    bins: list[DepthBin] = []
    seen_labels: set[str] = set()
    for idx, raw_bin in enumerate(raw_bins):
        if not isinstance(raw_bin, dict):
            raise ValueError("Each processing.depth_bins.bins entry must be a mapping with label, min, and max")

        label_raw = raw_bin.get("label")
        if label_raw is None:
            raise ValueError(f"Missing label for processing.depth_bins.bins[{idx}]")
        label = _sanitize_depth_bin_label(str(label_raw))
        if label in seen_labels:
            raise ValueError(f"Duplicate processing.depth_bins.bins label: {label}")
        seen_labels.add(label)

        if "min" not in raw_bin:
            raise ValueError(f"Missing min for processing.depth_bins.bins[{idx}]")
        min_value = float(raw_bin["min"])
        max_raw = raw_bin.get("max", None)
        max_value = None if max_raw is None else float(max_raw)
        if not np.isfinite(min_value):
            raise ValueError(f"processing.depth_bins.bins[{idx}].min must be finite")
        if max_value is not None:
            if not np.isfinite(max_value):
                raise ValueError(f"processing.depth_bins.bins[{idx}].max must be finite or null")
            if max_value <= min_value:
                raise ValueError(f"processing.depth_bins.bins[{idx}].max must be greater than min")

        bins.append(DepthBin(label=label, min_value=min_value, max_value=max_value))

    sorted_bins = tuple(sorted(bins, key=lambda item: item.min_value))
    for idx, current in enumerate(sorted_bins[:-1]):
        nxt = sorted_bins[idx + 1]
        if current.max_value is None:
            raise ValueError("Only the last depth bin may have max: null")
        if nxt.min_value < current.max_value:
            raise ValueError(
                "processing.depth_bins.bins must not overlap; "
                f"{nxt.label} starts before {current.label} ends"
            )

    return DepthBinConfig(enabled=True, output_mode=output_mode, bins=sorted_bins)


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


def _summarize_by_depth_bin(trajectories: list[pd.DataFrame]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for trajectory in trajectories:
        if trajectory.empty or "depth_bin" not in trajectory.columns:
            continue
        label = str(trajectory["depth_bin"].iloc[0])
        grouped.setdefault(label, []).append(trajectory)

    return {
        label: {
            "trajectories": len(frames),
            "platforms": _count_platforms(frames),
            "observations": _count_observations(frames),
        }
        for label, frames in sorted(grouped.items())
    }


@contextmanager
def _suppress_xarray_time_serialization_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unable to decode time axis into full numpy.datetime64.*",
            category=xr.SerializationWarning,
        )
        yield


def _decode_byte(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore")
    return str(value)


def _decode_string_values(values: Any) -> list[str]:
    arr = np.asarray(values)
    if arr.dtype.kind not in {"S", "U", "O"}:
        return [str(value) for value in arr.ravel().tolist()]

    if arr.ndim == 0:
        return [_decode_byte(arr.item()).strip()]

    if arr.dtype.kind in {"S", "U"} and arr.dtype.itemsize in {1, 4} and arr.ndim >= 1:
        rows = arr.reshape(-1, arr.shape[-1])
        decoded: list[str] = []
        for row in rows:
            decoded.append("".join(_decode_byte(value) for value in row).strip())
        return decoded

    return [_decode_byte(value).strip() for value in arr.ravel().tolist()]


def _decode_platform_number(ds: xr.Dataset, *, file_path: Path) -> int:
    if "PLATFORM_NUMBER" in ds:
        candidates = _decode_string_values(ds["PLATFORM_NUMBER"].values)
        for candidate in candidates:
            match = re.search(r"\d+", candidate)
            if match:
                return int(match.group(0))

    match = re.search(r"(\d+)_Rtraj$", file_path.stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"Unable to derive PLATFORM_NUMBER from Rtraj file: {file_path}")


def _decode_numeric_values(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "U", "O"}:
        strings = _decode_string_values(arr)
        return pd.to_numeric(pd.Series(strings), errors="coerce").to_numpy(dtype=float)

    return pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").to_numpy(dtype=float)


def _numeric_variable(ds: xr.Dataset, name: str) -> np.ndarray | None:
    if name not in ds:
        return None
    return _decode_numeric_values(ds[name].values)


def _datetime_variable(ds: xr.Dataset, name: str) -> pd.Series | None:
    if name not in ds:
        return None

    variable = ds[name]
    values = np.asarray(variable.values)

    if np.issubdtype(values.dtype, np.datetime64):
        return pd.Series(pd.to_datetime(values.ravel(), errors="coerce")).dt.tz_localize(None)

    if np.issubdtype(values.dtype, np.number):
        numeric = pd.to_numeric(pd.Series(values.ravel()), errors="coerce").to_numpy(dtype=float)
        out = pd.Series(pd.NaT, index=np.arange(len(numeric)), dtype="datetime64[ns]")
        valid = np.isfinite(numeric)
        if not valid.any():
            return out

        units = str(variable.attrs.get("units", "")).strip()
        calendar = variable.attrs.get("calendar", "standard")
        if units:
            try:
                with _suppress_xarray_time_serialization_warning():
                    decoded = xr.coding.times.decode_cf_datetime(numeric[valid], units, calendar=calendar)
                out.loc[valid] = pd.to_datetime(decoded, errors="coerce")
                return out.dt.tz_localize(None)
            except Exception:
                pass

        origin = pd.Timestamp("1950-01-01T00:00:00")
        out.loc[valid] = origin + pd.to_timedelta(numeric[valid], unit="D")
        return out

    decoded = pd.to_datetime(pd.Series(values.ravel()), utc=True, errors="coerce")
    return decoded.dt.tz_convert(None)


def _choose_adjusted_datetime(ds: xr.Dataset, adjusted_name: str, raw_name: str) -> pd.Series:
    raw = _datetime_variable(ds, raw_name)
    if raw is None:
        raise KeyError(f"Missing required Rtraj variable: {raw_name}")

    adjusted = _datetime_variable(ds, adjusted_name)
    if adjusted is None:
        return raw

    if len(adjusted) != len(raw):
        raise ValueError(f"{adjusted_name} and {raw_name} have different lengths")

    selected = adjusted.copy()
    selected.loc[selected.isna()] = raw.loc[selected.isna()]
    return selected


def _cycle_key(value: float) -> int | float | None:
    if not np.isfinite(value):
        return None
    rounded = round(float(value))
    if np.isclose(float(value), rounded):
        return int(rounded)
    return float(value)


def _build_pressure_lookup(cycle_index: np.ndarray | None, pressure: np.ndarray | None) -> dict[int | float, float]:
    if cycle_index is None or pressure is None:
        return {}

    count = min(len(cycle_index), len(pressure))
    lookup: dict[int | float, float] = {}
    for cycle_value, pressure_value in zip(cycle_index[:count], pressure[:count]):
        key = _cycle_key(float(cycle_value))
        if key is None or not np.isfinite(float(pressure_value)):
            continue
        lookup[key] = float(pressure_value)
    return lookup


def _map_pressure_from_lookup(cycle_numbers: np.ndarray, lookup: dict[int | float, float]) -> np.ndarray:
    z = np.full(len(cycle_numbers), np.nan, dtype=float)
    if not lookup:
        return z

    for idx, cycle_value in enumerate(cycle_numbers):
        key = _cycle_key(float(cycle_value))
        if key is None:
            continue
        mapped = lookup.get(key)
        if mapped is not None:
            z[idx] = float(mapped)
    return z


def _map_parking_pressure_to_observations(
    ds: xr.Dataset,
    *,
    fallback_value: float | None,
) -> tuple[np.ndarray, int, int]:
    pressure = _numeric_variable(ds, "REPRESENTATIVE_PARK_PRESSURE")
    adjusted_cycles = _numeric_variable(ds, "CYCLE_NUMBER_ADJUSTED")
    adjusted_cycle_index = _numeric_variable(ds, "CYCLE_NUMBER_INDEX_ADJUSTED")
    raw_cycles = _numeric_variable(ds, "CYCLE_NUMBER")
    raw_cycle_index = _numeric_variable(ds, "CYCLE_NUMBER_INDEX")

    if raw_cycles is None:
        raise KeyError("Missing required Rtraj variable: CYCLE_NUMBER")

    if adjusted_cycles is None:
        adjusted_cycles = raw_cycles
    elif len(adjusted_cycles) != len(raw_cycles):
        raise ValueError("CYCLE_NUMBER_ADJUSTED and CYCLE_NUMBER have different lengths")

    adjusted_lookup = _build_pressure_lookup(adjusted_cycle_index, pressure)
    raw_lookup = _build_pressure_lookup(raw_cycle_index, pressure)

    z = _map_pressure_from_lookup(adjusted_cycles, adjusted_lookup)
    missing_after_adjusted = ~np.isfinite(z)
    if missing_after_adjusted.any():
        raw_z = _map_pressure_from_lookup(raw_cycles, raw_lookup)
        z[missing_after_adjusted] = raw_z[missing_after_adjusted]

    unmapped_count = int(np.count_nonzero(~np.isfinite(z)))
    if fallback_value is not None:
        z[~np.isfinite(z)] = float(fallback_value)

    missing_final_count = int(np.count_nonzero(~np.isfinite(z)))
    return z, unmapped_count, missing_final_count


def _validate_equal_lengths(file_path: Path, **arrays: Any) -> int:
    lengths = {name: len(value) for name, value in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Inconsistent measurement variable lengths in {file_path}: {details}")
    return unique_lengths.pop()


def _read_rtraj_file(
    file_path: Path,
    *,
    parking_config: ParkingDepthConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    with _suppress_xarray_time_serialization_warning():
        ds_context = xr.open_dataset(
            file_path,
            decode_times=True,
            decode_timedelta=False,
            mask_and_scale=True,
        )

    with ds_context as ds:
        platform_code = _decode_platform_number(ds, file_path=file_path)
        time = _choose_adjusted_datetime(ds, "JULD_ADJUSTED", "JULD")
        lat = _numeric_variable(ds, "LATITUDE")
        lon = _numeric_variable(ds, "LONGITUDE")
        if lat is None:
            raise KeyError(f"Missing required Rtraj variable in {file_path}: LATITUDE")
        if lon is None:
            raise KeyError(f"Missing required Rtraj variable in {file_path}: LONGITUDE")

        z, unmapped_z_count, missing_z_count = _map_parking_pressure_to_observations(
            ds,
            fallback_value=parking_config.fallback_value,
        )

    n_rows = _validate_equal_lengths(file_path, time=time, lat=lat, lon=lon, z=z)
    frame = pd.DataFrame(
        {
            "platform_code": np.full(n_rows, platform_code, dtype=np.int64),
            "time": time,
            "lat": lat,
            "lon": lon,
            "z": z,
        }
    )

    valid = frame["time"].notna() & frame["lat"].notna() & frame["lon"].notna()
    frame = frame.loc[valid].sort_values(["time", "lat", "lon"], kind="stable").reset_index(drop=True)

    summary = {
        "rows": int(n_rows),
        "valid_rows": int(len(frame)),
        "unmapped_z": int(unmapped_z_count),
        "missing_z": int(missing_z_count),
    }
    return frame, summary


def _find_depth_bin(value: float, bins: tuple[DepthBin, ...]) -> DepthBin | None:
    if not np.isfinite(value):
        return None

    for depth_bin in bins:
        if value < depth_bin.min_value:
            continue
        if depth_bin.max_value is None or value < depth_bin.max_value:
            return depth_bin

    return None


def _split_trajectory_by_depth_bin(
    trajectory: pd.DataFrame,
    *,
    config: DepthBinConfig,
) -> list[pd.DataFrame]:
    if not config.enabled:
        return [trajectory.reset_index(drop=True)]

    if trajectory.empty:
        return []

    segments: list[pd.DataFrame] = []
    start_idx: int | None = None
    current_bin: DepthBin | None = None

    ordered = trajectory.sort_values("time", kind="stable").reset_index(drop=True)
    for idx, z_value in enumerate(ordered["z"].to_numpy(dtype=float)):
        next_bin = _find_depth_bin(float(z_value), config.bins)
        if next_bin is None:
            if start_idx is not None and current_bin is not None:
                segment = ordered.iloc[start_idx:idx].copy().reset_index(drop=True)
                segment["depth_bin"] = current_bin.label
                segment["depth_bin_interval"] = current_bin.interval_label
                segments.append(segment)
            start_idx = None
            current_bin = None
            continue

        if start_idx is None:
            start_idx = idx
            current_bin = next_bin
            continue

        if current_bin is not None and next_bin.label != current_bin.label:
            segment = ordered.iloc[start_idx:idx].copy().reset_index(drop=True)
            segment["depth_bin"] = current_bin.label
            segment["depth_bin_interval"] = current_bin.interval_label
            segments.append(segment)
            start_idx = idx
            current_bin = next_bin

    if start_idx is not None and current_bin is not None:
        segment = ordered.iloc[start_idx:].copy().reset_index(drop=True)
        segment["depth_bin"] = current_bin.label
        segment["depth_bin_interval"] = current_bin.interval_label
        segments.append(segment)

    return segments


def _apply_depth_bin_splitting(
    trajectories: list[pd.DataFrame],
    *,
    config: DepthBinConfig,
) -> list[pd.DataFrame]:
    if not config.enabled:
        return trajectories

    split: list[pd.DataFrame] = []
    trajectory_iterator = (
        tqdm(trajectories, desc="Splitting trajectories by depth bin", unit="traj")
        if len(trajectories) > 1
        else trajectories
    )
    for trajectory in trajectory_iterator:
        split.extend(_split_trajectory_by_depth_bin(trajectory, config=config))

    return [trajectory for trajectory in split if not trajectory.empty]


def _apply_region_selection(
    trajectories: list[pd.DataFrame],
    *,
    config: RegionSelectionConfig,
) -> list[pd.DataFrame]:
    return apply_region_selection(trajectories, config=config)


def _convert_rtraj_to_processed_trajectories(
    config: dict[str, Any],
) -> tuple[list[pd.DataFrame], DepthBinConfig, RtrajConversionSummary]:
    files = _resolve_input_files(config)
    parking_config = _resolve_parking_depth_config(config)
    depth_bin_config = _resolve_depth_bin_config(config)
    region_config = resolve_region_selection_config(config)
    resample_config = resolve_resample_config(config)
    conversion_summary = RtrajConversionSummary(
        input_files=len(files),
        depth_bins_enabled=depth_bin_config.enabled,
        region_names_or_labels=region_config.names_or_labels,
        region_selection_mode=region_config.selection_mode,
        resample_frequency=resample_config.frequency,
    )

    print(f"Resolved {len(files)} ARGO Rtraj NetCDF file(s)")
    file_iterator = tqdm(files, desc="Reading ARGO Rtraj files", unit="file") if len(files) > 1 else files

    trajectories: list[pd.DataFrame] = []
    total_rows = 0
    total_valid_rows = 0
    total_unmapped_z = 0
    total_missing_z = 0
    for file_path in file_iterator:
        trajectory, summary = _read_rtraj_file(file_path, parking_config=parking_config)
        total_rows += summary["rows"]
        total_valid_rows += summary["valid_rows"]
        total_unmapped_z += summary["unmapped_z"]
        total_missing_z += summary["missing_z"]
        if not trajectory.empty:
            trajectories.append(trajectory)

    print(f"Buffered {len(trajectories)} Rtraj platform trajectory/trajectories")
    print(f"Kept {total_valid_rows}/{total_rows} rows with valid time/lat/lon")
    print(
        "Parking-pressure z mapping: "
        f"{total_unmapped_z} unmapped row(s), {total_missing_z} final missing z row(s)"
    )
    conversion_summary.raw_trajectories = len(trajectories)
    conversion_summary.raw_platforms = _count_platforms(trajectories)
    conversion_summary.raw_rows = int(total_rows)
    conversion_summary.valid_rows = int(total_valid_rows)
    conversion_summary.unmapped_z_rows = int(total_unmapped_z)
    conversion_summary.missing_z_rows = int(total_missing_z)

    if depth_bin_config.enabled:
        labels = ", ".join(depth_bin.label for depth_bin in depth_bin_config.bins)
        print(f"Splitting trajectories into depth bin(s): {labels}")
    trajectories = _apply_depth_bin_splitting(trajectories, config=depth_bin_config)
    if depth_bin_config.enabled:
        print(f"Built {len(trajectories)} depth-bin trajectory segment(s)")
        conversion_summary.depth_bin_segments = len(trajectories)
        conversion_summary.depth_bin_counts = _summarize_by_depth_bin(trajectories)

    if region_config.names_or_labels:
        print(
            "Applying region selection "
            f"({region_config.selection_mode}) for: {', '.join(region_config.names_or_labels)}"
        )
    conversion_summary.region_input_trajectories = len(trajectories)
    conversion_summary.region_input_platforms = _count_platforms(trajectories)
    trajectories = _apply_region_selection(trajectories, config=region_config)
    print(f"Kept {len(trajectories)} trajectory/trajectories after region selection")
    conversion_summary.region_kept_trajectories = len(trajectories)
    conversion_summary.region_kept_platforms = _count_platforms(trajectories)

    if resample_config.frequency:
        print(f"Resampling trajectories at frequency: {resample_config.frequency}")
    if resample_config.shared_time and resample_config.reference_time is not None:
        print(f"Using shared time grid from reference time: {resample_config.reference_time.isoformat()}")
    elif resample_config.reference_time is not None:
        print(f"Anchoring resampling to reference time: {resample_config.reference_time.isoformat()}")
    if resample_config.shift_start_to_reference and resample_config.reference_time is not None:
        print(f"Shifting trajectory starts to reference time: {resample_config.reference_time.isoformat()}")
    trajectories = apply_resampling(
        trajectories,
        config=resample_config,
        non_interpolated_columns=NON_INTERPOLATED_COLUMNS,
    )

    pre_drop_count = len(trajectories)
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    dropped_empty = pre_drop_count - len(trajectories)
    if dropped_empty > 0:
        print(f"Dropped {dropped_empty} empty trajectory/trajectories after resampling")
    conversion_summary.resample_dropped_empty = int(dropped_empty)
    conversion_summary.final_trajectories = len(trajectories)
    conversion_summary.final_platforms = _count_platforms(trajectories)
    conversion_summary.final_observations = _count_observations(trajectories)

    if not trajectories:
        raise ValueError("No trajectories were produced after Rtraj filtering and processing")

    return trajectories, depth_bin_config, conversion_summary


def convert_rtraj_to_dataframe(config: dict[str, Any]) -> list[pd.DataFrame]:
    trajectories, _, _ = _convert_rtraj_to_processed_trajectories(config)
    normalized = normalize_trajectories(trajectories)
    if not normalized:
        raise ValueError("No trajectories were produced after Rtraj filtering and processing")

    return normalized


def _build_rtraj_dataset(
    trajectories: list[pd.DataFrame],
    *,
    dataset_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    print("Building output dataset")
    attrs = dict(DEFAULT_DATASET_ATTRS)
    if dataset_attrs:
        attrs.update(dataset_attrs)

    return build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
        dataset_attrs=attrs,
    )


def _depth_bin_output_path(output_path: Path, depth_bin_label: str) -> Path:
    suffix = output_path.suffix
    if suffix:
        return output_path.with_name(f"{output_path.stem}_{depth_bin_label}{suffix}")
    return output_path.with_name(f"{output_path.name}_{depth_bin_label}")


def _depth_bin_attrs(depth_bin: DepthBin) -> dict[str, Any]:
    return {
        "depth_bins_enabled": True,
        "depth_bin_label": depth_bin.label,
        "depth_bin_min": float(depth_bin.min_value),
        "depth_bin_max": "+inf" if depth_bin.max_value is None else float(depth_bin.max_value),
        "depth_bin_interval": depth_bin.interval_label,
    }


def _write_dataset_to_zarr(ds: xr.Dataset, output_path: Path) -> None:
    encoding = build_zarr_encoding(ds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode="w", encoding=encoding)


def _print_conversion_summary(summary: RtrajConversionSummary) -> None:
    print("")
    print("ARGO Rtraj conversion summary")
    print(f"  input files: {summary.input_files}")
    print(f"  raw trajectories: {summary.raw_trajectories} ({summary.raw_platforms} platform(s))")
    print(f"  valid rows: {summary.valid_rows}/{summary.raw_rows}")
    print(
        "  parking-pressure z mapping: "
        f"{summary.unmapped_z_rows} unmapped row(s), {summary.missing_z_rows} final missing z row(s)"
    )

    if summary.depth_bins_enabled:
        print(f"  depth-bin segments before region selection: {summary.depth_bin_segments}")
        for label, counts in summary.depth_bin_counts.items():
            print(
                f"    {label}: {counts['trajectories']} trajectory segment(s), "
                f"{counts['platforms']} platform(s), {counts['observations']} observation(s)"
            )
    else:
        print("  depth-bin splitting: disabled")

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

    if summary.output_counts:
        print("  output datasets:")
        for label, counts in summary.output_counts.items():
            print(
                f"    {label}: {counts['trajectories']} trajectory segment(s), "
                f"{counts['platforms']} platform(s), {counts['observations']} observation(s), "
                f"path={counts['path']}"
            )


def convert_rtraj_to_zarr(config: dict[str, Any]) -> xr.Dataset | dict[str, xr.Dataset]:
    trajectories, depth_bin_config, conversion_summary = _convert_rtraj_to_processed_trajectories(config)
    output_path = _resolve_output_path(config)

    if depth_bin_config.enabled:
        datasets: dict[str, xr.Dataset] = {}
        for depth_bin in depth_bin_config.bins:
            bin_trajectories = [
                trajectory.reset_index(drop=True)
                for trajectory in trajectories
                if "depth_bin" in trajectory.columns and str(trajectory["depth_bin"].iloc[0]) == depth_bin.label
            ]
            if not bin_trajectories:
                print(f"Skipping empty depth bin: {depth_bin.label}")
                continue

            normalized = normalize_trajectories(bin_trajectories)
            ds = _build_rtraj_dataset(normalized, dataset_attrs=_depth_bin_attrs(depth_bin))
            current_output_path = _depth_bin_output_path(output_path, depth_bin.label)
            _write_dataset_to_zarr(ds, current_output_path)
            conversion_summary.output_counts[depth_bin.label] = {
                "path": str(current_output_path),
                "trajectories": len(normalized),
                "platforms": _count_platforms(normalized),
                "observations": _count_observations(normalized),
            }
            datasets[depth_bin.label] = ds

        if not datasets:
            raise ValueError("No depth-bin Zarr datasets were produced after Rtraj filtering and processing")

        _print_conversion_summary(conversion_summary)
        print("ARGO Rtraj conversion completed")
        return datasets

    normalized = normalize_trajectories(trajectories)
    if not normalized:
        raise ValueError("No trajectories were produced after Rtraj filtering and processing")

    ds = _build_rtraj_dataset(normalized, dataset_attrs={"depth_bins_enabled": False})
    _write_dataset_to_zarr(ds, output_path)
    conversion_summary.output_counts["all"] = {
        "path": str(output_path),
        "trajectories": len(normalized),
        "platforms": _count_platforms(normalized),
        "observations": _count_observations(normalized),
    }
    _print_conversion_summary(conversion_summary)
    print("ARGO Rtraj conversion completed")
    return ds


def run_conversion(config_path: str | Path) -> xr.Dataset | dict[str, xr.Dataset]:
    config = load_config(config_path)
    return convert_rtraj_to_zarr(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args.config)


if __name__ == "__main__":
    main()
