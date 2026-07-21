from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import glob
from pathlib import Path
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
    filter_trajectories_by_min_duration,
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
    "source": "RAFOS/SOFAR subsurface float NetCDF conversion",
    "z_source": "RAFOS pressure",
    "z_approximation": (
        "z is copied from RAFOS pressure in dbar, with positive values and "
        "no pressure-to-geometric-depth conversion."
    ),
}
TRAJECTORY_LEVEL_COLUMNS = {
    "platform_code",
    "floatID",
    "trajectoryID",
    "float_type",
    "depth_bin",
    "depth_bin_interval",
}
NON_INTERPOLATED_COLUMNS = {
    "platform_code",
    "floatID",
    "trajectoryID",
    "float_type",
    "depth_bin",
    "depth_bin_interval",
}
INTERNAL_OUTPUT_COLUMNS = {
    "surface_date",
    "source_file",
    "source_index",
    "depth_bin_filled",
    "depth_bin_repaired",
}


@dataclass(frozen=True)
class OutputConfig:
    zarr_path: Path
    overwrite: bool = True


@dataclass(frozen=True)
class SurfaceConfig:
    clip_after_surface_date: bool = True


@dataclass(frozen=True)
class DepthBin:
    label: str
    min_value: float
    max_value: float | None

    @property
    def interval_label(self) -> str:
        upper = "+inf" if self.max_value is None else f"{self.max_value:g}"
        return f"[{self.min_value:g}, {upper})"


@dataclass(frozen=True)
class MissingDepthConfig:
    strategy: str
    max_fill_points: int
    fill_between_same_bin_only: bool


@dataclass(frozen=True)
class IsolatedDepthBinConfig:
    enabled: bool
    max_run_points: int
    require_same_neighbor_bin: bool


@dataclass(frozen=True)
class DepthBinConfig:
    enabled: bool
    output_mode: str
    bins: tuple[DepthBin, ...]
    missing_depth: MissingDepthConfig
    isolated_outlier: IsolatedDepthBinConfig


@dataclass
class RafosConversionSummary:
    input_files: int = 0
    raw_rows: int = 0
    valid_rows: int = 0
    raw_trajectories: int = 0
    valid_trajectories: int = 0
    skipped_empty_after_surface_clip: int = 0
    surface_clipped_rows: int = 0
    multi_surface_date_trajectories: int = 0
    multi_float_type_trajectories: int = 0
    depth_bins_enabled: bool = False
    depth_bin_segments: int = 0
    depth_bin_unassigned_points: int = 0
    depth_bin_fill_count: int = 0
    depth_bin_repair_count: int = 0
    depth_bin_transition_count: int = 0
    depth_bin_counts: dict[str, int] = field(default_factory=dict)
    region_names_or_labels: tuple[str, ...] = ()
    region_selection_mode: str = ""
    region_input_trajectories: int = 0
    region_kept_trajectories: int = 0
    resample_frequency: str | None = None
    resample_duration_dropped: int = 0
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
        description="Convert RAFOS/SOFAR subsurface float NetCDF files into Parcels-compatible trajectory Zarr datasets."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the RAFOS conversion YAML configuration file.",
    )
    return parser


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_input_files(config: dict[str, Any]) -> list[Path]:
    input_cfg = config.get("input", {}) or {}
    files: list[Path] = []

    for key in ("netcdf_files", "rafos_files"):
        for item in _as_list(input_cfg.get(key)):
            files.append(Path(item))

    for key in ("netcdf_glob", "rafos_glob"):
        for pattern in _as_list(input_cfg.get(key)):
            files.extend(Path(path) for path in glob.glob(str(pattern)))

    input_dirs: list[Any] = []
    for key in ("netcdf_dir", "netcdf_dirs", "rafos_dir", "rafos_dirs"):
        input_dirs.extend(_as_list(input_cfg.get(key)))

    pattern = str(input_cfg.get("pattern", "*.nc"))
    for input_dir in input_dirs:
        files.extend(sorted(Path(input_dir).glob(pattern)))

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError("No RAFOS NetCDF files were found from the provided configuration.")

    missing = [path for path in unique_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"The following RAFOS NetCDF files were not found: {missing_str}")

    return unique_files


def _resolve_output_config(config: dict[str, Any]) -> OutputConfig:
    raw = config.get("output", {}) or {}
    zarr_path = raw.get("zarr_path") or raw.get("path")
    if not zarr_path:
        raise ValueError("The RAFOS conversion config must define output.zarr_path or output.path.")
    return OutputConfig(
        zarr_path=Path(zarr_path),
        overwrite=bool(raw.get("overwrite", True)),
    )


def _resolve_surface_config(config: dict[str, Any]) -> SurfaceConfig:
    raw = (config.get("processing", {}) or {}).get("surface", {}) or {}
    return SurfaceConfig(
        clip_after_surface_date=bool(raw.get("clip_after_surface_date", True)),
    )


def _resolve_region_config(config: dict[str, Any]):
    raw = config.get("regions", {}) or (config.get("processing", {}) or {}).get("regions", {}) or {}
    return resolve_region_selection_config({"processing": {"regions": raw}})


def _resolve_resample_config(config: dict[str, Any]):
    raw = config.get("resample", {}) or (config.get("processing", {}) or {}).get("resample", {}) or {}
    return resolve_resample_config({"processing": {"resample": raw}})


def _resolve_depth_bin_config(config: dict[str, Any]) -> DepthBinConfig:
    raw = config.get("depth_bins", {}) or {}
    enabled = bool(raw.get("enabled", False))
    output_mode = str(raw.get("output_mode", "per_bin"))
    if output_mode not in {"per_bin", "single"}:
        raise ValueError("depth_bins.output_mode must be 'per_bin' or 'single'")

    missing_raw = raw.get("missing_depth", {}) or {}
    missing_depth = MissingDepthConfig(
        strategy=str(missing_raw.get("strategy", "bounded_neighbor")),
        max_fill_points=int(missing_raw.get("max_fill_points", 2)),
        fill_between_same_bin_only=bool(missing_raw.get("fill_between_same_bin_only", True)),
    )
    if missing_depth.strategy not in {"none", "bounded_neighbor"}:
        raise ValueError("depth_bins.missing_depth.strategy must be 'none' or 'bounded_neighbor'")
    if missing_depth.max_fill_points < 0:
        raise ValueError("depth_bins.missing_depth.max_fill_points must be non-negative")

    isolated_raw = raw.get("isolated_outlier", {}) or {}
    isolated_outlier = IsolatedDepthBinConfig(
        enabled=bool(isolated_raw.get("enabled", False)),
        max_run_points=int(isolated_raw.get("max_run_points", 1)),
        require_same_neighbor_bin=bool(isolated_raw.get("require_same_neighbor_bin", True)),
    )
    if isolated_outlier.max_run_points < 0:
        raise ValueError("depth_bins.isolated_outlier.max_run_points must be non-negative")

    if not enabled:
        return DepthBinConfig(
            enabled=False,
            output_mode=output_mode,
            bins=(),
            missing_depth=missing_depth,
            isolated_outlier=isolated_outlier,
        )

    bins: list[DepthBin] = []
    for idx, item in enumerate(raw.get("bins", []) or []):
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError(f"depth_bins.bins[{idx}].label is required")
        min_value = float(item.get("min", 0.0))
        max_raw = item.get("max")
        max_value = None if max_raw is None else float(max_raw)
        if not np.isfinite(min_value):
            raise ValueError(f"depth_bins.bins[{idx}].min must be finite")
        if max_value is not None and (not np.isfinite(max_value) or max_value <= min_value):
            raise ValueError(f"depth_bins.bins[{idx}].max must be null or greater than min")
        bins.append(DepthBin(label=label, min_value=min_value, max_value=max_value))

    if not bins:
        raise ValueError("depth_bins.bins must contain at least one bin when depth_bins.enabled is true")

    return DepthBinConfig(
        enabled=True,
        output_mode=output_mode,
        bins=tuple(bins),
        missing_depth=missing_depth,
        isolated_outlier=isolated_outlier,
    )


def _decode_byte(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _decode_string_value(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        value = arr.item()
        if isinstance(value, (float, int, np.number)):
            numeric = float(value)
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        return _decode_byte(value).strip()
    if arr.dtype.kind in {"S", "U"} and arr.dtype.itemsize in {1, 4}:
        return "".join(_decode_byte(item) for item in arr.ravel()).strip()
    return _decode_byte(arr.ravel()[0]).strip() if arr.size else ""


def _normalize_identifier(value: Any, *, missing: str = "N/A") -> str:
    text = _decode_string_value(value).strip()
    if text.lower() in {"", "nan", "nat", "none", "na", "n/a"}:
        return missing
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _coerce_times(values: Any) -> pd.Series:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.Series(pd.to_datetime(arr.ravel(), errors="coerce")).dt.tz_localize(None)
    decoded = pd.to_datetime(pd.Series(arr.ravel()), utc=True, errors="coerce")
    return decoded.dt.tz_convert(None)


def _required_series(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Missing required RAFOS variable: {name}")
    return np.asarray(ds[name].values).ravel()


def _optional_series(ds: xr.Dataset, name: str, length: int, default: Any) -> np.ndarray:
    if name not in ds:
        return np.full(length, default, dtype=object)
    values = np.asarray(ds[name].values).ravel()
    if len(values) == length:
        return values
    if len(values) == 1:
        return np.full(length, values[0], dtype=object)
    raise ValueError(f"RAFOS variable {name!r} has length {len(values)}, expected {length}")


def _first_non_missing(values: pd.Series, *, default: str = "N/A") -> str:
    for value in values:
        text = _normalize_identifier(value, missing="")
        if text:
            return text
    return default


def _unique_non_missing_count(values: pd.Series) -> int:
    return len({_normalize_identifier(value, missing="") for value in values if _normalize_identifier(value, missing="")})


def _platform_code(float_id: str, trajectory_id: str) -> str:
    return f"{float_id}::{trajectory_id}"


def _read_file_trajectories(
    file_path: Path,
    *,
    surface_config: SurfaceConfig,
) -> tuple[list[pd.DataFrame], dict[str, int]]:
    with xr.open_dataset(file_path, decode_times=True, decode_timedelta=False, mask_and_scale=True) as ds:
        trajectory_id_raw = _required_series(ds, "trajectoryID")
        n_rows = len(trajectory_id_raw)
        float_id_raw = _required_series(ds, "floatID")
        time = _coerce_times(_required_series(ds, "time"))
        lat = pd.to_numeric(pd.Series(_required_series(ds, "latitude")), errors="coerce")
        lon = pd.to_numeric(pd.Series(_required_series(ds, "longitude")), errors="coerce")
        pressure = pd.to_numeric(pd.Series(_required_series(ds, "pressure")), errors="coerce")
        float_type_raw = _optional_series(ds, "float_type", n_rows, "N/A")
        surface_date = _coerce_times(_optional_series(ds, "surface_date", n_rows, np.datetime64("NaT")))

        lengths = {
            "trajectoryID": n_rows,
            "floatID": len(float_id_raw),
            "time": len(time),
            "latitude": len(lat),
            "longitude": len(lon),
            "pressure": len(pressure),
            "float_type": len(float_type_raw),
            "surface_date": len(surface_date),
        }
        min_length = min(lengths.values())
        if len(set(lengths.values())) != 1:
            bad = ", ".join(f"{name}={length}" for name, length in sorted(lengths.items()))
            raise ValueError(f"RAFOS variables in {file_path} have inconsistent row lengths: {bad}")

        frame = pd.DataFrame(
            {
                "source_file": str(file_path),
                "source_index": np.arange(min_length, dtype=np.int64),
                "floatID": [_normalize_identifier(value) for value in float_id_raw[:min_length]],
                "trajectoryID": [_normalize_identifier(value) for value in trajectory_id_raw[:min_length]],
                "float_type": [_normalize_identifier(value) for value in float_type_raw[:min_length]],
                "time": time.iloc[:min_length].reset_index(drop=True),
                "lat": lat.iloc[:min_length].reset_index(drop=True),
                "lon": lon.iloc[:min_length].reset_index(drop=True),
                "z": pressure.iloc[:min_length].reset_index(drop=True),
                "surface_date": surface_date.iloc[:min_length].reset_index(drop=True),
            }
        )
        frame["platform_code"] = [
            _platform_code(float_id, trajectory_id)
            for float_id, trajectory_id in zip(frame["floatID"], frame["trajectoryID"])
        ]

        summary = {
            "raw_rows": int(len(frame)),
            "valid_rows": 0,
            "raw_trajectories": int(frame[["floatID", "trajectoryID"]].drop_duplicates().shape[0]),
            "valid_trajectories": 0,
            "skipped_empty_after_surface_clip": 0,
            "surface_clipped_rows": 0,
            "multi_surface_date_trajectories": 0,
            "multi_float_type_trajectories": 0,
        }

        trajectories: list[pd.DataFrame] = []
        grouped = frame.groupby(["floatID", "trajectoryID"], sort=True, dropna=False)
        for (_, _), group in grouped:
            current = group.sort_values(["time", "source_index"], kind="stable").reset_index(drop=True)
            valid = current["time"].notna() & current["lat"].notna() & current["lon"].notna()
            current = current.loc[valid].reset_index(drop=True)
            summary["valid_rows"] += int(len(current))

            if current.empty:
                summary["skipped_empty_after_surface_clip"] += 1
                continue

            if _unique_non_missing_count(current["float_type"]) > 1:
                summary["multi_float_type_trajectories"] += 1
            current.loc[:, "float_type"] = _first_non_missing(current["float_type"])

            surface_dates = pd.to_datetime(current["surface_date"], errors="coerce").dropna()
            if len(surface_dates.drop_duplicates()) > 1:
                summary["multi_surface_date_trajectories"] += 1
            if surface_config.clip_after_surface_date and not surface_dates.empty:
                surface_time = surface_dates.min()
                before_count = len(current)
                current = current.loc[current["time"] < surface_time].reset_index(drop=True)
                summary["surface_clipped_rows"] += int(before_count - len(current))

            if current.empty:
                summary["skipped_empty_after_surface_clip"] += 1
                continue

            float_id = _first_non_missing(current["floatID"])
            trajectory_id = _first_non_missing(current["trajectoryID"])
            current.loc[:, "floatID"] = float_id
            current.loc[:, "trajectoryID"] = trajectory_id
            current.loc[:, "platform_code"] = _platform_code(float_id, trajectory_id)
            trajectories.append(current)
            summary["valid_trajectories"] += 1

    return trajectories, summary


def _find_depth_bin(value: Any, config: DepthBinConfig) -> DepthBin | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(float(numeric)):
        return None

    depth = float(numeric)
    for depth_bin in config.bins:
        if depth < depth_bin.min_value:
            continue
        if depth_bin.max_value is None or depth < depth_bin.max_value:
            return depth_bin
    return None


def _depth_bin_metadata(label: str | None, config: DepthBinConfig) -> tuple[str | None, str | None]:
    if label is None:
        return None, None
    for depth_bin in config.bins:
        if depth_bin.label == label:
            return depth_bin.label, depth_bin.interval_label
    return None, None


def _fill_missing_depth_bins(segment: pd.DataFrame, config: DepthBinConfig) -> tuple[pd.DataFrame, int]:
    missing_config = config.missing_depth
    out = segment.copy()
    out["depth_bin_filled"] = False
    if missing_config.strategy == "none" or missing_config.max_fill_points == 0 or out.empty:
        return out, 0

    labels = out["depth_bin"].where(out["depth_bin"].notna(), None).tolist()
    filled_count = 0
    idx = 0
    while idx < len(labels):
        if labels[idx] is not None:
            idx += 1
            continue

        start = idx
        while idx < len(labels) and labels[idx] is None:
            idx += 1
        stop = idx
        run_length = stop - start
        if run_length > missing_config.max_fill_points:
            continue

        previous_label = labels[start - 1] if start > 0 else None
        next_label = labels[stop] if stop < len(labels) else None
        fill_label: str | None = None
        if previous_label is not None and next_label is not None:
            if previous_label == next_label:
                fill_label = str(previous_label)
            elif not missing_config.fill_between_same_bin_only:
                fill_label = str(previous_label)
        elif previous_label is not None:
            fill_label = str(previous_label)
        elif next_label is not None:
            fill_label = str(next_label)

        if fill_label is None:
            continue

        label, interval = _depth_bin_metadata(fill_label, config)
        if label is None:
            continue
        out.loc[start : stop - 1, "depth_bin"] = label
        out.loc[start : stop - 1, "depth_bin_interval"] = interval
        out.loc[start : stop - 1, "depth_bin_filled"] = True
        for fill_idx in range(start, stop):
            labels[fill_idx] = label
        filled_count += run_length

    return out, filled_count


def _repair_isolated_depth_bin_runs(segment: pd.DataFrame, config: DepthBinConfig) -> tuple[pd.DataFrame, int]:
    repair_config = config.isolated_outlier
    out = segment.copy()
    out["depth_bin_repaired"] = False
    if (
        not repair_config.enabled
        or repair_config.max_run_points == 0
        or out.empty
        or "depth_bin" not in out.columns
    ):
        return out, 0

    labels = out["depth_bin"].where(out["depth_bin"].notna(), None).tolist()
    repaired_count = 0
    idx = 0
    while idx < len(labels):
        current_label = labels[idx]
        start = idx
        while idx < len(labels) and labels[idx] == current_label:
            idx += 1
        stop = idx

        if current_label is None:
            continue
        run_length = stop - start
        if run_length > repair_config.max_run_points:
            continue

        previous_label = labels[start - 1] if start > 0 else None
        next_label = labels[stop] if stop < len(labels) else None
        if previous_label is None or next_label is None:
            continue
        if previous_label == current_label or next_label == current_label:
            continue
        if repair_config.require_same_neighbor_bin and previous_label != next_label:
            continue

        fill_label = str(previous_label)
        label, interval = _depth_bin_metadata(fill_label, config)
        if label is None:
            continue
        out.loc[start : stop - 1, "depth_bin"] = label
        out.loc[start : stop - 1, "depth_bin_interval"] = interval
        out.loc[start : stop - 1, "depth_bin_repaired"] = True
        for repair_idx in range(start, stop):
            labels[repair_idx] = label
        repaired_count += run_length

    return out, repaired_count


def _assign_depth_bins(segment: pd.DataFrame, config: DepthBinConfig) -> tuple[pd.DataFrame, int, int]:
    out = segment.copy().reset_index(drop=True)
    out["depth_bin"] = None
    out["depth_bin_interval"] = None
    out["depth_bin_filled"] = False
    out["depth_bin_repaired"] = False
    if not config.enabled:
        return out, 0, 0

    for idx, depth_value in enumerate(out["z"]):
        depth_bin = _find_depth_bin(depth_value, config)
        if depth_bin is None:
            continue
        out.at[idx, "depth_bin"] = depth_bin.label
        out.at[idx, "depth_bin_interval"] = depth_bin.interval_label

    out, fill_count = _fill_missing_depth_bins(out, config)
    out, repair_count = _repair_isolated_depth_bin_runs(out, config)
    return out, fill_count, repair_count


def _split_segment_by_depth_bin(segment: pd.DataFrame, *, min_segment_points: int) -> list[pd.DataFrame]:
    if segment.empty or "depth_bin" not in segment.columns:
        return [segment.reset_index(drop=True)] if len(segment) >= min_segment_points else []

    output: list[pd.DataFrame] = []
    current_indices: list[int] = []
    current_label: str | None = None

    for idx, label_raw in enumerate(segment["depth_bin"].tolist()):
        label = None if pd.isna(label_raw) else str(label_raw)
        if label is None:
            if len(current_indices) >= min_segment_points:
                output.append(segment.iloc[current_indices].copy().reset_index(drop=True))
            current_indices = []
            current_label = None
            continue

        if current_label is None or label == current_label:
            current_indices.append(idx)
            current_label = label
            continue

        if len(current_indices) >= min_segment_points:
            output.append(segment.iloc[current_indices].copy().reset_index(drop=True))
        current_indices = [idx]
        current_label = label

    if len(current_indices) >= min_segment_points:
        output.append(segment.iloc[current_indices].copy().reset_index(drop=True))

    return output


def apply_depth_bin_segmentation(
    trajectories: list[pd.DataFrame],
    config: DepthBinConfig,
    *,
    min_segment_points: int,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    if not config.enabled:
        return trajectories, {
            "depth_bin_enabled": False,
            "depth_bin_segments": int(len(trajectories)),
            "depth_bin_unassigned_points": 0,
            "depth_bin_fill_count": 0,
            "depth_bin_repair_count": 0,
            "depth_bin_transition_count": 0,
            "depth_bin_counts": {},
        }

    output: list[pd.DataFrame] = []
    unassigned_points = 0
    fill_count = 0
    repair_count = 0
    transition_count = 0
    bin_counts: dict[str, int] = {}

    for trajectory in trajectories:
        binned, segment_fill_count, segment_repair_count = _assign_depth_bins(trajectory, config)
        fill_count += int(segment_fill_count)
        repair_count += int(segment_repair_count)
        unassigned_points += int(binned["depth_bin"].isna().sum())
        labels = [str(value) for value in binned["depth_bin"].dropna().tolist()]
        transition_count += int(sum(1 for previous, current in zip(labels[:-1], labels[1:]) if previous != current))
        for label in labels:
            bin_counts[label] = bin_counts.get(label, 0) + 1
        output.extend(_split_segment_by_depth_bin(binned, min_segment_points=min_segment_points))

    return output, {
        "depth_bin_enabled": True,
        "depth_bin_segments": int(len(output)),
        "depth_bin_unassigned_points": int(unassigned_points),
        "depth_bin_fill_count": int(fill_count),
        "depth_bin_repair_count": int(repair_count),
        "depth_bin_transition_count": int(transition_count),
        "depth_bin_counts": dict(sorted(bin_counts.items())),
    }


def _count_platforms(trajectories: list[pd.DataFrame]) -> int:
    platforms = {
        str(trajectory["platform_code"].iloc[0])
        for trajectory in trajectories
        if not trajectory.empty and "platform_code" in trajectory.columns
    }
    return len(platforms)


def _count_observations(trajectories: list[pd.DataFrame]) -> int:
    return int(sum(len(trajectory) for trajectory in trajectories))


def _min_segment_points(config: dict[str, Any]) -> int:
    raw = config.get("segmentation", {}) or {}
    value = int(raw.get("min_segment_points", 1))
    if value <= 0:
        raise ValueError("segmentation.min_segment_points must be a positive integer")
    return value


def _prepare_output_trajectories(trajectories: list[pd.DataFrame]) -> list[pd.DataFrame]:
    prepared: list[pd.DataFrame] = []
    for trajectory in trajectories:
        if trajectory.empty:
            continue
        drop_columns = [column for column in INTERNAL_OUTPUT_COLUMNS if column in trajectory.columns]
        current = trajectory.drop(columns=drop_columns).sort_values("time", kind="stable").reset_index(drop=True)
        if not current.empty:
            prepared.append(current)
    return prepared


def _convert_rafos_to_processed_trajectories(
    config: dict[str, Any],
) -> tuple[list[pd.DataFrame], DepthBinConfig, OutputConfig, RafosConversionSummary]:
    files = _resolve_input_files(config)
    output_config = _resolve_output_config(config)
    surface_config = _resolve_surface_config(config)
    region_config = _resolve_region_config(config)
    resample_config = _resolve_resample_config(config)
    depth_bin_config = _resolve_depth_bin_config(config)
    min_segment_points = _min_segment_points(config)

    summary = RafosConversionSummary(
        input_files=len(files),
        depth_bins_enabled=depth_bin_config.enabled,
        region_names_or_labels=region_config.names_or_labels,
        region_selection_mode=region_config.selection_mode,
        resample_frequency=resample_config.frequency,
    )

    print(f"Resolved {len(files)} RAFOS NetCDF file(s)")
    file_iterator = tqdm(files, desc="Reading RAFOS files", unit="file") if len(files) > 1 else files
    trajectories: list[pd.DataFrame] = []
    for file_path in file_iterator:
        file_trajectories, file_summary = _read_file_trajectories(
            file_path,
            surface_config=surface_config,
        )
        trajectories.extend(file_trajectories)
        for key in (
            "raw_rows",
            "valid_rows",
            "raw_trajectories",
            "valid_trajectories",
            "skipped_empty_after_surface_clip",
            "surface_clipped_rows",
            "multi_surface_date_trajectories",
            "multi_float_type_trajectories",
        ):
            setattr(summary, key, getattr(summary, key) + int(file_summary[key]))

    print(f"Buffered {len(trajectories)} RAFOS trajectory/trajectories")
    print(f"Kept {summary.valid_rows}/{summary.raw_rows} rows with valid time/lat/lon")
    if surface_config.clip_after_surface_date:
        print(f"Surface-date clipping removed {summary.surface_clipped_rows} row(s)")

    if depth_bin_config.enabled:
        labels = ", ".join(depth_bin.label for depth_bin in depth_bin_config.bins)
        print(f"Splitting trajectories into depth bin(s): {labels}")
    trajectories, depth_summary = apply_depth_bin_segmentation(
        trajectories,
        depth_bin_config,
        min_segment_points=min_segment_points,
    )
    summary.depth_bin_segments = int(depth_summary["depth_bin_segments"])
    summary.depth_bin_unassigned_points = int(depth_summary["depth_bin_unassigned_points"])
    summary.depth_bin_fill_count = int(depth_summary["depth_bin_fill_count"])
    summary.depth_bin_repair_count = int(depth_summary["depth_bin_repair_count"])
    summary.depth_bin_transition_count = int(depth_summary["depth_bin_transition_count"])
    summary.depth_bin_counts = dict(depth_summary["depth_bin_counts"])

    if region_config.names_or_labels:
        print(
            "Applying region selection "
            f"({region_config.selection_mode}) for: {', '.join(region_config.names_or_labels)}"
        )
    summary.region_input_trajectories = len(trajectories)
    trajectories = apply_region_selection(trajectories, config=region_config)
    summary.region_kept_trajectories = len(trajectories)
    print(f"Kept {len(trajectories)} trajectory segment(s) after region selection")

    trajectories, duration_dropped = filter_trajectories_by_min_duration(
        trajectories,
        resample_config.min_duration_days,
    )
    summary.resample_duration_dropped = int(duration_dropped)

    if resample_config.frequency:
        print(f"Resampling trajectories at frequency: {resample_config.frequency}")
    trajectories = apply_resampling(
        trajectories,
        config=resample_config,
        non_interpolated_columns=NON_INTERPOLATED_COLUMNS,
    )
    pre_drop_count = len(trajectories)
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    summary.resample_dropped_empty = int(pre_drop_count - len(trajectories))

    trajectories = _prepare_output_trajectories(trajectories)
    summary.final_trajectories = len(trajectories)
    summary.final_platforms = _count_platforms(trajectories)
    summary.final_observations = _count_observations(trajectories)
    if not trajectories:
        raise ValueError("No trajectories were produced after RAFOS filtering and processing")

    return trajectories, depth_bin_config, output_config, summary


def convert_rafos_to_dataframe(config: dict[str, Any]) -> list[pd.DataFrame]:
    trajectories, _, _, _ = _convert_rafos_to_processed_trajectories(config)
    normalized = normalize_trajectories(trajectories)
    if not normalized:
        raise ValueError("No trajectories were produced after RAFOS filtering and processing")
    return normalized


def _build_rafos_dataset(
    trajectories: list[pd.DataFrame],
    *,
    dataset_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    attrs = dict(DEFAULT_DATASET_ATTRS)
    if dataset_attrs:
        attrs.update(dataset_attrs)
    return build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
        dataset_attrs=attrs,
    )


def _write_dataset_to_zarr(ds: xr.Dataset, output_path: Path, *, overwrite: bool) -> None:
    encoding = build_zarr_encoding(ds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "w-"
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode=mode, encoding=encoding)


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


def _single_output_attrs(depth_bin_config: DepthBinConfig) -> dict[str, Any]:
    return {"depth_bins_enabled": bool(depth_bin_config.enabled)}


def _print_conversion_summary(summary: RafosConversionSummary) -> None:
    print("")
    print("RAFOS conversion summary")
    print(f"  input files: {summary.input_files}")
    print(f"  raw trajectories: {summary.raw_trajectories}")
    print(f"  valid trajectories after surface clipping: {summary.valid_trajectories}")
    print(f"  valid rows: {summary.valid_rows}/{summary.raw_rows}")
    print(f"  surface-clipped rows: {summary.surface_clipped_rows}")
    print(f"  skipped empty after surface clipping: {summary.skipped_empty_after_surface_clip}")
    print(f"  trajectories with multiple finite surface_date values: {summary.multi_surface_date_trajectories}")
    print(f"  trajectories with multiple float_type values: {summary.multi_float_type_trajectories}")
    if summary.depth_bins_enabled:
        print(f"  depth-bin segments before region selection: {summary.depth_bin_segments}")
        print(f"  depth-bin unassigned points: {summary.depth_bin_unassigned_points}")
        print(f"  depth-bin filled points: {summary.depth_bin_fill_count}")
        print(f"  depth-bin repaired points: {summary.depth_bin_repair_count}")
        print(f"  depth-bin transitions: {summary.depth_bin_transition_count}")
        print(f"  depth-bin point counts: {summary.depth_bin_counts}")
    else:
        print("  depth-bin splitting: disabled")
    if summary.region_names_or_labels:
        regions = ", ".join(summary.region_names_or_labels)
        print(f"  region selection: {summary.region_selection_mode} over {regions}")
    else:
        print("  region selection: disabled")
    if summary.resample_frequency:
        print(f"  resampling frequency: {summary.resample_frequency}")
    else:
        print("  resampling: disabled")
    print(f"  duration-filtered trajectories after region selection: {summary.resample_duration_dropped}")
    print(f"  empty trajectories dropped after resampling: {summary.resample_dropped_empty}")
    print(
        "  final trajectories: "
        f"{summary.final_trajectories} ({summary.final_platforms} platform(s), "
        f"{summary.final_observations} observation(s))"
    )
    for label, counts in summary.output_counts.items():
        print(
            f"  output {label}: {counts['trajectories']} trajectory segment(s), "
            f"{counts['platforms']} platform(s), {counts['observations']} observation(s), "
            f"path={counts['path']}"
        )


def convert_rafos_to_zarr(config: dict[str, Any]) -> xr.Dataset | dict[str, xr.Dataset]:
    trajectories, depth_bin_config, output_config, summary = _convert_rafos_to_processed_trajectories(config)

    if depth_bin_config.enabled and depth_bin_config.output_mode == "per_bin":
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
            normalized = normalize_trajectories(bin_trajectories, show_progress=False)
            ds = _build_rafos_dataset(normalized, dataset_attrs=_depth_bin_attrs(depth_bin))
            current_output_path = _depth_bin_output_path(output_config.zarr_path, depth_bin.label)
            _write_dataset_to_zarr(ds, current_output_path, overwrite=output_config.overwrite)
            summary.output_counts[depth_bin.label] = {
                "path": str(current_output_path),
                "trajectories": len(normalized),
                "platforms": _count_platforms(normalized),
                "observations": _count_observations(normalized),
            }
            datasets[depth_bin.label] = ds

        if not datasets:
            raise ValueError("No depth-bin Zarr datasets were produced after RAFOS filtering and processing")
        _print_conversion_summary(summary)
        print("RAFOS conversion completed")
        return datasets

    normalized = normalize_trajectories(trajectories, show_progress=False)
    if not normalized:
        raise ValueError("No trajectories were produced after RAFOS filtering and processing")
    ds = _build_rafos_dataset(normalized, dataset_attrs=_single_output_attrs(depth_bin_config))
    _write_dataset_to_zarr(ds, output_config.zarr_path, overwrite=output_config.overwrite)
    summary.output_counts["all"] = {
        "path": str(output_config.zarr_path),
        "trajectories": len(normalized),
        "platforms": _count_platforms(normalized),
        "observations": _count_observations(normalized),
    }
    _print_conversion_summary(summary)
    print("RAFOS conversion completed")
    return ds


def run_conversion(config_path: str | Path) -> xr.Dataset | dict[str, xr.Dataset]:
    config = load_config(config_path)
    return convert_rafos_to_zarr(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args.config)


if __name__ == "__main__":
    main()
