from __future__ import annotations

import argparse
import glob
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
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
    RegionSelectionConfig,
    ResampleConfig,
    apply_region_selection,
    apply_resampling,
    filter_trajectories_by_min_duration,
    normalize_trajectories,
    resolve_resample_config,
)
from kinematicparcels.tools.zarr_writer import (
    DEFAULT_TRAJECTORY_DATASET_ATTRS,
    build_dataset_from_trajectories,
    build_zarr_encoding,
)

SECONDS_PER_DAY = 86400.0
NON_INTERPOLATED_COLUMNS = {
    "platform_code",
    "source_index",
    "_qc_order",
    "cycle_number",
    "measurement_code",
    "depth",
    "qc_keep",
    "depth_bin_filled",
    "depth_bin_repaired",
}
DEFAULT_DATASET_ATTRS = {
    **DEFAULT_TRAJECTORY_DATASET_ATTRS,
    "source": "ARGO Rtraj NetCDF conversion",
    "z_source": "Argo parking pressure inferred per cycle",
    "z_approximation": (
        "z is approximated from Argo parking pressure in dbar, "
        "with positive values and no pressure-to-geometric-depth conversion."
    ),
}
TRAJECTORY_LEVEL_COLUMNS = {"platform_code", "depth_bin", "depth_bin_interval"}
INTERNAL_OUTPUT_COLUMNS = {
    "source_file",
    "source_index",
    "_qc_order",
    "cycle_number",
    "measurement_code",
    "position_qc",
    "time_qc",
    "qc_keep",
    "qc_drop_reasons",
    "depth_source",
    "depth_bin_filled",
    "depth_bin_repaired",
    "jump_qc_drop_reason",
    "jump_qc_block_id",
}
OBSERVATION_TIME_COLUMN = "_observation_time"
OBSERVATION_PRESSURE_COLUMN = "_observation_pressure"
OBSERVATION_INDEX_COLUMN = "_observation_index"
OBSERVATION_METADATA_KEYS = ("units", "long_name", "standard_name")


@dataclass(frozen=True)
class OutputConfig:
    zarr_path: Path
    write_zarr: bool
    overwrite: bool


@dataclass(frozen=True)
class MergeConfig:
    enabled: bool
    max_gap_points: int | None
    max_gap_duration_days: float | None
    max_bridge_speed_m_per_s: float | None
    max_bridge_vertical_rate_m_per_day: float | None


@dataclass(frozen=True)
class JumpQcConfig:
    enabled: bool
    max_speed_m_per_s: float
    auto_drop_enabled: bool
    max_block_points: int | None
    max_block_duration_days: float | None
    split_remaining_jumps: bool


@dataclass(frozen=True)
class TrajectoryFixConfig:
    one_per_cycle: bool
    require_finite_cycle: bool
    prefer_valid_position_qc: bool
    prefer_valid_time_qc: bool
    prefer_repeated_position: bool
    position_round_decimals: int
    tie_breaker: str


@dataclass(frozen=True)
class ParkingDepthConfig:
    mode: str
    fallback_value: float | None
    fill_missing: bool
    infer_from_park_window: bool
    pressure_variable: str
    fallback_pressure_variable: str | None
    percentile: float
    min_pressure: float | None


@dataclass(frozen=True)
class ObservationSourceConfig:
    adjusted: str | None
    fallback: str
    adjusted_qc: str | None = None
    fallback_qc: str | None = None
    valid_qc: tuple[str, ...] | None = None
    missing_qc: str = "accept"
    valid_min: float | None = None
    valid_max: float | None = None


@dataclass(frozen=True)
class ObservationConfig:
    enabled: bool
    time: ObservationSourceConfig
    pressure: ObservationSourceConfig
    variables: dict[str, ObservationSourceConfig]
    sample_at_fallback_depth: bool = False


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


@dataclass(frozen=True)
class RtrajConfig:
    path: Path
    raw: dict[str, Any]
    mode: str
    max_files: int | None
    input_files: list[Path]
    source_variables: dict[str, str]
    normalized_variables: dict[str, str]
    observations: ObservationConfig
    trajectory_fixes: TrajectoryFixConfig
    parking_depth: ParkingDepthConfig
    qc: dict[str, Any]
    merge: MergeConfig
    jump_qc: JumpQcConfig
    depth_bins: DepthBinConfig
    region_selection: RegionSelectionConfig
    resample: ResampleConfig
    output: OutputConfig
    min_segment_points: int
    diagnostics_dir: Path
    diagnostics_formats: tuple[str, ...]


@dataclass(frozen=True)
class QcSegmentResult:
    raw: pd.DataFrame
    kept: pd.DataFrame
    dropped: pd.DataFrame
    initial_segments: list[pd.DataFrame]
    merged_segments: list[pd.DataFrame]
    jump_dropped: pd.DataFrame
    jump_segments: list[pd.DataFrame]
    controlled_segments: list[pd.DataFrame]
    output_segments: list[pd.DataFrame]
    merge_events: list[dict[str, Any]]
    jump_events: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class RtrajFileData:
    trajectory_fixes: pd.DataFrame
    observations: pd.DataFrame
    observation_variable_attrs: dict[str, dict[str, Any]]
    observation_filter_counts: dict[str, int]


@dataclass(frozen=True)
class ObservationSamplingResult:
    segments: list[pd.DataFrame]
    summary: dict[str, Any]
    time_mismatch_days: dict[str, list[float]]
    pressure_mismatch_dbar: dict[str, list[float]]


@dataclass
class ObservationRunArtifacts:
    variable_attrs: dict[str, dict[str, Any]]
    time_mismatch_days: dict[str, list[float]]
    pressure_mismatch_dbar: dict[str, list[float]]


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the staged ARGO Rtraj-to-Zarr converter."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the staged Rtraj conversion YAML configuration file.",
    )
    return parser


def _resolve_input_files(config: dict[str, Any]) -> list[Path]:
    input_cfg = config.get("input", {}) or {}
    files: list[Path] = []

    for item in input_cfg.get("rtraj_files", []) or []:
        files.append(Path(item))

    rtraj_glob = input_cfg.get("rtraj_glob")
    if rtraj_glob:
        files.extend(Path(path) for path in glob.glob(str(rtraj_glob)))

    input_dir = input_cfg.get("rtraj_dir")
    pattern = input_cfg.get("pattern", "*_Rtraj.nc")
    if input_dir:
        files.extend(sorted(Path(input_dir).glob(str(pattern))))

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError("No input ARGO Rtraj NetCDF files were found from the configuration.")

    missing = [path for path in unique_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"The following ARGO Rtraj files were not found: {missing_str}")

    return unique_files


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be null or a positive finite number")
    return parsed


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be null or a finite number")
    return parsed


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be null or a non-negative integer")
    return parsed


def _resolve_merge_config(config: dict[str, Any]) -> MergeConfig:
    raw = (config.get("segmentation", {}) or {}).get("merge", {}) or {}
    return MergeConfig(
        enabled=bool(raw.get("enabled", False)),
        max_gap_points=_optional_nonnegative_int(raw.get("max_gap_points"), "segmentation.merge.max_gap_points"),
        max_gap_duration_days=_optional_positive_float(
            raw.get("max_gap_duration_days"),
            "segmentation.merge.max_gap_duration_days",
        ),
        max_bridge_speed_m_per_s=_optional_positive_float(
            raw.get("max_bridge_speed_m_per_s"),
            "segmentation.merge.max_bridge_speed_m_per_s",
        ),
        max_bridge_vertical_rate_m_per_day=_optional_positive_float(
            raw.get("max_bridge_vertical_rate_m_per_day"),
            "segmentation.merge.max_bridge_vertical_rate_m_per_day",
        ),
    )


def _resolve_jump_qc_config(config: dict[str, Any]) -> JumpQcConfig:
    raw = config.get("jump_qc", {}) or {}
    auto_drop = raw.get("auto_drop", {}) or {}
    return JumpQcConfig(
        enabled=bool(raw.get("enabled", False)),
        max_speed_m_per_s=_optional_positive_float(
            raw.get("max_speed_m_per_s", 2.0),
            "jump_qc.max_speed_m_per_s",
        )
        or 2.0,
        auto_drop_enabled=bool(auto_drop.get("enabled", True)),
        max_block_points=_optional_nonnegative_int(
            auto_drop.get("max_block_points", 3),
            "jump_qc.auto_drop.max_block_points",
        ),
        max_block_duration_days=_optional_positive_float(
            auto_drop.get("max_block_duration_days", 10.0),
            "jump_qc.auto_drop.max_block_duration_days",
        ),
        split_remaining_jumps=bool(raw.get("split_remaining_jumps", True)),
    )


def _resolve_trajectory_fix_config(config: dict[str, Any]) -> TrajectoryFixConfig:
    raw = config.get("trajectory_fixes", {}) or {}
    selection = raw.get("cycle_selection", {}) or {}
    tie_breaker = str(selection.get("tie_breaker", "first")).lower()
    if tie_breaker not in {"first", "last"}:
        raise ValueError("trajectory_fixes.cycle_selection.tie_breaker must be 'first' or 'last'")
    position_round_decimals = int(selection.get("position_round_decimals", 5))
    if position_round_decimals < 0:
        raise ValueError("trajectory_fixes.cycle_selection.position_round_decimals must be non-negative")
    return TrajectoryFixConfig(
        one_per_cycle=bool(raw.get("one_per_cycle", False)),
        require_finite_cycle=bool(selection.get("require_finite_cycle", True)),
        prefer_valid_position_qc=bool(selection.get("prefer_valid_position_qc", True)),
        prefer_valid_time_qc=bool(selection.get("prefer_valid_time_qc", True)),
        prefer_repeated_position=bool(selection.get("prefer_repeated_position", True)),
        position_round_decimals=position_round_decimals,
        tie_breaker=tie_breaker,
    )


def _resolve_parking_depth_config(config: dict[str, Any]) -> ParkingDepthConfig:
    raw = config.get("parking_depth", {}) or config.get("processing", {}).get("parking_depth", {}) or {}
    mode = str(raw.get("mode", "representative_park_pressure"))
    if mode != "representative_park_pressure":
        raise ValueError("parking_depth.mode must be 'representative_park_pressure'")
    fallback_raw = raw.get("fallback_value")
    fallback_value = None if fallback_raw is None else float(fallback_raw)
    if fallback_value is not None and not np.isfinite(fallback_value):
        raise ValueError("parking_depth.fallback_value must be null or finite")
    infer_raw = raw.get("infer_from_park_window", {}) or {}
    method = str(infer_raw.get("method", "percentile"))
    if method != "percentile":
        raise ValueError("parking_depth.infer_from_park_window.method must be 'percentile'")
    percentile = float(infer_raw.get("percentile", 95.0))
    if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
        raise ValueError("parking_depth.infer_from_park_window.percentile must be between 0 and 100")
    min_pressure_raw = infer_raw.get("min_pressure", 50.0)
    min_pressure = None if min_pressure_raw is None else float(min_pressure_raw)
    if min_pressure is not None and not np.isfinite(min_pressure):
        raise ValueError("parking_depth.infer_from_park_window.min_pressure must be null or finite")
    fallback_pressure = infer_raw.get("fallback_pressure_variable", "PRES")
    return ParkingDepthConfig(
        mode=mode,
        fallback_value=fallback_value,
        fill_missing=bool(raw.get("fill_missing", True)),
        infer_from_park_window=bool(infer_raw.get("enabled", True)),
        pressure_variable=str(infer_raw.get("pressure_variable", "PRES_ADJUSTED")),
        fallback_pressure_variable=None if fallback_pressure is None else str(fallback_pressure),
        percentile=percentile,
        min_pressure=min_pressure,
    )


def _resolve_observation_source_config(
    raw: Any,
    *,
    name: str,
    default_adjusted: str | None,
    default_fallback: str,
) -> ObservationSourceConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping with adjusted and fallback keys")

    adjusted_raw = raw.get("adjusted", default_adjusted)
    adjusted = None if adjusted_raw is None else str(adjusted_raw).strip()
    if adjusted == "":
        raise ValueError(f"{name}.adjusted must be null or a non-empty variable name")

    fallback = str(raw.get("fallback", default_fallback)).strip()
    if not fallback:
        raise ValueError(f"{name}.fallback must be a non-empty variable name")

    adjusted_qc_raw = raw.get("adjusted_qc")
    adjusted_qc = None if adjusted_qc_raw is None else str(adjusted_qc_raw).strip()
    if adjusted_qc == "":
        raise ValueError(f"{name}.adjusted_qc must be null or a non-empty variable name")

    fallback_qc_raw = raw.get("fallback_qc")
    fallback_qc = None if fallback_qc_raw is None else str(fallback_qc_raw).strip()
    if fallback_qc == "":
        raise ValueError(f"{name}.fallback_qc must be null or a non-empty variable name")

    valid_qc_raw = raw.get("valid_qc")
    valid_qc = (
        None
        if valid_qc_raw is None
        else _normalize_qc_values(valid_qc_raw, f"{name}.valid_qc")
    )
    missing_qc = str(raw.get("missing_qc", "accept")).strip().lower()
    if missing_qc not in {"accept", "reject"}:
        raise ValueError(f"{name}.missing_qc must be 'accept' or 'reject'")

    valid_min = _optional_finite_float(raw.get("valid_min"), f"{name}.valid_min")
    valid_max = _optional_finite_float(raw.get("valid_max"), f"{name}.valid_max")
    if valid_min is not None and valid_max is not None and valid_min > valid_max:
        raise ValueError(f"{name}.valid_min must be less than or equal to valid_max")

    return ObservationSourceConfig(
        adjusted=adjusted,
        fallback=fallback,
        adjusted_qc=adjusted_qc,
        fallback_qc=fallback_qc,
        valid_qc=valid_qc,
        missing_qc=missing_qc,
        valid_min=valid_min,
        valid_max=valid_max,
    )


def _resolve_observation_config(
    config: dict[str, Any],
    *,
    normalized_variables: dict[str, str],
    depth_bins: DepthBinConfig,
) -> ObservationConfig:
    raw = config.get("observations", {}) or {}
    if not isinstance(raw, dict):
        raise TypeError("observations must be a mapping")

    enabled = bool(raw.get("enabled", False))
    sample_at_fallback_depth = bool(raw.get("sample_at_fallback_depth", False))
    time = _resolve_observation_source_config(
        raw.get("time"),
        name="observations.time",
        default_adjusted="JULD_ADJUSTED",
        default_fallback="JULD",
    )
    pressure = _resolve_observation_source_config(
        raw.get("pressure"),
        name="observations.pressure",
        default_adjusted="PRES_ADJUSTED",
        default_fallback="PRES",
    )

    variables_raw = raw.get("variables", {}) or {}
    if not isinstance(variables_raw, dict):
        raise TypeError("observations.variables must be a mapping")

    reserved_names = {
        "trajectory",
        "obs",
        "z",
        "platform_code",
        "depth_bin",
        "depth_bin_interval",
        OBSERVATION_TIME_COLUMN,
        OBSERVATION_PRESSURE_COLUMN,
        OBSERVATION_INDEX_COLUMN,
        "_observation_depth_bin",
        *normalized_variables.values(),
        *INTERNAL_OUTPUT_COLUMNS,
    }
    variables: dict[str, ObservationSourceConfig] = {}
    for output_name_raw, source_raw in variables_raw.items():
        output_name = str(output_name_raw).strip()
        if not output_name:
            raise ValueError("observations.variables output names must be non-empty")
        if output_name in reserved_names:
            raise ValueError(
                f"observations.variables output name {output_name!r} conflicts with a trajectory field"
            )
        variables[output_name] = _resolve_observation_source_config(
            source_raw,
            name=f"observations.variables.{output_name}",
            default_adjusted=None,
            default_fallback=output_name,
        )

    if enabled and not variables:
        raise ValueError("observations.variables must contain at least one variable when observations are enabled")
    if enabled and not depth_bins.enabled:
        raise ValueError("depth_bins.enabled must be true when observations are enabled")

    return ObservationConfig(
        enabled=enabled,
        time=time,
        pressure=pressure,
        variables=variables,
        sample_at_fallback_depth=sample_at_fallback_depth,
    )


def _resolve_depth_bin_config(config: dict[str, Any]) -> DepthBinConfig:
    raw = config.get("depth_bins", {}) or {}
    enabled = bool(raw.get("enabled", False))
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
            output_mode=str(raw.get("output_mode", "per_bin")),
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
        output_mode=str(raw.get("output_mode", "per_bin")),
        bins=tuple(bins),
        missing_depth=missing_depth,
        isolated_outlier=isolated_outlier,
    )


def _resolve_region_selection_config(config: dict[str, Any]) -> RegionSelectionConfig:
    raw = config.get("regions", {}) or (config.get("processing", {}) or {}).get("regions", {}) or {}
    names_or_labels = tuple(str(item) for item in (raw.get("names_or_labels", []) or []))
    selection_mode = str(raw.get("selection_mode", "from_first_entry"))
    if selection_mode not in {"from_first_entry", "full_if_enters", "initial_inside"}:
        raise ValueError(
            "regions.selection_mode must be one of: from_first_entry, full_if_enters, initial_inside"
        )

    input_lon_mode = str(raw.get("input_lon_mode", "-180_180"))
    if input_lon_mode in {"-180180", "180180"}:
        input_lon_mode = "-180_180"

    return RegionSelectionConfig(
        names_or_labels=names_or_labels,
        selection_mode=selection_mode,
        input_lon_mode=input_lon_mode,
    )


def _resolve_resample_stage_config(config: dict[str, Any]) -> ResampleConfig:
    raw = config.get("resample", {}) or (config.get("processing", {}) or {}).get("resample", {}) or {}
    return resolve_resample_config({"processing": {"resample": raw}})


def _resolve_output_config(config: dict[str, Any]) -> OutputConfig:
    raw = config.get("output", {}) or {}
    zarr_path_raw = raw.get("zarr_path") or raw.get("path")
    if not zarr_path_raw:
        raise ValueError("output.zarr_path is required")

    return OutputConfig(
        zarr_path=Path(zarr_path_raw),
        write_zarr=bool(raw.get("write_zarr", False)),
        overwrite=bool(raw.get("overwrite", False)),
    )


def resolve_config(path: str | Path) -> RtrajConfig:
    config_path = Path(path)
    config = load_config(config_path)
    input_files = _resolve_input_files(config)

    run_cfg = config.get("run", {}) or {}
    max_files_raw = run_cfg.get("max_files")
    max_files = None if max_files_raw is None else int(max_files_raw)
    if max_files is not None and max_files <= 0:
        raise ValueError("run.max_files must be null or a positive integer")
    mode = str(run_cfg.get("mode", "convert")).lower()
    resolved_input_files = input_files[:max_files] if mode == "diagnostics" and max_files is not None else input_files

    source_variables = {
        "time": "JULD",
        "lon": "LONGITUDE",
        "lat": "LATITUDE",
        "depth": "PRES",
        "position_qc": "POSITION_QC",
        "time_qc": "JULD_QC",
        **(config.get("source_variables", {}) or {}),
    }
    normalized_variables = {
        "time": "time",
        "lon": "lon",
        "lat": "lat",
        "depth": "depth",
        **(config.get("normalized_variables", {}) or {}),
    }
    depth_bins = _resolve_depth_bin_config(config)
    observations = _resolve_observation_config(
        config,
        normalized_variables=normalized_variables,
        depth_bins=depth_bins,
    )

    segmentation_cfg = config.get("segmentation", {}) or {}
    min_segment_points = int(segmentation_cfg.get("min_segment_points", 2))
    if min_segment_points <= 0:
        raise ValueError("segmentation.min_segment_points must be a positive integer")

    diagnostics_cfg = config.get("diagnostics", {}) or {}
    diagnostics_dir_raw = diagnostics_cfg.get("output_dir")
    if not diagnostics_dir_raw:
        raise ValueError("diagnostics.output_dir is required")

    formats = tuple(str(item).lower().lstrip(".") for item in diagnostics_cfg.get("formats", ["png"]))
    if not formats:
        raise ValueError("diagnostics.formats must contain at least one file format")

    return RtrajConfig(
        path=config_path,
        raw=config,
        mode=mode,
        max_files=max_files,
        input_files=resolved_input_files,
        source_variables={key: str(value) for key, value in source_variables.items()},
        normalized_variables={key: str(value) for key, value in normalized_variables.items()},
        observations=observations,
        trajectory_fixes=_resolve_trajectory_fix_config(config),
        parking_depth=_resolve_parking_depth_config(config),
        qc=config.get("qc", {}) or {},
        merge=_resolve_merge_config(config),
        jump_qc=_resolve_jump_qc_config(config),
        depth_bins=depth_bins,
        region_selection=_resolve_region_selection_config(config),
        resample=_resolve_resample_stage_config(config),
        output=_resolve_output_config(config),
        min_segment_points=min_segment_points,
        diagnostics_dir=Path(diagnostics_dir_raw),
        diagnostics_formats=formats,
    )


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


def _numeric_variable(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Missing required Rtraj variable: {name}")
    return _decode_numeric_values(ds[name].values)


def _optional_numeric_variable(ds: xr.Dataset, name: str) -> np.ndarray | None:
    if name not in ds:
        return None
    return _decode_numeric_values(ds[name].values)


def _qc_variable(ds: xr.Dataset, name: str) -> np.ndarray | None:
    if name not in ds:
        return None

    arr = np.asarray(ds[name].values)
    if arr.ndim == 1 and arr.dtype.kind in {"S", "U"}:
        values = [_decode_byte(value).strip() for value in arr.tolist()]
    else:
        values = _decode_string_values(arr)
    return np.asarray([str(value).strip() for value in values], dtype=object)


def _datetime_variable(ds: xr.Dataset, name: str) -> pd.Series:
    if name not in ds:
        raise KeyError(f"Missing required Rtraj variable: {name}")

    variable = ds[name]
    values = np.asarray(variable.values)

    if np.issubdtype(values.dtype, np.datetime64):
        return pd.Series(pd.to_datetime(values.ravel(), errors="coerce")).dt.tz_localize(None).reset_index(drop=True)

    if np.issubdtype(values.dtype, np.number):
        numeric = pd.to_numeric(pd.Series(values.ravel()), errors="coerce").to_numpy(dtype=float)
        out = pd.Series(pd.NaT, index=pd.RangeIndex(len(numeric)), dtype="datetime64[ns]")
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
                return out.dt.tz_localize(None).reset_index(drop=True)
            except Exception:
                pass

        origin = pd.Timestamp("1950-01-01T00:00:00")
        out.loc[valid] = origin + pd.to_timedelta(numeric[valid], unit="D")
        return out.reset_index(drop=True)

    decoded = pd.to_datetime(pd.Series(values.ravel()), utc=True, errors="coerce")
    return decoded.dt.tz_convert(None).reset_index(drop=True)


def _optional_datetime_variable(ds: xr.Dataset, name: str) -> pd.Series | None:
    if name not in ds:
        return None
    return _datetime_variable(ds, name)


def _adjusted_time_with_raw_fallback(ds: xr.Dataset, raw_name: str) -> pd.Series:
    raw_time = _datetime_variable(ds, raw_name)
    if raw_name == "JULD_ADJUSTED":
        return raw_time.reset_index(drop=True)

    adjusted_time = _optional_datetime_variable(ds, "JULD_ADJUSTED")
    if adjusted_time is None or len(adjusted_time) != len(raw_time):
        return raw_time.reset_index(drop=True)

    out = adjusted_time.copy()
    missing = out.isna()
    if missing.any():
        out.loc[missing] = raw_time.loc[missing]
    return out.reset_index(drop=True)


def _observation_datetime_with_raw_fallback(
    ds: xr.Dataset,
    source: ObservationSourceConfig,
    *,
    length: int,
) -> pd.Series:
    out = pd.Series(pd.NaT, index=pd.RangeIndex(length), dtype="datetime64[ns]")
    if source.adjusted:
        adjusted = _optional_datetime_variable(ds, source.adjusted)
        if adjusted is not None and len(adjusted) == length:
            out.loc[:] = adjusted.reset_index(drop=True)

    fallback = _optional_datetime_variable(ds, source.fallback)
    if fallback is not None and len(fallback) == length:
        missing = out.isna()
        if missing.any():
            fallback = fallback.reset_index(drop=True)
            out.loc[missing] = fallback.loc[missing]
    return out.reset_index(drop=True)


def _observation_numeric_with_raw_fallback(
    ds: xr.Dataset,
    source: ObservationSourceConfig,
    *,
    length: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve one numeric observation source and filter the chosen values."""

    out = np.full(length, np.nan, dtype=float)
    counts: dict[str, int] = {}

    def add_count(reason: str, mask: np.ndarray) -> None:
        count = int(np.count_nonzero(mask))
        if count:
            counts[reason] = count

    adjusted = None
    if source.adjusted:
        adjusted = _optional_numeric_variable(ds, source.adjusted)
        if adjusted is None or len(adjusted) != length:
            adjusted = None

    fallback = _optional_numeric_variable(ds, source.fallback)
    if fallback is None or len(fallback) != length:
        fallback = None

    adjusted_selected = (
        np.isfinite(adjusted)
        if adjusted is not None
        else np.zeros(length, dtype=bool)
    )
    fallback_selected = (
        ~adjusted_selected & np.isfinite(fallback)
        if fallback is not None
        else np.zeros(length, dtype=bool)
    )

    def resolve_selected_source(
        *,
        values: np.ndarray | None,
        selected: np.ndarray,
        qc_name: str | None,
        label: str,
    ) -> None:
        if values is None or not selected.any():
            return

        accepted = selected.copy()
        if source.valid_qc is not None:
            qc = _qc_variable(ds, qc_name) if qc_name else None
            if qc is None or len(qc) != length:
                normalized_qc = np.full(length, "", dtype=object)
            else:
                normalized_qc = np.asarray(
                    [str(value).strip() for value in qc],
                    dtype=object,
                )

            qc_lower = np.asarray(
                [str(value).strip().lower() for value in normalized_qc],
                dtype=object,
            )
            qc_missing = np.isin(qc_lower, ["", "nan", "none", "<na>"])
            missing_rejected = (
                accepted & qc_missing
                if source.missing_qc == "reject"
                else np.zeros(length, dtype=bool)
            )
            add_count(f"{label}_missing_qc_rejected", missing_rejected)
            accepted &= ~missing_rejected

            present = accepted & ~qc_missing
            valid_qc = np.isin(normalized_qc, source.valid_qc)
            invalid_qc = present & ~valid_qc
            for flag in sorted(set(normalized_qc[invalid_qc].tolist())):
                add_count(
                    f"{label}_qc_{flag}_rejected",
                    invalid_qc & (normalized_qc == flag),
                )
            accepted &= ~(present & ~valid_qc)

        if source.valid_min is not None:
            below_minimum = accepted & (values < source.valid_min)
            add_count(f"{label}_below_valid_min_rejected", below_minimum)
            accepted &= ~below_minimum
        if source.valid_max is not None:
            above_maximum = accepted & (values > source.valid_max)
            add_count(f"{label}_above_valid_max_rejected", above_maximum)
            accepted &= ~above_maximum

        out[accepted] = values[accepted]
        add_count(f"{label}_accepted", accepted)

    resolve_selected_source(
        values=adjusted,
        selected=adjusted_selected,
        qc_name=source.adjusted_qc,
        label="adjusted",
    )
    resolve_selected_source(
        values=fallback,
        selected=fallback_selected,
        qc_name=source.fallback_qc,
        label="raw_fallback",
    )

    add_count("unavailable", ~(adjusted_selected | fallback_selected))
    return out, counts


def _observation_source_metadata(
    ds: xr.Dataset,
    source: ObservationSourceConfig,
    *,
    length: int,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    source_names = [name for name in (source.adjusted, source.fallback) if name]
    for source_name in source_names:
        if source_name not in ds or int(ds[source_name].size) != length:
            continue
        for key in OBSERVATION_METADATA_KEYS:
            value = ds[source_name].attrs.get(key)
            if value is not None and str(value).strip() and key not in attrs:
                attrs[key] = value
    return attrs


def _build_observation_table(
    ds: xr.Dataset,
    config: ObservationConfig,
    *,
    length: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, int]]:
    if not config.enabled:
        return pd.DataFrame(), {}, {}

    pressure, pressure_counts = _observation_numeric_with_raw_fallback(
        ds,
        config.pressure,
        length=length,
    )

    data: dict[str, Any] = {
        OBSERVATION_INDEX_COLUMN: np.arange(length, dtype=np.int64),
        OBSERVATION_TIME_COLUMN: _observation_datetime_with_raw_fallback(
            ds,
            config.time,
            length=length,
        ),
        OBSERVATION_PRESSURE_COLUMN: pressure,
    }
    filter_counts = {
        f"pressure.{reason}": count
        for reason, count in pressure_counts.items()
    }
    variable_attrs: dict[str, dict[str, Any]] = {}
    for output_name, source in config.variables.items():
        values, counts = _observation_numeric_with_raw_fallback(
            ds,
            source,
            length=length,
        )
        data[output_name] = values
        filter_counts.update(
            {
                f"{output_name}.{reason}": count
                for reason, count in counts.items()
            }
        )
        variable_attrs[output_name] = _observation_source_metadata(ds, source, length=length)

    return pd.DataFrame(data), variable_attrs, filter_counts


def _time_window_values(ds: xr.Dataset, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if name not in ds:
        return None

    values = np.asarray(ds[name].values).ravel()
    if np.issubdtype(values.dtype, np.datetime64):
        valid = ~np.isnat(values)
        return values, valid

    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric, np.isfinite(numeric)


def _validate_equal_lengths(file_path: Path, **arrays: Any) -> int:
    lengths = {name: len(value) for name, value in arrays.items() if value is not None}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Inconsistent measurement variable lengths in {file_path}: {details}")
    return unique_lengths.pop()


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
    depth = np.full(len(cycle_numbers), np.nan, dtype=float)
    if not lookup:
        return depth

    for idx, cycle_value in enumerate(cycle_numbers):
        key = _cycle_key(float(cycle_value))
        if key is None:
            continue
        mapped = lookup.get(key)
        if mapped is not None:
            depth[idx] = float(mapped)
    return depth


def _map_representative_pressure_to_observations(
    ds: xr.Dataset,
    pressure_name: str,
    *,
    fallback_value: float | None = None,
) -> np.ndarray:
    pressure = _optional_numeric_variable(ds, pressure_name)
    raw_cycles = _numeric_variable(ds, "CYCLE_NUMBER")
    adjusted_cycles = _optional_numeric_variable(ds, "CYCLE_NUMBER_ADJUSTED")
    raw_cycle_index = _optional_numeric_variable(ds, "CYCLE_NUMBER_INDEX")
    adjusted_cycle_index = _optional_numeric_variable(ds, "CYCLE_NUMBER_INDEX_ADJUSTED")

    if adjusted_cycles is None:
        adjusted_cycles = raw_cycles
    elif len(adjusted_cycles) != len(raw_cycles):
        raise ValueError("CYCLE_NUMBER_ADJUSTED and CYCLE_NUMBER have different lengths")

    adjusted_lookup = _build_pressure_lookup(adjusted_cycle_index, pressure)
    raw_lookup = _build_pressure_lookup(raw_cycle_index, pressure)

    depth = _map_pressure_from_lookup(adjusted_cycles, adjusted_lookup)
    missing_after_adjusted = ~np.isfinite(depth)
    if missing_after_adjusted.any():
        raw_depth = _map_pressure_from_lookup(raw_cycles, raw_lookup)
        depth[missing_after_adjusted] = raw_depth[missing_after_adjusted]

    if fallback_value is not None:
        depth[~np.isfinite(depth)] = float(fallback_value)

    return depth


def _cycle_lookup_from_representative_pressure(ds: xr.Dataset, pressure_name: str) -> dict[int | float, float]:
    pressure = _optional_numeric_variable(ds, pressure_name)
    adjusted_cycle_index = _optional_numeric_variable(ds, "CYCLE_NUMBER_INDEX_ADJUSTED")
    raw_cycle_index = _optional_numeric_variable(ds, "CYCLE_NUMBER_INDEX")
    lookup = _build_pressure_lookup(adjusted_cycle_index, pressure)
    lookup.update({key: value for key, value in _build_pressure_lookup(raw_cycle_index, pressure).items() if key not in lookup})
    return lookup


def _cycle_lookup_from_park_window(
    ds: xr.Dataset,
    *,
    pressure_name: str,
    percentile: float,
    min_pressure: float | None,
) -> dict[int | float, float]:
    pressure = _optional_numeric_variable(ds, pressure_name)
    if pressure is None:
        return {}

    raw_cycles = _optional_numeric_variable(ds, "CYCLE_NUMBER")
    cycle_index = _optional_numeric_variable(ds, "CYCLE_NUMBER_INDEX")
    juld_data = _time_window_values(ds, "JULD")
    park_start_data = _time_window_values(ds, "JULD_PARK_START")
    park_end_data = _time_window_values(ds, "JULD_PARK_END")
    if raw_cycles is None or cycle_index is None or juld_data is None or park_start_data is None or park_end_data is None:
        return {}
    juld, juld_valid = juld_data
    park_start, park_start_valid = park_start_data
    park_end, park_end_valid = park_end_data
    if len(raw_cycles) != len(pressure) or len(raw_cycles) != len(juld):
        return {}

    raw_cycle_keys = np.asarray([_cycle_key(float(value)) for value in raw_cycles], dtype=object)
    lookup: dict[int | float, float] = {}
    count = min(len(cycle_index), len(park_start), len(park_end), len(park_start_valid), len(park_end_valid))
    for idx, cycle_value in enumerate(cycle_index[:count]):
        cycle = _cycle_key(float(cycle_value))
        if cycle is None or not park_start_valid[idx] or not park_end_valid[idx]:
            continue
        start = park_start[idx]
        end = park_end[idx]
        if end < start:
            continue
        time_mask = juld_valid & (juld >= start) & (juld <= end)
        cycle_mask = raw_cycle_keys == cycle
        values = pressure[time_mask & cycle_mask]
        values = values[np.isfinite(values)]
        if min_pressure is not None:
            values = values[values >= min_pressure]
        if values.size == 0:
            continue
        lookup[cycle] = float(np.nanpercentile(values, percentile))
    return lookup


def _fill_cycle_depth_lookup(
    cycle_keys: list[int | float | None],
    lookup: dict[int | float, float],
    source_lookup: dict[int | float, str],
    *,
    fallback_value: float | None,
    fill_missing: bool,
) -> tuple[dict[int | float, float], dict[int | float, str]]:
    ordered_cycles: list[int | float] = []
    seen: set[int | float] = set()
    for key in cycle_keys:
        if key is None or key in seen:
            continue
        ordered_cycles.append(key)
        seen.add(key)

    filled = dict(lookup)
    sources = dict(source_lookup)
    if fill_missing and ordered_cycles:
        last_value: float | None = None
        for key in ordered_cycles:
            if key in filled:
                last_value = filled[key]
            elif last_value is not None:
                filled[key] = last_value
                sources[key] = "depth_ffill"

        next_value: float | None = None
        for key in reversed(ordered_cycles):
            if key in lookup:
                next_value = lookup[key]
            elif key not in filled and next_value is not None:
                filled[key] = next_value
                sources[key] = "depth_bfill"

    if fallback_value is not None:
        for key in ordered_cycles:
            if key not in filled:
                filled[key] = float(fallback_value)
                sources[key] = "fallback"

    return filled, sources


def _resolve_parking_depth_to_observations(
    ds: xr.Dataset,
    config: ParkingDepthConfig,
    *,
    pressure_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_cycles = _numeric_variable(ds, "CYCLE_NUMBER")
    cycle_keys = [_cycle_key(float(value)) for value in raw_cycles]
    lookup: dict[int | float, float] = {}
    source_lookup: dict[int | float, str] = {}

    representative = _cycle_lookup_from_representative_pressure(ds, pressure_name)
    for key, value in representative.items():
        lookup[key] = value
        source_lookup[key] = "representative_park_pressure"

    if config.infer_from_park_window:
        inferred_adjusted = _cycle_lookup_from_park_window(
            ds,
            pressure_name=config.pressure_variable,
            percentile=config.percentile,
            min_pressure=config.min_pressure,
        )
        for key, value in inferred_adjusted.items():
            if key not in lookup:
                lookup[key] = value
                source_lookup[key] = f"park_window_{config.pressure_variable.lower()}_p{config.percentile:g}"

        if config.fallback_pressure_variable:
            inferred_raw = _cycle_lookup_from_park_window(
                ds,
                pressure_name=config.fallback_pressure_variable,
                percentile=config.percentile,
                min_pressure=config.min_pressure,
            )
            for key, value in inferred_raw.items():
                if key not in lookup:
                    lookup[key] = value
                    source_lookup[key] = f"park_window_{config.fallback_pressure_variable.lower()}_p{config.percentile:g}"

    lookup, source_lookup = _fill_cycle_depth_lookup(
        cycle_keys,
        lookup,
        source_lookup,
        fallback_value=config.fallback_value,
        fill_missing=config.fill_missing,
    )

    depth = np.full(len(raw_cycles), np.nan, dtype=float)
    depth_source = np.full(len(raw_cycles), "missing", dtype=object)
    for idx, key in enumerate(cycle_keys):
        if key is None:
            continue
        value = lookup.get(key)
        if value is None:
            continue
        depth[idx] = float(value)
        depth_source[idx] = source_lookup.get(key, "unknown")
    return depth, depth_source


def _depth_variable(ds: xr.Dataset, name: str, *, config: ParkingDepthConfig) -> tuple[np.ndarray, np.ndarray]:
    if name == "REPRESENTATIVE_PARK_PRESSURE":
        return _resolve_parking_depth_to_observations(ds, config, pressure_name=name)

    depth = _optional_numeric_variable(ds, name)
    raw_cycles = _optional_numeric_variable(ds, "CYCLE_NUMBER")
    if depth is not None and raw_cycles is not None and len(depth) == len(raw_cycles):
        source = np.full(len(depth), name.lower(), dtype=object)
        return depth, source

    return _resolve_parking_depth_to_observations(ds, config, pressure_name="REPRESENTATIVE_PARK_PRESSURE")


def _filter_trajectory_fixes(frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    time_name = names["time"]
    lon_name = names["lon"]
    lat_name = names["lat"]
    lon = pd.to_numeric(frame[lon_name], errors="coerce")
    lat = pd.to_numeric(frame[lat_name], errors="coerce")
    fix_mask = frame[time_name].notna() & lon.notna() & lat.notna()

    out = frame.loc[fix_mask].copy()
    sort_cols = [time_name, "source_index"]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    out.attrs["raw_measurement_rows"] = int(len(frame))
    out.attrs["finite_trajectory_fix_rows"] = int(len(out))
    out.attrs["trajectory_fix_rows"] = int(len(out))
    out.attrs["non_fix_rows"] = int(len(frame) - len(out))
    out.attrs["cycle_representative_dropped_points"] = 0
    out.attrs["missing_cycle_dropped_points"] = 0
    return out


def _configured_valid_qc_values(qc_config: dict[str, Any], rule_name: str) -> set[str]:
    rule = (qc_config.get("variables", {}) or {}).get(rule_name, {}) or {}
    raw_values = rule.get("valid_values", ["1", "2"])
    if isinstance(raw_values, (str, int)):
        values = [raw_values]
    else:
        values = list(raw_values or [])
    return {str(value).strip() for value in values if str(value).strip()}


def _select_cycle_representative_fixes(frame: pd.DataFrame, config: RtrajConfig) -> pd.DataFrame:
    if not config.trajectory_fixes.one_per_cycle or frame.empty or "cycle_number" not in frame.columns:
        out = frame.copy()
        out.attrs.update(frame.attrs)
        out.attrs["cycle_representative_dropped_points"] = 0
        out.attrs["missing_cycle_dropped_points"] = 0
        return out

    names = config.normalized_variables
    fix_config = config.trajectory_fixes
    work = frame.copy()
    work.attrs.update(frame.attrs)
    finite_fix_rows = int(frame.attrs.get("finite_trajectory_fix_rows", len(frame)))

    cycle_values = pd.to_numeric(work["cycle_number"], errors="coerce")
    missing_cycle = ~np.isfinite(cycle_values.to_numpy(dtype=float))
    missing_cycle_count = int(missing_cycle.sum())
    if fix_config.require_finite_cycle:
        work = work.loc[~missing_cycle].copy()
        cycle_values = cycle_values.loc[work.index]
    if work.empty:
        out = work.reset_index(drop=True)
        out.attrs.update(frame.attrs)
        out.attrs["finite_trajectory_fix_rows"] = finite_fix_rows
        out.attrs["trajectory_fix_rows"] = 0
        out.attrs["missing_cycle_dropped_points"] = missing_cycle_count
        out.attrs["cycle_representative_dropped_points"] = max(finite_fix_rows - missing_cycle_count, 0)
        return out

    rounded_cycle = np.rint(cycle_values.to_numpy(dtype=float))
    cycle_key = np.where(
        np.isclose(cycle_values.to_numpy(dtype=float), rounded_cycle),
        rounded_cycle.astype(np.int64),
        cycle_values.to_numpy(dtype=float),
    )
    work["_cycle_key"] = cycle_key
    lon = pd.to_numeric(work[names["lon"]], errors="coerce").round(fix_config.position_round_decimals)
    lat = pd.to_numeric(work[names["lat"]], errors="coerce").round(fix_config.position_round_decimals)
    work["_position_key"] = list(zip(lon.tolist(), lat.tolist()))

    position_qc_valid = _configured_valid_qc_values(config.qc, "position_qc")
    time_qc_valid = _configured_valid_qc_values(config.qc, "time_qc")
    work["_position_qc_score"] = (
        work["position_qc"].astype("string").str.strip().isin(position_qc_valid).astype(int)
        if fix_config.prefer_valid_position_qc
        else 0
    )
    work["_time_qc_score"] = (
        work["time_qc"].astype("string").str.strip().isin(time_qc_valid).astype(int)
        if fix_config.prefer_valid_time_qc
        else 0
    )
    if fix_config.prefer_repeated_position:
        work["_position_count"] = work.groupby(["platform_code", "_cycle_key", "_position_key"], dropna=False)[
            "source_index"
        ].transform("size")
    else:
        work["_position_count"] = 0
    work["_nonzero_position_score"] = (
        (pd.to_numeric(work[names["lon"]], errors="coerce").abs() > 1.0e-12)
        | (pd.to_numeric(work[names["lat"]], errors="coerce").abs() > 1.0e-12)
    ).astype(int)

    source_ascending = fix_config.tie_breaker == "first"
    selected = (
        work.sort_values(
            [
                "platform_code",
                "_cycle_key",
                "_position_qc_score",
                "_time_qc_score",
                "_nonzero_position_score",
                "_position_count",
                "source_index",
            ],
            ascending=[True, True, False, False, False, False, source_ascending],
            kind="stable",
        )
        .drop_duplicates(["platform_code", "_cycle_key"], keep="first")
        .drop(
            columns=[
                "_cycle_key",
                "_position_key",
                "_position_qc_score",
                "_time_qc_score",
                "_position_count",
                "_nonzero_position_score",
            ]
        )
        .sort_values([names["time"], "source_index"], kind="stable")
        .reset_index(drop=True)
    )
    selected.attrs.update(frame.attrs)
    selected.attrs["finite_trajectory_fix_rows"] = finite_fix_rows
    selected.attrs["trajectory_fix_rows"] = int(len(selected))
    selected.attrs["missing_cycle_dropped_points"] = missing_cycle_count if fix_config.require_finite_cycle else 0
    selected.attrs["cycle_representative_dropped_points"] = int(finite_fix_rows - missing_cycle_count - len(selected))
    return selected


def _read_rtraj_file_data(file_path: Path, config: RtrajConfig) -> RtrajFileData:
    names = config.source_variables
    out_names = config.normalized_variables

    with _suppress_xarray_time_serialization_warning():
        ds_context = xr.open_dataset(
            file_path,
            decode_times=True,
            decode_timedelta=False,
            mask_and_scale=True,
        )

    with ds_context as ds:
        platform_code = _decode_platform_number(ds, file_path=file_path)
        time = _adjusted_time_with_raw_fallback(ds, names["time"])
        lon = _numeric_variable(ds, names["lon"])
        lat = _numeric_variable(ds, names["lat"])
        depth, depth_source = _depth_variable(ds, names["depth"], config=config.parking_depth)
        position_qc = _qc_variable(ds, names["position_qc"])
        time_qc = _qc_variable(ds, names["time_qc"])
        cycle_number = _optional_numeric_variable(ds, "CYCLE_NUMBER")
        measurement_code = _optional_numeric_variable(ds, "MEASUREMENT_CODE")
        n_rows = _validate_equal_lengths(
            file_path,
            time=time,
            lon=lon,
            lat=lat,
            depth=depth,
            depth_source=depth_source,
            position_qc=position_qc,
            time_qc=time_qc,
        )
        (
            observations,
            observation_variable_attrs,
            observation_filter_counts,
        ) = _build_observation_table(
            ds,
            config.observations,
            length=n_rows,
        )

    if cycle_number is not None and len(cycle_number) != n_rows:
        cycle_number = None
    if measurement_code is not None and len(measurement_code) != n_rows:
        measurement_code = None

    frame = pd.DataFrame(
        {
            "source_file": str(file_path),
            "platform_code": np.full(n_rows, platform_code, dtype=np.int64),
            "source_index": np.arange(n_rows, dtype=np.int64),
            out_names["time"]: time,
            out_names["lon"]: lon,
            out_names["lat"]: lat,
            out_names["depth"]: depth,
            "depth_source": depth_source,
            "position_qc": position_qc,
            "time_qc": time_qc,
        }
    )
    if cycle_number is not None:
        frame["cycle_number"] = cycle_number
    if measurement_code is not None:
        frame["measurement_code"] = measurement_code

    fixes = _filter_trajectory_fixes(frame, out_names)
    return RtrajFileData(
        trajectory_fixes=_select_cycle_representative_fixes(fixes, config),
        observations=observations,
        observation_variable_attrs=observation_variable_attrs,
        observation_filter_counts=observation_filter_counts,
    )


def read_and_normalize_rtraj_file(file_path: Path, config: RtrajConfig) -> pd.DataFrame:
    return _read_rtraj_file_data(file_path, config).trajectory_fixes


def _normalize_qc_values(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        raise ValueError(f"{name} must not be null")
    if isinstance(values, (str, int)):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raise ValueError(f"{name} must be a string, integer, or list of strings/integers")

    normalized = tuple(str(value).strip() for value in raw_values)
    if not normalized or any(value == "" for value in normalized):
        raise ValueError(f"{name} must contain at least one non-empty QC flag")
    return normalized


def apply_qc_mask(raw: pd.DataFrame, config: RtrajConfig) -> pd.DataFrame:
    if not bool(config.qc.get("enabled", False)):
        out = raw.copy()
        out["_qc_order"] = np.arange(len(out), dtype=np.int64)
        out["qc_keep"] = True
        out["qc_drop_reasons"] = ""
        return out

    missing_qc = str(config.qc.get("missing_qc", "fail")).lower()
    if missing_qc not in {"fail", "pass"}:
        raise ValueError("qc.missing_qc must be 'fail' or 'pass'")

    out = raw.copy()
    out["_qc_order"] = np.arange(len(out), dtype=np.int64)
    keep = np.ones(len(out), dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(len(out))]

    for rule_name, rule in (config.qc.get("variables", {}) or {}).items():
        source = str(rule.get("source", rule_name))
        if source == config.source_variables.get("position_qc"):
            qc_column = "position_qc"
        elif source == config.source_variables.get("time_qc"):
            qc_column = "time_qc"
        else:
            qc_column = source

        if qc_column not in out.columns:
            raise KeyError(f"QC rule {rule_name!r} refers to missing column {qc_column!r}")

        accepted = set(_normalize_qc_values(rule.get("valid_values", ["1", "2"]), f"qc.variables.{rule_name}.valid_values"))
        values = out[qc_column].astype("string").str.strip()
        missing = values.isna() | (values == "")
        bad = ~values.isin(accepted)
        if missing_qc == "pass":
            bad = bad & ~missing

        bad_array = bad.to_numpy(dtype=bool)
        keep &= ~bad_array
        for idx in np.flatnonzero(bad_array):
            reasons[int(idx)].append(str(rule_name))

    out["qc_keep"] = keep
    out["qc_drop_reasons"] = [";".join(item) for item in reasons]
    return out


def _split_kept_points(qc_frame: pd.DataFrame, *, min_segment_points: int) -> list[pd.DataFrame]:
    segments: list[pd.DataFrame] = []
    current: list[int] = []

    for idx, keep in enumerate(qc_frame["qc_keep"].to_numpy(dtype=bool)):
        if keep:
            current.append(idx)
            continue
        if len(current) >= min_segment_points:
            segments.append(qc_frame.iloc[current].copy())
        current = []

    if len(current) >= min_segment_points:
        segments.append(qc_frame.iloc[current].copy())

    return [segment.reset_index(drop=True) for segment in segments]


def _haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float | None:
    values = (lon1, lat1, lon2, lat2)
    if not all(np.isfinite(float(value)) for value in values):
        return None

    dlon_deg = ((float(lon2) - float(lon1) + 180.0) % 360.0) - 180.0
    dlat_deg = float(lat2) - float(lat1)
    lon_delta = np.deg2rad(dlon_deg)
    lat_delta = np.deg2rad(dlat_deg)
    lat1_rad = np.deg2rad(float(lat1))
    lat2_rad = np.deg2rad(float(lat2))

    a = (
        np.sin(lat_delta / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(lon_delta / 2.0) ** 2
    )
    return float(2.0 * 6371.0088 * np.arcsin(np.sqrt(a)))


def _gap_metrics(
    previous: pd.Series,
    current: pd.Series,
    *,
    gap_points: int,
    config: RtrajConfig,
) -> dict[str, float | int | None]:
    names = config.normalized_variables
    previous_time = previous[names["time"]]
    current_time = current[names["time"]]
    if pd.isna(previous_time) or pd.isna(current_time):
        gap_days = None
    else:
        gap_days_raw = (pd.Timestamp(current_time) - pd.Timestamp(previous_time)).total_seconds() / SECONDS_PER_DAY
        gap_days = float(gap_days_raw) if np.isfinite(gap_days_raw) else None
    gap_hours = None if gap_days is None else float(gap_days * 24.0)

    distance_km = _haversine_distance_km(
        previous[names["lon"]],
        previous[names["lat"]],
        current[names["lon"]],
        current[names["lat"]],
    )
    speed_m_s = None
    if distance_km is not None and gap_days is not None and gap_days > 0:
        speed_m_s = float(distance_km * 1000.0 / (gap_days * SECONDS_PER_DAY))

    vertical_rate = None
    if gap_days is not None and gap_days > 0:
        previous_depth = pd.to_numeric(pd.Series([previous[names["depth"]]]), errors="coerce").iloc[0]
        current_depth = pd.to_numeric(pd.Series([current[names["depth"]]]), errors="coerce").iloc[0]
        if pd.notna(previous_depth) and pd.notna(current_depth):
            vertical_rate = float(abs(float(current_depth) - float(previous_depth)) / gap_days)

    return {
        "gap_points": int(gap_points),
        "gap_days": gap_days,
        "gap_hours": gap_hours,
        "distance_km": distance_km,
        "bridge_speed_m_per_s": speed_m_s,
        "bridge_vertical_rate_m_per_day": vertical_rate,
    }


def _merge_decision(metrics: dict[str, float | int | None], merge: MergeConfig) -> tuple[bool, str]:
    if not merge.enabled:
        return False, "merge_disabled"

    gap_points = metrics["gap_points"]
    gap_days = metrics["gap_days"]
    speed = metrics["bridge_speed_m_per_s"]
    vertical_rate = metrics["bridge_vertical_rate_m_per_day"]

    if merge.max_gap_points is not None and int(gap_points or 0) > merge.max_gap_points:
        return False, "gap_points_too_large"
    if merge.max_gap_duration_days is not None:
        if gap_days is None or float(gap_days) < 0 or float(gap_days) > merge.max_gap_duration_days:
            return False, "gap_duration_too_large"
    if merge.max_bridge_speed_m_per_s is not None:
        if speed is None or float(speed) > merge.max_bridge_speed_m_per_s:
            return False, "bridge_speed_too_large"
    if merge.max_bridge_vertical_rate_m_per_day is not None:
        if vertical_rate is None or float(vertical_rate) > merge.max_bridge_vertical_rate_m_per_day:
            return False, "bridge_vertical_rate_too_large"

    return True, "merged"


def merge_qc_segments(
    segments: list[pd.DataFrame],
    qc_frame: pd.DataFrame,
    config: RtrajConfig,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    if not segments:
        return [], []

    merged: list[pd.DataFrame] = [segments[0].copy()]
    events: list[dict[str, Any]] = []
    names = config.normalized_variables

    for next_segment in segments[1:]:
        previous_segment = merged[-1]
        previous_row = previous_segment.iloc[-1]
        current_row = next_segment.iloc[0]
        previous_order = int(previous_segment["_qc_order"].iloc[-1])
        current_order = int(next_segment["_qc_order"].iloc[0])
        previous_source_index = int(previous_segment["source_index"].iloc[-1])
        current_source_index = int(next_segment["source_index"].iloc[0])
        gap_points = max(current_order - previous_order - 1, 0)

        metrics = _gap_metrics(
            previous_row,
            current_row,
            gap_points=gap_points,
            config=config,
        )
        should_merge, reason = _merge_decision(metrics, config.merge)
        events.append(
            {
                "boundary_id": len(events) + 1,
                "previous_source_index": previous_source_index,
                "current_source_index": current_source_index,
                "previous_qc_order": previous_order,
                "current_qc_order": current_order,
                "previous_time": previous_row[names["time"]],
                "current_time": current_row[names["time"]],
                "previous_lon": previous_row[names["lon"]],
                "previous_lat": previous_row[names["lat"]],
                "current_lon": current_row[names["lon"]],
                "current_lat": current_row[names["lat"]],
                "merged": should_merge,
                "reason": reason,
                **metrics,
            }
        )

        if should_merge:
            merged[-1] = (
                pd.concat([previous_segment, next_segment], ignore_index=True)
                .sort_values("source_index", kind="stable")
                .reset_index(drop=True)
            )
        else:
            merged.append(next_segment.reset_index(drop=True))

    return merged, events


def _link_exceeds_speed_limit(metrics: dict[str, float | int | None], limit_m_per_s: float) -> bool:
    speed = metrics["bridge_speed_m_per_s"]
    if speed is not None:
        return float(speed) > limit_m_per_s

    gap_days = metrics["gap_days"]
    distance_km = metrics["distance_km"]
    if gap_days is not None and float(gap_days) <= 0:
        return distance_km is None or float(distance_km) > 0.0
    return False


def _bridge_is_plausible(metrics: dict[str, float | int | None], limit_m_per_s: float) -> bool:
    speed = metrics["bridge_speed_m_per_s"]
    if speed is not None:
        return float(speed) <= limit_m_per_s

    gap_days = metrics["gap_days"]
    distance_km = metrics["distance_km"]
    if gap_days is not None and float(gap_days) >= 0 and distance_km is not None:
        return float(distance_km) == 0.0
    return False


def _compute_jump_links(segment: pd.DataFrame, config: RtrajConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = segment.sort_values(config.normalized_variables["time"], kind="stable").reset_index(drop=True)
    for idx in range(len(ordered) - 1):
        previous = ordered.iloc[idx]
        current = ordered.iloc[idx + 1]
        metrics = _gap_metrics(previous, current, gap_points=0, config=config)
        rows.append(
            {
                "local_previous_idx": idx,
                "local_current_idx": idx + 1,
                "previous_source_index": int(previous["source_index"]),
                "current_source_index": int(current["source_index"]),
                "previous_qc_order": int(previous["_qc_order"]) if "_qc_order" in previous else None,
                "current_qc_order": int(current["_qc_order"]) if "_qc_order" in current else None,
                "too_fast": _link_exceeds_speed_limit(metrics, config.jump_qc.max_speed_m_per_s),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _block_duration_days(block: pd.DataFrame, time_name: str) -> float | None:
    if block.empty:
        return 0.0
    start = block[time_name].iloc[0]
    end = block[time_name].iloc[-1]
    if pd.isna(start) or pd.isna(end):
        return None
    duration = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / SECONDS_PER_DAY
    return float(duration) if np.isfinite(duration) else None


def _find_auto_drop_candidate(
    segment: pd.DataFrame,
    links: pd.DataFrame,
    config: RtrajConfig,
) -> dict[str, Any] | None:
    if not config.jump_qc.auto_drop_enabled or segment.empty or links.empty:
        return None

    time_name = config.normalized_variables["time"]
    max_points = config.jump_qc.max_block_points
    max_duration = config.jump_qc.max_block_duration_days
    n_rows = len(segment)
    too_fast = links["too_fast"].to_numpy(dtype=bool)

    for start in range(1, n_rows - 1):
        if not too_fast[start - 1]:
            continue

        max_end = n_rows - 2
        if max_points is not None:
            max_end = min(max_end, start + max_points - 1)

        for end in range(start, max_end + 1):
            if not too_fast[end]:
                continue

            block = segment.iloc[start : end + 1]
            duration_days = _block_duration_days(block, time_name)
            if max_duration is not None:
                if duration_days is None or duration_days < 0 or duration_days > max_duration:
                    continue

            bridge_metrics = _gap_metrics(
                segment.iloc[start - 1],
                segment.iloc[end + 1],
                gap_points=end - start + 1,
                config=config,
            )
            if not _bridge_is_plausible(bridge_metrics, config.jump_qc.max_speed_m_per_s):
                continue

            return {
                "start": start,
                "end": end,
                "duration_days": duration_days,
                "bridge_metrics": bridge_metrics,
                "reason": "isolated_spike" if start == end else "short_bad_location_block",
            }

    return None


def _clean_jump_segment(
    segment: pd.DataFrame,
    config: RtrajConfig,
    *,
    segment_id: int,
    first_block_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], int]:
    ordered = segment.sort_values(config.normalized_variables["time"], kind="stable").reset_index(drop=True)
    dropped_frames: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    block_id = first_block_id

    while len(ordered) >= 3:
        links = _compute_jump_links(ordered, config)
        candidate = _find_auto_drop_candidate(ordered, links, config)
        if candidate is None:
            break

        start = int(candidate["start"])
        end = int(candidate["end"])
        block = ordered.iloc[start : end + 1].copy()
        block["jump_qc_drop_reason"] = candidate["reason"]
        block["jump_qc_block_id"] = block_id
        dropped_frames.append(block)

        events.append(
            {
                "event_type": "auto_drop",
                "segment_id": segment_id,
                "block_id": block_id,
                "reason": candidate["reason"],
                "dropped_points": int(len(block)),
                "dropped_source_indices": ";".join(str(int(value)) for value in block["source_index"].tolist()),
                "previous_source_index": int(ordered.iloc[start - 1]["source_index"]),
                "current_source_index": int(ordered.iloc[end + 1]["source_index"]),
                "duration_days": candidate["duration_days"],
                **candidate["bridge_metrics"],
            }
        )

        ordered = ordered.drop(ordered.index[start : end + 1]).reset_index(drop=True)
        block_id += 1

    dropped = pd.concat(dropped_frames, ignore_index=True) if dropped_frames else pd.DataFrame(columns=segment.columns)
    return ordered.reset_index(drop=True), dropped, events, block_id


def _split_segment_on_remaining_jumps(
    segment: pd.DataFrame,
    config: RtrajConfig,
    *,
    segment_id: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    if segment.empty:
        return [], []

    ordered = segment.sort_values(config.normalized_variables["time"], kind="stable").reset_index(drop=True)
    links = _compute_jump_links(ordered, config)
    if links.empty or not config.jump_qc.split_remaining_jumps:
        return [ordered] if len(ordered) >= config.min_segment_points else [], []

    split_link_indices = [
        int(row.local_previous_idx)
        for row in links.itertuples(index=False)
        if bool(row.too_fast)
    ]
    events: list[dict[str, Any]] = []
    output: list[pd.DataFrame] = []
    start = 0

    for local_previous_idx in split_link_indices:
        link = links.loc[links["local_previous_idx"] == local_previous_idx].iloc[0]
        events.append(
            {
                "event_type": "remaining_jump_split",
                "segment_id": segment_id,
                "reason": "remaining_jump_split",
                "previous_source_index": int(link["previous_source_index"]),
                "current_source_index": int(link["current_source_index"]),
                "previous_qc_order": None if pd.isna(link["previous_qc_order"]) else int(link["previous_qc_order"]),
                "current_qc_order": None if pd.isna(link["current_qc_order"]) else int(link["current_qc_order"]),
                "dropped_points": 0,
                "dropped_source_indices": "",
                "duration_days": None,
                "gap_points": int(link["gap_points"]),
                "gap_days": link["gap_days"],
                "gap_hours": link["gap_hours"],
                "distance_km": link["distance_km"],
                "bridge_speed_m_per_s": link["bridge_speed_m_per_s"],
                "bridge_vertical_rate_m_per_day": link["bridge_vertical_rate_m_per_day"],
            }
        )
        part = ordered.iloc[start : local_previous_idx + 1].copy().reset_index(drop=True)
        if len(part) >= config.min_segment_points:
            output.append(part)
        start = local_previous_idx + 1

    part = ordered.iloc[start:].copy().reset_index(drop=True)
    if len(part) >= config.min_segment_points:
        output.append(part)
    return output, events


def _merge_jump_qc_segments(
    segments: list[pd.DataFrame],
    config: RtrajConfig,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    if not segments:
        return [], []

    merge = MergeConfig(
        enabled=config.merge.enabled,
        max_gap_points=None,
        max_gap_duration_days=config.merge.max_gap_duration_days,
        max_bridge_speed_m_per_s=config.merge.max_bridge_speed_m_per_s,
        max_bridge_vertical_rate_m_per_day=config.merge.max_bridge_vertical_rate_m_per_day,
    )
    if not merge.enabled:
        return [segment.reset_index(drop=True) for segment in segments], []

    merged: list[pd.DataFrame] = [segments[0].copy().reset_index(drop=True)]
    events: list[dict[str, Any]] = []
    names = config.normalized_variables

    for next_segment in segments[1:]:
        previous_segment = merged[-1]
        previous_row = previous_segment.iloc[-1]
        current_row = next_segment.iloc[0]
        previous_order = int(previous_segment["_qc_order"].iloc[-1])
        current_order = int(next_segment["_qc_order"].iloc[0])
        previous_source_index = int(previous_segment["source_index"].iloc[-1])
        current_source_index = int(next_segment["source_index"].iloc[0])
        gap_points = max(current_order - previous_order - 1, 0)
        metrics = _gap_metrics(
            previous_row,
            current_row,
            gap_points=gap_points,
            config=config,
        )
        should_merge, reason = _merge_decision(metrics, merge)
        events.append(
            {
                "event_type": "post_jump_merge",
                "reason": reason,
                "previous_source_index": previous_source_index,
                "current_source_index": current_source_index,
                "previous_qc_order": previous_order,
                "current_qc_order": current_order,
                "previous_time": previous_row[names["time"]],
                "current_time": current_row[names["time"]],
                "previous_lon": previous_row[names["lon"]],
                "previous_lat": previous_row[names["lat"]],
                "current_lon": current_row[names["lon"]],
                "current_lat": current_row[names["lat"]],
                "merged": should_merge,
                "dropped_points": 0,
                "dropped_source_indices": "",
                "duration_days": None,
                **metrics,
            }
        )

        if should_merge:
            merged[-1] = (
                pd.concat([previous_segment, next_segment], ignore_index=True)
                .sort_values(names["time"], kind="stable")
                .reset_index(drop=True)
            )
        else:
            merged.append(next_segment.reset_index(drop=True))

    return merged, events


def apply_jump_qc_segments(
    segments: list[pd.DataFrame],
    config: RtrajConfig,
) -> tuple[list[pd.DataFrame], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if not config.jump_qc.enabled:
        return segments, pd.DataFrame(), [], {
            "jump_qc_enabled": False,
            "jump_qc_dropped_points": 0,
            "jump_qc_dropped_blocks": 0,
            "jump_qc_remaining_jumps": 0,
            "jump_qc_split_count": 0,
            "jump_qc_post_merge_count": 0,
            "jump_qc_post_merge_rejection_count": 0,
            "jump_qc_reason_counts": {},
            "segments_after_jump_qc_pre_merge": int(len(segments)),
            "segments_after_jump_qc": int(len(segments)),
        }

    cleaned_segments: list[pd.DataFrame] = []
    dropped_frames: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    block_id = 1

    for segment_id, segment in enumerate(segments, start=1):
        cleaned, dropped, drop_events, block_id = _clean_jump_segment(
            segment,
            config,
            segment_id=segment_id,
            first_block_id=block_id,
        )
        if not dropped.empty:
            dropped_frames.append(dropped)
        events.extend(drop_events)

        split_segments, split_events = _split_segment_on_remaining_jumps(
            cleaned,
            config,
            segment_id=segment_id,
        )
        cleaned_segments.extend(split_segments)
        events.extend(split_events)

    pre_merge_segment_count = len(cleaned_segments)
    cleaned_segments, post_merge_events = _merge_jump_qc_segments(cleaned_segments, config)
    events.extend(post_merge_events)

    dropped = pd.concat(dropped_frames, ignore_index=True) if dropped_frames else pd.DataFrame()
    reason_counts: dict[str, int] = {}
    for event in events:
        reason = str(event.get("reason", "unknown"))
        if event.get("event_type") == "auto_drop":
            increment = int(event.get("dropped_points", 0))
        elif event.get("event_type") == "post_jump_merge" and event.get("merged"):
            increment = 0
        else:
            increment = 1
        if increment:
            reason_counts[reason] = reason_counts.get(reason, 0) + increment

    remaining_jumps = int(sum(1 for event in events if event.get("event_type") == "remaining_jump_split"))
    post_merge_count = int(sum(1 for event in post_merge_events if event.get("merged")))
    post_merge_rejection_count = int(sum(1 for event in post_merge_events if not event.get("merged")))
    summary = {
        "jump_qc_enabled": True,
        "jump_qc_dropped_points": int(len(dropped)),
        "jump_qc_dropped_blocks": int(sum(1 for event in events if event.get("event_type") == "auto_drop")),
        "jump_qc_remaining_jumps": remaining_jumps,
        "jump_qc_split_count": remaining_jumps if config.jump_qc.split_remaining_jumps else 0,
        "jump_qc_post_merge_count": post_merge_count,
        "jump_qc_post_merge_rejection_count": post_merge_rejection_count,
        "jump_qc_reason_counts": dict(sorted(reason_counts.items())),
        "segments_after_jump_qc_pre_merge": int(pre_merge_segment_count),
        "segments_after_jump_qc": int(len(cleaned_segments)),
    }
    return cleaned_segments, dropped, events, summary


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


def _assign_depth_bins(segment: pd.DataFrame, config: RtrajConfig) -> tuple[pd.DataFrame, int, int]:
    out = segment.copy().reset_index(drop=True)
    out["depth_bin"] = None
    out["depth_bin_interval"] = None
    out["depth_bin_filled"] = False
    out["depth_bin_repaired"] = False
    if not config.depth_bins.enabled:
        return out, 0, 0

    depth_name = config.normalized_variables["depth"]
    for idx, depth_value in enumerate(out[depth_name]):
        depth_bin = _find_depth_bin(depth_value, config.depth_bins)
        if depth_bin is None:
            continue
        out.at[idx, "depth_bin"] = depth_bin.label
        out.at[idx, "depth_bin_interval"] = depth_bin.interval_label

    out, fill_count = _fill_missing_depth_bins(out, config.depth_bins)
    out, repair_count = _repair_isolated_depth_bin_runs(out, config.depth_bins)
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
    segments: list[pd.DataFrame],
    config: RtrajConfig,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    if not config.depth_bins.enabled:
        return segments, {
            "depth_bin_enabled": False,
            "depth_bin_segments": int(len(segments)),
            "depth_bin_unassigned_points": 0,
            "depth_finite_points": 0,
            "depth_missing_points": 0,
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
    finite_depth_points = 0
    missing_depth_points = 0
    bin_counts: dict[str, int] = {}
    depth_name = config.normalized_variables["depth"]

    for segment in segments:
        depth_values = pd.to_numeric(segment[depth_name], errors="coerce")
        finite_depth_points += int(depth_values.notna().sum())
        missing_depth_points += int(depth_values.isna().sum())
        binned, segment_fill_count, segment_repair_count = _assign_depth_bins(segment, config)
        fill_count += int(segment_fill_count)
        repair_count += int(segment_repair_count)
        unassigned_points += int(binned["depth_bin"].isna().sum())
        labels = [str(value) for value in binned["depth_bin"].dropna().tolist()]
        transition_count += int(sum(1 for previous, current in zip(labels[:-1], labels[1:]) if previous != current))
        for label in labels:
            bin_counts[label] = bin_counts.get(label, 0) + 1
        output.extend(_split_segment_by_depth_bin(binned, min_segment_points=config.min_segment_points))

    return output, {
        "depth_bin_enabled": True,
        "depth_bin_segments": int(len(output)),
        "depth_bin_unassigned_points": int(unassigned_points),
        "depth_finite_points": int(finite_depth_points),
        "depth_missing_points": int(missing_depth_points),
        "depth_bin_fill_count": int(fill_count),
        "depth_bin_repair_count": int(repair_count),
        "depth_bin_transition_count": int(transition_count),
        "depth_bin_counts": dict(sorted(bin_counts.items())),
    }


def apply_region_selection_segments(
    segments: list[pd.DataFrame],
    config: RtrajConfig,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    input_segments = [segment.reset_index(drop=True) for segment in segments if not segment.empty]
    input_points = int(sum(len(segment) for segment in input_segments))
    region_config = config.region_selection
    if not region_config.names_or_labels:
        return input_segments, {
            "region_selection_enabled": False,
            "region_names_or_labels": "",
            "region_selection_mode": region_config.selection_mode,
            "region_input_segments": int(len(input_segments)),
            "region_output_segments": int(len(input_segments)),
            "region_dropped_segments": 0,
            "region_trimmed_segments": 0,
            "region_min_length_dropped_segments": 0,
            "region_input_points": input_points,
            "region_output_points": input_points,
        }

    output: list[pd.DataFrame] = []
    dropped_segments = 0
    trimmed_segments = 0
    min_length_dropped = 0
    for segment in input_segments:
        selected = apply_region_selection([segment], config=region_config)
        if not selected:
            dropped_segments += 1
            continue

        selected_segment = selected[0].reset_index(drop=True)
        if len(selected_segment) < len(segment):
            trimmed_segments += 1
        if len(selected_segment) < config.min_segment_points:
            dropped_segments += 1
            min_length_dropped += 1
            continue
        output.append(selected_segment)

    return output, {
        "region_selection_enabled": True,
        "region_names_or_labels": ";".join(region_config.names_or_labels),
        "region_selection_mode": region_config.selection_mode,
        "region_input_segments": int(len(input_segments)),
        "region_output_segments": int(len(output)),
        "region_dropped_segments": int(dropped_segments),
        "region_trimmed_segments": int(trimmed_segments),
        "region_min_length_dropped_segments": int(min_length_dropped),
        "region_input_points": input_points,
        "region_output_points": int(sum(len(segment) for segment in output)),
    }


def apply_resampling_segments(
    segments: list[pd.DataFrame],
    config: RtrajConfig,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    input_segments = [segment.reset_index(drop=True) for segment in segments if not segment.empty]
    input_points = int(sum(len(segment) for segment in input_segments))
    resample_config = config.resample
    enabled = bool(
        resample_config.frequency
        or resample_config.min_duration_days is not None
        or resample_config.shared_time
        or resample_config.shift_start_to_reference
    )

    if not enabled:
        return input_segments, {
            "resample_enabled": False,
            "resample_frequency": "",
            "resample_input_segments": int(len(input_segments)),
            "resample_duration_dropped_segments": 0,
            "resample_empty_dropped_segments": 0,
            "resample_output_segments": int(len(input_segments)),
            "resample_input_points": input_points,
            "resample_output_points": input_points,
        }

    duration_filtered, duration_dropped = filter_trajectories_by_min_duration(
        input_segments,
        resample_config.min_duration_days,
    )
    resampled = apply_resampling(
        duration_filtered,
        config=resample_config,
        non_interpolated_columns=NON_INTERPOLATED_COLUMNS,
        show_progress=False,
    )
    output = [segment.reset_index(drop=True) for segment in resampled if not segment.empty]
    empty_dropped = len(resampled) - len(output)

    return output, {
        "resample_enabled": True,
        "resample_frequency": "" if resample_config.frequency is None else str(resample_config.frequency),
        "resample_input_segments": int(len(input_segments)),
        "resample_duration_dropped_segments": int(duration_dropped),
        "resample_empty_dropped_segments": int(empty_dropped),
        "resample_output_segments": int(len(output)),
        "resample_input_points": input_points,
        "resample_output_points": int(sum(len(segment) for segment in output)),
    }


def sample_observations_onto_segments(
    segments: list[pd.DataFrame],
    observations: pd.DataFrame,
    config: RtrajConfig,
) -> ObservationSamplingResult:
    output = [segment.copy().reset_index(drop=True) for segment in segments]
    variable_names = tuple(config.observations.variables)
    empty_mismatches = {name: [] for name in variable_names}
    if not config.observations.enabled:
        return ObservationSamplingResult(
            segments=output,
            summary={
                "observation_sampling_enabled": False,
                "observation_eligible_points": 0,
                "observation_unknown_depth_skipped_points": 0,
                "observation_matched_counts": {},
                "observation_unmatched_counts": {},
                "observation_median_abs_time_mismatch_days": {},
                "observation_median_abs_pressure_mismatch_dbar": {},
            },
            time_mismatch_days=empty_mismatches,
            pressure_mismatch_dbar={name: [] for name in variable_names},
        )

    for segment in output:
        for name in variable_names:
            segment[name] = np.nan

    observation_work = observations.copy().reset_index(drop=True)
    if OBSERVATION_TIME_COLUMN not in observation_work:
        observation_work[OBSERVATION_TIME_COLUMN] = pd.NaT
    if OBSERVATION_PRESSURE_COLUMN not in observation_work:
        observation_work[OBSERVATION_PRESSURE_COLUMN] = np.nan
    if OBSERVATION_INDEX_COLUMN not in observation_work:
        observation_work[OBSERVATION_INDEX_COLUMN] = np.arange(len(observation_work), dtype=np.int64)
    observation_work[OBSERVATION_TIME_COLUMN] = pd.to_datetime(
        observation_work[OBSERVATION_TIME_COLUMN],
        errors="coerce",
    )
    observation_work[OBSERVATION_PRESSURE_COLUMN] = pd.to_numeric(
        observation_work[OBSERVATION_PRESSURE_COLUMN],
        errors="coerce",
    )
    observation_work["_observation_depth_bin"] = [
        None if (depth_bin := _find_depth_bin(value, config.depth_bins)) is None else depth_bin.label
        for value in observation_work[OBSERVATION_PRESSURE_COLUMN]
    ]

    candidate_groups: dict[str, dict[str, pd.DataFrame]] = {}
    for name in variable_names:
        if name not in observation_work:
            observation_work[name] = np.nan
        observation_work[name] = pd.to_numeric(observation_work[name], errors="coerce")
        valid = (
            observation_work[OBSERVATION_TIME_COLUMN].notna()
            & np.isfinite(observation_work[OBSERVATION_PRESSURE_COLUMN].to_numpy(dtype=float))
            & np.isfinite(observation_work[name].to_numpy(dtype=float))
            & observation_work["_observation_depth_bin"].notna()
        )
        candidates = observation_work.loc[valid].copy()
        candidate_groups[name] = {
            str(label): group.reset_index(drop=True)
            for label, group in candidates.groupby("_observation_depth_bin", sort=False)
        }

    eligible_points = 0
    unknown_depth_points = 0
    matched_counts = {name: 0 for name in variable_names}
    time_mismatches = {name: [] for name in variable_names}
    pressure_mismatches = {name: [] for name in variable_names}
    time_name = config.normalized_variables["time"]
    depth_name = config.normalized_variables["depth"]

    for segment in output:
        for point_index, point in segment.iterrows():
            if (
                str(point.get("depth_source", "")) == "fallback"
                and not config.observations.sample_at_fallback_depth
            ):
                unknown_depth_points += 1
                continue

            eligible_points += 1
            point_time = pd.to_datetime(point.get(time_name), errors="coerce")
            point_pressure = pd.to_numeric(pd.Series([point.get(depth_name)]), errors="coerce").iloc[0]
            depth_bin_raw = point.get("depth_bin")
            depth_bin_label = None if pd.isna(depth_bin_raw) else str(depth_bin_raw)
            if pd.isna(point_time) or pd.isna(point_pressure) or depth_bin_label is None:
                continue

            point_time_ns = np.datetime64(point_time, "ns").astype(np.int64)
            point_pressure_float = float(point_pressure)
            for name in variable_names:
                candidates = candidate_groups[name].get(depth_bin_label)
                if candidates is None or candidates.empty:
                    continue

                candidate_time_ns = candidates[OBSERVATION_TIME_COLUMN].to_numpy(dtype="datetime64[ns]").astype(
                    np.int64
                )
                time_difference_ns = np.abs(candidate_time_ns - point_time_ns)
                pressure_difference = np.abs(
                    candidates[OBSERVATION_PRESSURE_COLUMN].to_numpy(dtype=float) - point_pressure_float
                )
                measurement_index = pd.to_numeric(
                    candidates[OBSERVATION_INDEX_COLUMN],
                    errors="coerce",
                ).to_numpy(dtype=np.int64)
                selected_position = int(
                    np.lexsort((measurement_index, pressure_difference, time_difference_ns))[0]
                )
                selected = candidates.iloc[selected_position]
                segment.at[point_index, name] = float(selected[name])
                matched_counts[name] += 1
                time_mismatches[name].append(
                    float(time_difference_ns[selected_position]) / (SECONDS_PER_DAY * 1.0e9)
                )
                pressure_mismatches[name].append(float(pressure_difference[selected_position]))

    unmatched_counts = {name: eligible_points - count for name, count in matched_counts.items()}
    median_time = {
        name: (float(np.median(values)) if values else np.nan)
        for name, values in time_mismatches.items()
    }
    median_pressure = {
        name: (float(np.median(values)) if values else np.nan)
        for name, values in pressure_mismatches.items()
    }
    return ObservationSamplingResult(
        segments=output,
        summary={
            "observation_sampling_enabled": True,
            "observation_eligible_points": int(eligible_points),
            "observation_unknown_depth_skipped_points": int(unknown_depth_points),
            "observation_matched_counts": matched_counts,
            "observation_unmatched_counts": unmatched_counts,
            "observation_median_abs_time_mismatch_days": median_time,
            "observation_median_abs_pressure_mismatch_dbar": median_pressure,
        },
        time_mismatch_days=time_mismatches,
        pressure_mismatch_dbar=pressure_mismatches,
    )


def process_qc_stage(raw: pd.DataFrame, config: RtrajConfig) -> QcSegmentResult:
    qc_frame = apply_qc_mask(raw, config)
    kept = qc_frame.loc[qc_frame["qc_keep"]].copy().reset_index(drop=True)
    dropped = qc_frame.loc[~qc_frame["qc_keep"]].copy().reset_index(drop=True)
    initial_segments = _split_kept_points(qc_frame, min_segment_points=config.min_segment_points)
    merged_segments, merge_events = merge_qc_segments(initial_segments, qc_frame, config)
    jump_segments, jump_dropped, jump_events, jump_summary = apply_jump_qc_segments(merged_segments, config)
    depth_segments, depth_summary = apply_depth_bin_segmentation(jump_segments, config)
    controlled_segments, region_summary = apply_region_selection_segments(depth_segments, config)
    output_segments, resample_summary = apply_resampling_segments(controlled_segments, config)

    drop_reason_counts: dict[str, int] = {}
    for reason_text in dropped.get("qc_drop_reasons", pd.Series(dtype=str)).dropna():
        for reason in str(reason_text).split(";"):
            if reason:
                drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

    merge_rejection_counts: dict[str, int] = {}
    for event in merge_events:
        if event["merged"]:
            continue
        reason = str(event["reason"])
        merge_rejection_counts[reason] = merge_rejection_counts.get(reason, 0) + 1
    depth_source_counts = {
        str(source): int(count)
        for source, count in qc_frame.get("depth_source", pd.Series(dtype=object)).value_counts(dropna=False).items()
    }
    raw_measurement_rows = int(raw.attrs.get("raw_measurement_rows", len(raw)))
    finite_trajectory_fix_rows = int(raw.attrs.get("finite_trajectory_fix_rows", len(raw)))
    trajectory_fix_rows = int(raw.attrs.get("trajectory_fix_rows", len(raw)))
    non_fix_rows = int(raw.attrs.get("non_fix_rows", raw_measurement_rows - trajectory_fix_rows))
    cycle_representative_dropped = int(raw.attrs.get("cycle_representative_dropped_points", 0))
    missing_cycle_dropped = int(raw.attrs.get("missing_cycle_dropped_points", 0))

    summary = {
        "raw_measurement_rows": raw_measurement_rows,
        "finite_trajectory_fix_rows": finite_trajectory_fix_rows,
        "trajectory_fix_rows": trajectory_fix_rows,
        "non_fix_rows": non_fix_rows,
        "cycle_representative_dropped_points": cycle_representative_dropped,
        "missing_cycle_dropped_points": missing_cycle_dropped,
        "raw_points": int(len(raw)),
        "kept_points": int(len(kept)),
        "dropped_points": int(len(dropped)),
        "initial_segments": int(len(initial_segments)),
        "merged_segments": int(len(merged_segments)),
        "merge_count": int(sum(1 for event in merge_events if event["merged"])),
        "merge_rejection_count": int(sum(1 for event in merge_events if not event["merged"])),
        "drop_reason_counts": drop_reason_counts,
        "merge_rejection_counts": merge_rejection_counts,
        "depth_source_counts": depth_source_counts,
        **jump_summary,
        **depth_summary,
        **region_summary,
        **resample_summary,
        "controlled_segments": int(len(controlled_segments)),
        "output_segments": int(len(output_segments)),
    }
    return QcSegmentResult(
        raw=qc_frame,
        kept=kept,
        dropped=dropped,
        initial_segments=initial_segments,
        merged_segments=merged_segments,
        jump_dropped=jump_dropped,
        jump_segments=jump_segments,
        controlled_segments=controlled_segments,
        output_segments=output_segments,
        merge_events=merge_events,
        jump_events=jump_events,
        summary=summary,
    )


def _split_longitude_wrapped_path(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    max_lon_step: float = 180.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    if lon_arr.size < 2:
        return [(lon_arr, lat_arr)]

    valid = np.isfinite(lon_arr[:-1]) & np.isfinite(lon_arr[1:])
    jump_idx = np.flatnonzero(valid & (np.abs(np.diff(lon_arr)) > max_lon_step))
    if jump_idx.size == 0:
        return [(lon_arr, lat_arr)]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for idx in jump_idx:
        stop = idx + 1
        segments.append((lon_arr[start:stop], lat_arr[start:stop]))
        start = stop
    segments.append((lon_arr[start:], lat_arr[start:]))
    return segments


def _combined_extent(frames: list[pd.DataFrame], *, names: dict[str, str]) -> list[float]:
    lon_values: list[np.ndarray] = []
    lat_values: list[np.ndarray] = []
    lon_name = names["lon"]
    lat_name = names["lat"]
    for frame in frames:
        if frame.empty or lon_name not in frame.columns or lat_name not in frame.columns:
            continue
        lon_values.append(pd.to_numeric(frame[lon_name], errors="coerce").to_numpy(dtype=float))
        lat_values.append(pd.to_numeric(frame[lat_name], errors="coerce").to_numpy(dtype=float))

    if not lon_values or not lat_values:
        return [-180.0, 180.0, -80.0, -30.0]

    lon = np.concatenate(lon_values)
    lat = np.concatenate(lat_values)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        return [-180.0, 180.0, -80.0, -30.0]

    lon = lon[valid]
    lat = lat[valid]
    lon_min = float(lon.min())
    lon_max = float(lon.max())
    lat_min = float(lat.min())
    lat_max = float(lat.max())
    lon_span = max(lon_max - lon_min, 1.0)
    lat_span = max(lat_max - lat_min, 1.0)
    lon_pad = max(2.0, min(20.0, 0.15 * lon_span))
    lat_pad = max(2.0, min(10.0, 0.20 * lat_span))
    return [
        max(-180.0, lon_min - lon_pad),
        min(180.0, lon_max + lon_pad),
        max(-90.0, lat_min - lat_pad),
        min(90.0, lat_max + lat_pad),
    ]


def _setup_map_axis(ax: Any, *, title: str, extent: list[float]) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    land = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        "10m",
        edgecolor="black",
        facecolor=cfeature.COLORS["land"],
        linewidth=0.3,
    )
    ax.add_feature(land, zorder=0)
    ax.coastlines(resolution="10m", linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4, alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title)


def _setup_plain_axis(ax: Any, *, title: str, extent: list[float]) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_title(title)


def _plot_raw_vs_qc_segments(
    result: QcSegmentResult,
    *,
    file_path: Path,
    output_path: Path,
    config: RtrajConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs

        use_cartopy = True
    except ModuleNotFoundError:
        ccrs = None
        use_cartopy = False

    names = config.normalized_variables
    raw = result.raw
    dropped = result.dropped
    jump_dropped = result.jump_dropped

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [raw, *result.controlled_segments, *result.output_segments]
    extent = _combined_extent(frames, names=names)
    if use_cartopy:
        fig = plt.figure(figsize=(22, 7))
        ax_raw = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax_controlled = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        ax_resampled = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
        _setup_map_axis(ax_raw, title="Raw trajectory", extent=extent)
        _setup_map_axis(ax_controlled, title="Controlled trajectory", extent=extent)
        _setup_map_axis(ax_resampled, title="Resampled output", extent=extent)
        transform_kwargs = {"transform": ccrs.PlateCarree()}
    else:
        fig, (ax_raw, ax_controlled, ax_resampled) = plt.subplots(1, 3, figsize=(22, 7))
        _setup_plain_axis(ax_raw, title="Raw trajectory", extent=extent)
        _setup_plain_axis(ax_controlled, title="Controlled trajectory", extent=extent)
        _setup_plain_axis(ax_resampled, title="Resampled output", extent=extent)
        transform_kwargs = {}

    raw_ordered = raw.sort_values(names["time"], kind="stable").reset_index(drop=True)
    for lon_part, lat_part in _split_longitude_wrapped_path(raw_ordered[names["lon"]], raw_ordered[names["lat"]]):
        if len(lon_part) >= 2:
            ax_raw.plot(
                lon_part,
                lat_part,
                color="0.25",
                linewidth=0.8,
                alpha=0.7,
                zorder=2,
                **transform_kwargs,
            )
    ax_raw.scatter(
        raw_ordered[names["lon"]],
        raw_ordered[names["lat"]],
        s=8,
        color="tab:blue",
        alpha=0.45,
        zorder=3,
        **transform_kwargs,
    )
    if not dropped.empty:
        ax_raw.scatter(
            dropped[names["lon"]],
            dropped[names["lat"]],
            s=18,
            color="tab:red",
            alpha=0.9,
            zorder=4,
            label="QC dropped",
            **transform_kwargs,
        )
        ax_raw.legend(loc="best", fontsize=8)
    if not jump_dropped.empty:
        ax_raw.scatter(
            jump_dropped[names["lon"]],
            jump_dropped[names["lat"]],
            s=30,
            marker="D",
            color="tab:orange",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
            zorder=5,
            label="Jump dropped",
            **transform_kwargs,
        )
        ax_raw.legend(loc="best", fontsize=8)

    from matplotlib.lines import Line2D

    segment_cmap = plt.get_cmap("tab20")
    depth_bin_labels = [depth_bin.label for depth_bin in config.depth_bins.bins]
    depth_bin_markers = ["o", "s", "^", "D", "P", "v", "<", ">"]
    depth_bin_styles = {
        label: depth_bin_markers[idx % len(depth_bin_markers)]
        for idx, label in enumerate(depth_bin_labels)
    }
    used_depth_labels: set[str] = set()
    for idx, segment in enumerate(result.controlled_segments):
        ordered = segment.sort_values(names["time"], kind="stable").reset_index(drop=True)
        depth_label = None
        if "depth_bin" in ordered.columns and not ordered["depth_bin"].dropna().empty:
            depth_label = str(ordered["depth_bin"].dropna().iloc[0])
        color = segment_cmap(idx % segment_cmap.N)
        marker = depth_bin_styles.get(depth_label, "o")
        for lon_part, lat_part in _split_longitude_wrapped_path(ordered[names["lon"]], ordered[names["lat"]]):
            if len(lon_part) >= 2:
                ax_controlled.plot(
                    lon_part,
                    lat_part,
                    color=color,
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=2,
                    **transform_kwargs,
                )
        ax_controlled.scatter(
            ordered[names["lon"]],
            ordered[names["lat"]],
            s=10,
            color=color,
            marker=marker,
            alpha=0.65,
            zorder=3,
            **transform_kwargs,
        )
        if depth_label:
            used_depth_labels.add(depth_label)

    not_merged_events = [event for event in result.merge_events if not event["merged"]]
    jump_split_events = [event for event in result.jump_events if event.get("event_type") == "remaining_jump_split"]
    if not_merged_events or jump_split_events:
        marker_cmap = plt.get_cmap("tab10")
        reasons = sorted(
            {str(event["reason"]) for event in not_merged_events}
            | {str(event["reason"]) for event in jump_split_events}
        )
        reason_colors = {
            reason: marker_cmap(idx % marker_cmap.N)
            for idx, reason in enumerate(reasons)
        }
        for reason in reasons:
            endpoint_orders = [
                int(event["previous_qc_order"])
                for event in not_merged_events
                if str(event["reason"]) == reason and event.get("previous_qc_order") is not None
            ] + [
                int(event["previous_qc_order"])
                for event in jump_split_events
                if str(event["reason"]) == reason and event.get("previous_qc_order") is not None
            ]
            endpoints = raw.loc[raw["_qc_order"].isin(endpoint_orders)]
            if endpoints.empty:
                continue
            ax_controlled.scatter(
                endpoints[names["lon"]],
                endpoints[names["lat"]],
                s=58,
                marker="X",
                color=reason_colors[reason],
                edgecolors="black",
                linewidths=0.45,
                alpha=0.95,
                zorder=5,
                label=reason,
                **transform_kwargs,
            )
        for event in [*not_merged_events, *jump_split_events]:
            if event.get("previous_qc_order") is None:
                continue
            endpoint = raw.loc[raw["_qc_order"] == int(event["previous_qc_order"])]
            if endpoint.empty:
                continue
            row = endpoint.iloc[0]
            ax_controlled.text(
                row[names["lon"]],
                row[names["lat"]],
                str(event.get("boundary_id", "J")),
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=6,
                bbox={
                    "boxstyle": "circle,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
                **transform_kwargs,
            )
        first_legend = ax_controlled.legend(title="Segment boundaries", loc="upper right", fontsize=8, title_fontsize=8)
        ax_controlled.add_artist(first_legend)

    if used_depth_labels:
        depth_handles = [
            Line2D(
                [0],
                [0],
                marker=depth_bin_styles[label],
                color="0.2",
                linestyle="None",
                markersize=6,
                label=label,
            )
            for label in depth_bin_labels
            if label in used_depth_labels
        ]
        ax_controlled.legend(
            handles=depth_handles,
            title="Depth bin",
            loc="lower right",
            fontsize=8,
            title_fontsize=8,
        )

    used_output_depth_labels: set[str] = set()
    for idx, segment in enumerate(result.output_segments):
        ordered = segment.sort_values(names["time"], kind="stable").reset_index(drop=True)
        depth_label = None
        if "depth_bin" in ordered.columns and not ordered["depth_bin"].dropna().empty:
            depth_label = str(ordered["depth_bin"].dropna().iloc[0])
        color = segment_cmap(idx % segment_cmap.N)
        marker = depth_bin_styles.get(depth_label, "o")
        for lon_part, lat_part in _split_longitude_wrapped_path(ordered[names["lon"]], ordered[names["lat"]]):
            if len(lon_part) >= 2:
                ax_resampled.plot(
                    lon_part,
                    lat_part,
                    color=color,
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=2,
                    **transform_kwargs,
                )
        ax_resampled.scatter(
            ordered[names["lon"]],
            ordered[names["lat"]],
            s=9,
            color=color,
            marker=marker,
            alpha=0.70,
            zorder=3,
            **transform_kwargs,
        )
        if depth_label:
            used_output_depth_labels.add(depth_label)

    if used_output_depth_labels:
        output_depth_handles = [
            Line2D(
                [0],
                [0],
                marker=depth_bin_styles[label],
                color="0.2",
                linestyle="None",
                markersize=6,
                label=label,
            )
            for label in depth_bin_labels
            if label in used_output_depth_labels
        ]
        ax_resampled.legend(
            handles=output_depth_handles,
            title="Depth bin",
            loc="lower right",
            fontsize=8,
            title_fontsize=8,
        )

    platform = raw["platform_code"].iloc[0] if not raw.empty else "unknown"
    fig.suptitle(
        f"RTRAJ diagnostic: platform {platform} "
        f"({len(raw_ordered)} raw fixes, {result.summary['dropped_points']} QC dropped, "
        f"{result.summary['jump_qc_dropped_points']} jump dropped, "
        f"{len(result.controlled_segments)} controlled segment(s), "
        f"{len(result.output_segments)} resampled output segment(s))"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_diagnostics(result: QcSegmentResult, *, file_path: Path, file_index: int, config: RtrajConfig) -> list[Path]:
    written: list[Path] = []
    platform = result.raw["platform_code"].iloc[0] if not result.raw.empty else file_path.stem
    safe_platform = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(platform))
    stem = f"{file_index:03d}_{safe_platform}_{file_path.stem}_raw_controlled_resampled"

    for fmt in config.diagnostics_formats:
        output_path = config.diagnostics_dir / "plots" / f"{stem}.{fmt}"
        _plot_raw_vs_qc_segments(result, file_path=file_path, output_path=output_path, config=config)
        written.append(output_path)

    return written


def _count_nested_reasons(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw_counts = row.get(column, {})
        if not isinstance(raw_counts, dict):
            continue
        for reason, count in raw_counts.items():
            counts[str(reason)] = counts.get(str(reason), 0) + int(count)
    return dict(sorted(counts.items()))


def _format_reason_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{reason}: {count}" for reason, count in counts.items())


def _print_diagnostics_summary(
    summaries: list[dict[str, Any]],
    merge_events: list[dict[str, Any]],
    observation_artifacts: ObservationRunArtifacts | None = None,
) -> None:
    if not summaries:
        print("No RTRAJ files were processed.")
        return

    frame = pd.DataFrame(summaries)
    total_files = int(len(frame))
    total_measurements = int(frame.get("raw_measurement_rows", frame["raw_points"]).sum())
    total_non_fixes = int(frame.get("non_fix_rows", pd.Series([0] * len(frame))).sum())
    total_finite_fixes = int(frame.get("finite_trajectory_fix_rows", frame["raw_points"]).sum())
    total_cycle_representative_dropped = int(
        frame.get("cycle_representative_dropped_points", pd.Series([0] * len(frame))).sum()
    )
    total_missing_cycle_dropped = int(frame.get("missing_cycle_dropped_points", pd.Series([0] * len(frame))).sum())
    total_raw = int(frame["raw_points"].sum())
    total_kept = int(frame["kept_points"].sum())
    total_dropped = int(frame["dropped_points"].sum())
    total_initial_segments = int(frame["initial_segments"].sum())
    total_qc_merged_segments = int(frame["merged_segments"].sum())
    total_controlled_segments = int(frame["controlled_segments"].sum())
    total_depth_bin_segments = int(frame["depth_bin_segments"].sum())
    total_depth_unassigned = int(frame["depth_bin_unassigned_points"].sum())
    total_depth_finite = int(frame["depth_finite_points"].sum())
    total_depth_missing = int(frame["depth_missing_points"].sum())
    total_depth_filled = int(frame["depth_bin_fill_count"].sum())
    total_depth_repaired = int(frame.get("depth_bin_repair_count", pd.Series([0] * len(frame))).sum())
    total_depth_transitions = int(frame["depth_bin_transition_count"].sum())
    total_region_input_segments = int(frame.get("region_input_segments", frame["depth_bin_segments"]).sum())
    total_region_output_segments = int(frame.get("region_output_segments", frame["controlled_segments"]).sum())
    total_region_dropped_segments = int(frame.get("region_dropped_segments", pd.Series([0] * len(frame))).sum())
    total_region_trimmed_segments = int(frame.get("region_trimmed_segments", pd.Series([0] * len(frame))).sum())
    total_region_min_length_dropped = int(
        frame.get("region_min_length_dropped_segments", pd.Series([0] * len(frame))).sum()
    )
    total_region_input_points = int(frame.get("region_input_points", pd.Series([0] * len(frame))).sum())
    total_region_output_points = int(frame.get("region_output_points", pd.Series([0] * len(frame))).sum())
    total_resample_input_segments = int(frame.get("resample_input_segments", frame["controlled_segments"]).sum())
    total_resample_duration_dropped = int(
        frame.get("resample_duration_dropped_segments", pd.Series([0] * len(frame))).sum()
    )
    total_resample_empty_dropped = int(
        frame.get("resample_empty_dropped_segments", pd.Series([0] * len(frame))).sum()
    )
    total_resample_output_segments = int(frame.get("resample_output_segments", frame["controlled_segments"]).sum())
    total_resample_input_points = int(frame.get("resample_input_points", pd.Series([0] * len(frame))).sum())
    total_resample_output_points = int(frame.get("resample_output_points", pd.Series([0] * len(frame))).sum())
    total_jump_dropped = int(frame.get("jump_qc_dropped_points", pd.Series([0] * len(frame))).sum())
    total_jump_blocks = int(frame.get("jump_qc_dropped_blocks", pd.Series([0] * len(frame))).sum())
    total_jump_splits = int(frame.get("jump_qc_split_count", pd.Series([0] * len(frame))).sum())
    total_jump_post_merges = int(frame.get("jump_qc_post_merge_count", pd.Series([0] * len(frame))).sum())
    total_jump_post_merge_rejections = int(
        frame.get("jump_qc_post_merge_rejection_count", pd.Series([0] * len(frame))).sum()
    )
    total_segments_after_jump = int(frame.get("segments_after_jump_qc", frame["merged_segments"]).sum())
    total_merge_boundaries = int(len(merge_events))
    total_merged = int(sum(1 for event in merge_events if event["merged"]))
    total_not_merged = total_merge_boundaries - total_merged

    controlled_segments = pd.to_numeric(frame["controlled_segments"], errors="coerce")
    raw_points = pd.to_numeric(frame["raw_points"], errors="coerce")
    kept_points = pd.to_numeric(frame["kept_points"], errors="coerce")

    drop_reasons = _count_nested_reasons(summaries, "drop_reason_counts")
    merge_rejections = _count_nested_reasons(summaries, "merge_rejection_counts")
    jump_reasons = _count_nested_reasons(summaries, "jump_qc_reason_counts")
    depth_bin_counts = _count_nested_reasons(summaries, "depth_bin_counts")
    depth_source_counts = _count_nested_reasons(summaries, "depth_source_counts")
    observation_matched_counts = _count_nested_reasons(summaries, "observation_matched_counts")
    observation_unmatched_counts = _count_nested_reasons(summaries, "observation_unmatched_counts")
    observation_filter_counts = _count_nested_reasons(summaries, "observation_filter_counts")
    observation_enabled = bool(
        frame.get("observation_sampling_enabled", pd.Series([False] * len(frame))).astype(bool).any()
    )
    observation_eligible_points = int(
        frame.get("observation_eligible_points", pd.Series([0] * len(frame))).sum()
    )
    observation_unknown_depth_points = int(
        frame.get("observation_unknown_depth_skipped_points", pd.Series([0] * len(frame))).sum()
    )
    zero_output = int((controlled_segments == 0).sum())

    print("")
    print("RTRAJ diagnostics summary")
    print(f"  files processed: {total_files}")
    print(f"  raw measurement rows: {total_measurements}")
    print(f"  finite trajectory fixes: {total_finite_fixes}")
    print(f"  cycle-representative dropped fixes: {total_cycle_representative_dropped}")
    print(f"  missing-cycle dropped fixes: {total_missing_cycle_dropped}")
    print(f"  trajectory fixes: {total_raw}")
    print(f"  non-fix rows skipped before QC: {total_non_fixes}")
    print(f"  kept points: {total_kept}")
    print(f"  QC dropped points: {total_dropped}")
    print(f"  initial QC segments: {total_initial_segments}")
    print(f"  QC-merged segments: {total_qc_merged_segments}")
    print(f"  jump-QC dropped points: {total_jump_dropped}")
    print(f"  jump-QC dropped blocks: {total_jump_blocks}")
    print(f"  jump-QC split boundaries: {total_jump_splits}")
    print(f"  jump-QC post-drop merges: {total_jump_post_merges}")
    print(f"  jump-QC post-drop merge rejections: {total_jump_post_merge_rejections}")
    print(f"  segments after jump QC: {total_segments_after_jump}")
    print(f"  depth-bin segments before region selection: {total_depth_bin_segments}")
    print(f"  finite depth points before binning: {total_depth_finite}")
    print(f"  missing depth points before binning: {total_depth_missing}")
    print(f"  depth-bin unassigned points: {total_depth_unassigned}")
    print(f"  depth-bin filled points: {total_depth_filled}")
    print(f"  depth-bin repaired isolated points: {total_depth_repaired}")
    print(f"  depth-bin transitions: {total_depth_transitions}")
    print(f"  region input segments: {total_region_input_segments}")
    print(f"  region output segments: {total_region_output_segments}")
    print(f"  region dropped segments: {total_region_dropped_segments}")
    print(f"  region trimmed segments: {total_region_trimmed_segments}")
    print(f"  region min-length dropped segments: {total_region_min_length_dropped}")
    print(f"  region input/output points: {total_region_input_points}/{total_region_output_points}")
    print(f"  controlled segments after region selection: {total_controlled_segments}")
    print(f"  resample input segments: {total_resample_input_segments}")
    print(f"  resample duration-dropped segments: {total_resample_duration_dropped}")
    print(f"  resample empty-dropped segments: {total_resample_empty_dropped}")
    print(f"  output segments after resampling: {total_resample_output_segments}")
    print(f"  resample input/output points: {total_resample_input_points}/{total_resample_output_points}")
    print(f"  files with zero controlled segments: {zero_output}")
    print(
        "  controlled segments per file: "
        f"min={int(controlled_segments.min())}, "
        f"median={float(controlled_segments.median()):.1f}, "
        f"max={int(controlled_segments.max())}"
    )
    print(
        "  points per file: "
        f"raw median={float(raw_points.median()):.1f}, "
        f"kept median={float(kept_points.median()):.1f}"
    )
    print(f"  merge boundaries checked: {total_merge_boundaries}")
    print(f"  merged boundaries: {total_merged}")
    print(f"  not-merged boundaries: {total_not_merged}")
    print(f"  QC drop reasons: {_format_reason_counts(drop_reasons)}")
    print(f"  not-merged reasons: {_format_reason_counts(merge_rejections)}")
    print(f"  jump-QC reasons: {_format_reason_counts(jump_reasons)}")
    print(f"  depth sources: {_format_reason_counts(depth_source_counts)}")
    print(f"  depth-bin point counts: {_format_reason_counts(depth_bin_counts)}")
    if observation_enabled:
        print(f"  observation-eligible resampled points: {observation_eligible_points}")
        print(f"  observation points skipped for unknown depth: {observation_unknown_depth_points}")
        print(
            "  observation source filtering: "
            f"{_format_reason_counts(observation_filter_counts)}"
        )
        variable_names = sorted(set(observation_matched_counts) | set(observation_unmatched_counts))
        for name in variable_names:
            time_values = (
                observation_artifacts.time_mismatch_days.get(name, [])
                if observation_artifacts is not None
                else []
            )
            pressure_values = (
                observation_artifacts.pressure_mismatch_dbar.get(name, [])
                if observation_artifacts is not None
                else []
            )
            median_time = float(np.median(time_values)) if time_values else np.nan
            median_pressure = float(np.median(pressure_values)) if pressure_values else np.nan
            print(
                f"  observation {name}: matched={observation_matched_counts.get(name, 0)}, "
                f"unmatched={observation_unmatched_counts.get(name, 0)}, "
                f"median |time mismatch|={median_time:.6g} days, "
                f"median |pressure mismatch|={median_pressure:.6g} dbar"
            )


def _diagnostic_plots_enabled(config: RtrajConfig) -> bool:
    if config.mode != "diagnostics":
        return False
    plots = (config.raw.get("diagnostics", {}) or {}).get("plots", {}) or {}
    return bool(plots.get("raw_vs_qc_segments", True))


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


def _count_platforms(trajectories: list[pd.DataFrame]) -> int:
    platforms: set[Any] = set()
    for trajectory in trajectories:
        if trajectory.empty or "platform_code" not in trajectory.columns:
            continue
        platforms.add(trajectory["platform_code"].iloc[0])
    return len(platforms)


def _count_observations(trajectories: list[pd.DataFrame]) -> int:
    return int(sum(len(trajectory) for trajectory in trajectories))


def _prepare_output_trajectory(trajectory: pd.DataFrame, config: RtrajConfig) -> pd.DataFrame:
    depth_name = config.normalized_variables["depth"]
    out = trajectory.copy().reset_index(drop=True)
    if depth_name not in out.columns:
        raise KeyError(f"Output trajectory is missing depth column: {depth_name}")

    out["z"] = pd.to_numeric(out[depth_name], errors="coerce")
    out = out.drop(columns=[depth_name])
    drop_columns = [column for column in INTERNAL_OUTPUT_COLUMNS if column in out.columns]
    if drop_columns:
        out = out.drop(columns=drop_columns)
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def prepare_output_trajectories(
    segments: list[pd.DataFrame],
    config: RtrajConfig,
) -> list[pd.DataFrame]:
    trajectories: list[pd.DataFrame] = []
    for segment in segments:
        if segment.empty:
            continue
        prepared = _prepare_output_trajectory(segment, config)
        if prepared.empty:
            continue
        trajectories.append(prepared)
    return trajectories


def _build_rtraj_dataset(
    trajectories: list[pd.DataFrame],
    *,
    dataset_attrs: dict[str, Any] | None = None,
    variable_attrs: dict[str, dict[str, Any]] | None = None,
) -> xr.Dataset:
    attrs = dict(DEFAULT_DATASET_ATTRS)
    if dataset_attrs:
        attrs.update(dataset_attrs)

    ds = build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
        dataset_attrs=attrs,
    )
    for name, attrs_for_variable in (variable_attrs or {}).items():
        if name in ds:
            ds[name].attrs.update(attrs_for_variable)
    return ds


def _write_dataset_to_zarr(ds: xr.Dataset, output_path: Path, *, overwrite: bool) -> None:
    encoding = build_zarr_encoding(ds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "w-"
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode=mode, encoding=encoding)


def write_output_zarr(
    segments: list[pd.DataFrame],
    config: RtrajConfig,
    *,
    variable_attrs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    prepared = prepare_output_trajectories(segments, config)
    if not prepared:
        raise ValueError("No output trajectories were produced after Rtraj filtering and processing")

    output_counts: dict[str, dict[str, Any]] = {}
    if config.depth_bins.enabled and config.depth_bins.output_mode == "per_bin":
        wrote_any = False
        depth_bin_iterator = (
            tqdm(config.depth_bins.bins, desc="Writing depth-bin Zarr", unit="bin")
            if len(config.depth_bins.bins) > 1
            else config.depth_bins.bins
        )
        for depth_bin in depth_bin_iterator:
            bin_trajectories = [
                trajectory.reset_index(drop=True)
                for trajectory in prepared
                if "depth_bin" in trajectory.columns and str(trajectory["depth_bin"].iloc[0]) == depth_bin.label
            ]
            if not bin_trajectories:
                print(f"Skipping empty depth bin: {depth_bin.label}")
                continue

            normalized = normalize_trajectories(bin_trajectories, show_progress=False)
            ds = _build_rtraj_dataset(
                normalized,
                dataset_attrs=_depth_bin_attrs(depth_bin),
                variable_attrs=variable_attrs,
            )
            current_output_path = _depth_bin_output_path(config.output.zarr_path, depth_bin.label)
            _write_dataset_to_zarr(ds, current_output_path, overwrite=config.output.overwrite)
            output_counts[depth_bin.label] = {
                "path": str(current_output_path),
                "trajectories": len(normalized),
                "platforms": _count_platforms(normalized),
                "observations": _count_observations(normalized),
            }
            wrote_any = True

        if not wrote_any:
            raise ValueError("No depth-bin Zarr datasets were produced after Rtraj filtering and processing")
        return output_counts

    normalized = normalize_trajectories(prepared, show_progress=False)
    ds = _build_rtraj_dataset(
        normalized,
        dataset_attrs={"depth_bins_enabled": bool(config.depth_bins.enabled)},
        variable_attrs=variable_attrs,
    )
    _write_dataset_to_zarr(ds, config.output.zarr_path, overwrite=config.output.overwrite)
    output_counts["all"] = {
        "path": str(config.output.zarr_path),
        "trajectories": len(normalized),
        "platforms": _count_platforms(normalized),
        "observations": _count_observations(normalized),
    }
    return output_counts


def _print_output_summary(output_counts: dict[str, dict[str, Any]]) -> None:
    print("")
    print("RTRAJ output datasets")
    for label, counts in output_counts.items():
        print(
            f"  {label}: {counts['trajectories']} trajectory segment(s), "
            f"{counts['platforms']} platform(s), {counts['observations']} observation(s), "
            f"path={counts['path']}"
        )


def run_stage_one(
    config: RtrajConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[pd.DataFrame],
    ObservationRunArtifacts,
]:
    summaries: list[dict[str, Any]] = []
    merge_events: list[dict[str, Any]] = []
    jump_events: list[dict[str, Any]] = []
    output_segments: list[pd.DataFrame] = []
    observation_artifacts = ObservationRunArtifacts(
        variable_attrs={},
        time_mismatch_days={name: [] for name in config.observations.variables},
        pressure_mismatch_dbar={name: [] for name in config.observations.variables},
    )
    should_write_plots = _diagnostic_plots_enabled(config)
    file_iterator = (
        tqdm(config.input_files, desc="Processing RTRAJ files", unit="file")
        if len(config.input_files) > 1
        else config.input_files
    )
    for file_index, file_path in enumerate(file_iterator, start=1):
        file_data = _read_rtraj_file_data(file_path, config)
        raw = file_data.trajectory_fixes
        result = process_qc_stage(raw, config)
        sampled = sample_observations_onto_segments(
            result.output_segments,
            file_data.observations,
            config,
        )
        output_segments.extend(sampled.segments)
        for name, attrs in file_data.observation_variable_attrs.items():
            merged_attrs = observation_artifacts.variable_attrs.setdefault(name, {})
            for key, value in attrs.items():
                merged_attrs.setdefault(key, value)
        for name, values in sampled.time_mismatch_days.items():
            observation_artifacts.time_mismatch_days.setdefault(name, []).extend(values)
        for name, values in sampled.pressure_mismatch_dbar.items():
            observation_artifacts.pressure_mismatch_dbar.setdefault(name, []).extend(values)
        written = (
            write_diagnostics(result, file_path=file_path, file_index=file_index, config=config)
            if should_write_plots
            else []
        )
        platform_code = raw["platform_code"].iloc[0] if not raw.empty else None
        for event in result.merge_events:
            merge_events.append(
                {
                    "file": str(file_path),
                    "platform_code": platform_code,
                    **event,
                }
            )
        for event in result.jump_events:
            jump_events.append(
                {
                    "file": str(file_path),
                    "platform_code": platform_code,
                    **event,
                }
            )
        summaries.append(
            {
                "file": str(file_path),
                "platform_code": platform_code,
                **result.summary,
                "observation_filter_counts": file_data.observation_filter_counts,
                **sampled.summary,
                "plots": ";".join(str(path) for path in written),
            }
        )
    return summaries, merge_events, jump_events, output_segments, observation_artifacts


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = resolve_config(args.config)
    summaries, merge_events, jump_events, output_segments, observation_artifacts = run_stage_one(config)

    summary_path = config.diagnostics_dir / "controlled_stage_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    if config.mode == "diagnostics":
        merge_events_path = config.diagnostics_dir / "qc_merge_events.csv"
        pd.DataFrame(merge_events).to_csv(merge_events_path, index=False)
        jump_events_path = config.diagnostics_dir / "jump_qc_events.csv"
        pd.DataFrame(jump_events).to_csv(jump_events_path, index=False)
        _print_diagnostics_summary(summaries, merge_events, observation_artifacts)
        print(f"Wrote staged diagnostics for {len(summaries)} RTRAJ file(s) to {config.diagnostics_dir}")
    else:
        print(f"Wrote staged conversion summary for {len(summaries)} RTRAJ file(s) to {summary_path}")

    if config.output.write_zarr:
        output_counts = write_output_zarr(
            output_segments,
            config,
            variable_attrs=observation_artifacts.variable_attrs,
        )
        _print_output_summary(output_counts)


if __name__ == "__main__":
    main()
