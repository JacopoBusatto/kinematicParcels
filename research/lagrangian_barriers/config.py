from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class InputConfig:
    transition_table: str
    companion_netcdf: str | None = None
    timestep_days: float = 10.0


@dataclass(frozen=True)
class GridConfig:
    lon_min: float = -180.0
    lon_max: float = 180.0
    lat_min: float = -80.0
    lat_max: float = -30.0
    dlon: float = 1.0
    dlat: float = 1.0
    periodic_longitude: bool = True

    @property
    def nlon(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.dlon))

    @property
    def nlat(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.dlat))


@dataclass(frozen=True)
class ValidationConfig:
    normalization_atol: float = 1.0e-12
    probability_rtol: float = 1.0e-10
    probability_atol: float = 1.0e-12
    center_atol_degrees: float = 1.0e-9
    fail_on_error: bool = True
    calculate_sha256: bool = True


@dataclass(frozen=True)
class GeometryConfig:
    ellipsoid: str = "WGS84"


@dataclass(frozen=True)
class ModesConfig:
    angular_bins: int = 72
    smoothing_bandwidth_degrees: float = 15.0
    min_start_count: int = 20
    min_moving_count: int = 10
    min_mode_count: int = 5
    min_mode_probability: float = 0.15
    min_relative_prominence: float = 0.10
    min_peak_separation_degrees: float = 30.0
    min_mean_distance_km: float = 0.0


@dataclass(frozen=True)
class GraphConfig:
    cluster_radius_grid_diagonals: float = 1.5
    cluster_bearing_degrees: float = 30.0
    alignment_scale_degrees: float = 30.0
    max_angular_mismatch_degrees: float = 60.0
    max_edge_distance_km: float = 400.0
    min_relative_score: float = 0.35
    cumulative_endpoint_mass: float = 0.80


@dataclass(frozen=True)
class BranchesConfig:
    sample_spacing_km: float = 50.0
    smoothing_window: int = 5
    smoothing_order: int = 2
    min_scan_length_km: float = 200.0
    major_branch_length_km: float = 1000.0


@dataclass(frozen=True)
class PermeabilityConfig:
    min_offset_km: float = -300.0
    max_offset_km: float = 300.0
    offset_spacing_km: float = 25.0
    source_along_halfwidth_km: float = 250.0
    source_normal_halfwidth_km: float = 150.0
    min_source_cells_per_side: int = 2
    min_counts_per_side: int = 100
    min_moving_counts_per_side: int = 50
    geometry_fraction: float = 0.5
    on_line_tolerance_km: float = 1.0e-9
    confidence_level: float = 0.95
    save_contributions: bool = True


@dataclass(frozen=True)
class BarriersConfig:
    smoothing_sigma_steps: float = 1.0
    min_prominence: float = 0.02
    min_width_km: float = 25.0
    min_separation_km: float = 50.0
    max_offset_jump_km: float = 75.0
    max_physical_gap_km: float = 150.0
    core_sign_band_km: float = 25.0
    max_missing_sections: int = 1
    min_segment_points: int = 3
    min_segment_length_km: float = 100.0


@dataclass(frozen=True)
class PlottingConfig:
    enabled: bool = True
    dpi: int = 150
    max_example_cells: int = 4
    projection: str = "PlateCarree"
    central_longitude: float = 0.0
    circular_boundary: bool = False
    draw_coastlines: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_name: str = "argo_10d_baseline"


@dataclass(frozen=True)
class BarrierAnalysisConfig:
    input: InputConfig
    output: OutputConfig
    grid: GridConfig = field(default_factory=GridConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    branches: BranchesConfig = field(default_factory=BranchesConfig)
    permeability: PermeabilityConfig = field(default_factory=PermeabilityConfig)
    barriers: BarriersConfig = field(default_factory=BarriersConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    random_seed: int = 0
    analysis_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.input.timestep_days <= 0:
            raise ValueError("input.timestep_days must be positive")
        if min(self.grid.dlon, self.grid.dlat) <= 0:
            raise ValueError("grid spacing must be positive")
        if self.grid.lon_max <= self.grid.lon_min or self.grid.lat_max <= self.grid.lat_min:
            raise ValueError("grid maximums must exceed minimums")
        if self.modes.angular_bins < 8:
            raise ValueError("modes.angular_bins must be at least 8")
        if not 0 < self.modes.min_mode_probability <= 1:
            raise ValueError("modes.min_mode_probability must be in (0, 1]")
        if not 0 < self.graph.cumulative_endpoint_mass <= 1:
            raise ValueError("graph.cumulative_endpoint_mass must be in (0, 1]")
        if self.branches.smoothing_window % 2 != 1:
            raise ValueError("branches.smoothing_window must be odd")
        if self.permeability.max_offset_km <= self.permeability.min_offset_km:
            raise ValueError("permeability offset range is empty")
        if self.permeability.offset_spacing_km <= 0:
            raise ValueError("permeability.offset_spacing_km must be positive")
        if self.plotting.projection not in {
            "PlateCarree", "SouthPolarStereo", "NorthPolarStereo", "Robinson", "Mercator"
        }:
            raise ValueError(
                "plotting.projection must be PlateCarree, SouthPolarStereo, "
                "NorthPolarStereo, Robinson, or Mercator"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _convert_value(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType) and type(None) in args:
        if value is None:
            return None
        annotation = next(arg for arg in args if arg is not type(None))
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        return _from_mapping(annotation, value, path)
    return value


def _from_mapping(cls: type, values: dict[str, Any], path: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown configuration keys under {path}: {unknown}")
    hints = get_type_hints(cls)
    kwargs = {
        name: _convert_value(hints[name], value, f"{path}.{name}")
        for name, value in values.items()
    }
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid or missing configuration under {path}: {exc}") from exc


def load_config(path: str | Path) -> BarrierAnalysisConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    cfg = _from_mapping(BarrierAnalysisConfig, raw, "config")
    input_path = Path(cfg.input.transition_table).expanduser()
    output_root = Path(cfg.output.root).expanduser()
    companion = cfg.input.companion_netcdf
    data = cfg.to_dict()
    data["input"]["transition_table"] = str(
        input_path if input_path.is_absolute() else (path.parent / input_path).resolve()
    )
    data["output"]["root"] = str(
        output_root if output_root.is_absolute() else (path.parent / output_root).resolve()
    )
    if companion is not None:
        cp = Path(companion).expanduser()
        data["input"]["companion_netcdf"] = str(
            cp if cp.is_absolute() else (path.parent / cp).resolve()
        )
    return _from_mapping(BarrierAnalysisConfig, data, "config")


def dump_config(config: BarrierAnalysisConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
