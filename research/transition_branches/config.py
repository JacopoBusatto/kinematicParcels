"""Compact production configuration for one transition-branch realization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    transition_table: str
    matrix_id: str
    timestep: float
    time_unit: str


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_name: str = "lagrangian_currents"


LENGTH_UNITS_TO_METERS = {
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "m": 1.0,
    "km": 1.0e3,
}
TIME_UNITS = ("s", "min", "h", "day")


@dataclass(frozen=True)
class SpatialGeometryConfig:
    coordinate_system: str
    length_unit: str
    ellipsoid: str | None = None

    @property
    def length_suffix(self) -> str:
        return self.length_unit

    def rate_suffix(self, time_unit: str) -> str:
        return f"{self.length_unit}_{time_unit}"

    def area_rate_suffix(self, time_unit: str) -> str:
        return f"{self.length_unit}2_{time_unit}"

    def rate_gradient_suffix(self, time_unit: str) -> str:
        return f"{self.length_unit}_{time_unit}_per_{self.length_unit}"


@dataclass(frozen=True)
class GridConfig:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    dlon: float
    dlat: float
    periodic_longitude: bool = True

    @property
    def nlon(self) -> int:
        return round((self.lon_max - self.lon_min) / self.dlon)

    @property
    def nlat(self) -> int:
        return round((self.lat_max - self.lat_min) / self.dlat)

    @property
    def x_min(self) -> float:
        return self.lon_min

    @property
    def x_max(self) -> float:
        return self.lon_max

    @property
    def y_min(self) -> float:
        return self.lat_min

    @property
    def y_max(self) -> float:
        return self.lat_max

    @property
    def dx(self) -> float:
        return self.dlon

    @property
    def dy(self) -> float:
        return self.dlat

    @property
    def nx(self) -> int:
        return self.nlon

    @property
    def ny(self) -> int:
        return self.nlat

    @property
    def periodic_x(self) -> bool:
        return self.periodic_longitude


@dataclass(frozen=True)
class CartesianGridConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float

    @property
    def nx(self) -> int:
        return round((self.x_max - self.x_min) / self.dx)

    @property
    def ny(self) -> int:
        return round((self.y_max - self.y_min) / self.dy)

    @property
    def periodic_x(self) -> bool:
        return False


@dataclass(frozen=True)
class StatisticsConfig:
    min_moving_support: int = 10
    angular_bins: int = 36
    direction_zero_tolerance: float = 1.0e-12
    high_R1: float = 0.8
    low_R1: float = 0.5


@dataclass(frozen=True)
class BranchConfig:
    transport_percentile: float = 0.9
    ridge_field: str = "raw"
    transverse_scale_grid: float = 1.0
    interpolation_weight_tolerance: float = 1.0e-10
    orientation_reliable_R1: float = 0.8
    orientation_ambiguous_R1: float = 0.5
    direction_disagreement_degrees: float = 20.0
    abrupt_tangent_mismatch_degrees: float = 45.0
    ridge_comparison_tolerance: float = 1.0e-12
    smoothing_window_cells: int = 3


@dataclass(frozen=True)
class DirectionalConfig:
    minimum_P_move: float = 0.5
    minimum_R1: float = 0.8
    minimum_strength: float = 0.5
    maximum_neighbor_direction_difference_degrees: float = 45.0
    maximum_step_direction_mismatch_degrees: float = 45.0
    minimum_component_cells: int = 3
    transverse_scale_grid: float = 1.0


@dataclass(frozen=True)
class EdgeConfig:
    half_width_grid_scales: int = 5
    sampling_interval_grid_scales: float = 1.0
    core_refinement_grid_scales: float = 1.0
    robust_median_window_samples: int = 3
    composite_half_window_sections: int = 2
    minimum_persistent_neighbor_sections: int = 2
    minimum_persistent_fraction: float = 0.5
    diagnostic_low_R1: float = 0.5
    diagnostic_large_direction_disagreement_degrees: float = 20.0
    diagnostic_high_curvature_degrees: float = 60.0
    diagnostic_strong_outer_recovery_fraction: float = 0.5
    diagnostic_min_full_section_valid_samples: int = 7
    nearby_branch_cross_distance_scales: float = 5.0
    nearby_branch_along_distance_scales: float = 1.0


@dataclass(frozen=True)
class ValidationConfig:
    normalization_atol: float = 1.0e-12
    probability_rtol: float = 1.0e-10
    probability_atol: float = 1.0e-12
    center_atol: float = 1.0e-9
    gradient_zero_tolerance: float = 1.0e-12
    interpolation_weight_tolerance: float = 1.0e-10
    gradient_search_radius_grid_scales: float = 1.0
    local_background_radius_grid_scales: float = 2.0
    duplicate_disagreement_grid_scales: float = 1.0
    core_gradient_ratio_epsilon: float = 1.0e-12
    multiple_drop_similarity_fraction: float = 0.1
    direct_sample_atol_grid_cells: float = 1.0e-8


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    dpi: int = 160
    projection: str = "SouthPolarStereo"
    central_longitude: float = 0.0
    circular_boundary: bool = True
    draw_coastlines: bool = True
    vector_stride_cells: int = 5
    vector_reference: float = 5.0
    directional_vector_reference: float = 0.5
    structure_map_max_percentile: float = 100.0
    debug_plots: bool = False


@dataclass(frozen=True)
class CompactConfig:
    input: InputConfig
    output: OutputConfig
    geometry: SpatialGeometryConfig
    grid: GridConfig | CartesianGridConfig
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    branches: BranchConfig = field(default_factory=BranchConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
    write_debug_outputs: bool = False
    run_validation: bool = False
    analysis_version: str = "4.0.0-production"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input.matrix_id, str)
            or not self.input.matrix_id.strip()
            or not isinstance(self.input.timestep, (int, float))
            or not math.isfinite(self.input.timestep)
            or self.input.timestep <= 0
        ):
            raise ValueError("input matrix_id and positive timestep are required")
        if self.input.time_unit not in TIME_UNITS:
            raise ValueError(f"unsupported time_unit: {self.input.time_unit}")
        if self.geometry.coordinate_system not in {"geographic", "cartesian"}:
            raise ValueError(
                "geometry.coordinate_system must be 'geographic' or 'cartesian'"
            )
        if self.geometry.length_unit not in LENGTH_UNITS_TO_METERS:
            raise ValueError(
                f"unsupported length_unit: {self.geometry.length_unit}"
            )
        if self.geometry.coordinate_system == "geographic":
            if not isinstance(self.grid, GridConfig):
                raise ValueError("geographic geometry requires a longitude/latitude grid")
            if not self.geometry.ellipsoid or not self.geometry.ellipsoid.strip():
                raise ValueError("geographic geometry requires geometry.ellipsoid")
        else:
            if not isinstance(self.grid, CartesianGridConfig):
                raise ValueError("cartesian geometry requires an x/y grid")
            if self.geometry.ellipsoid is not None:
                raise ValueError("cartesian geometry must not define an ellipsoid")
        grid_values = (
            self.grid.x_min,
            self.grid.x_max,
            self.grid.y_min,
            self.grid.y_max,
            self.grid.dx,
            self.grid.dy,
        )
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in grid_values
        ):
            raise ValueError("grid bounds and spacing must be finite numbers")
        if min(self.grid.dx, self.grid.dy) <= 0:
            raise ValueError("grid spacing must be positive")
        if (
            self.grid.x_max <= self.grid.x_min
            or self.grid.y_max <= self.grid.y_min
        ):
            raise ValueError("grid maxima must exceed minima")
        for axis, span, spacing in (
            ("x", self.grid.x_max - self.grid.x_min, self.grid.dx),
            ("y", self.grid.y_max - self.grid.y_min, self.grid.dy),
        ):
            count = span / spacing
            if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"grid {axis} span must be an integer multiple of its spacing"
                )
        if isinstance(self.grid, GridConfig):
            if not -90.0 <= self.grid.lat_min < self.grid.lat_max <= 90.0:
                raise ValueError("geographic grid latitude bounds must lie in [-90, 90]")
            if not isinstance(self.grid.periodic_longitude, bool):
                raise TypeError("grid.periodic_longitude must be boolean")
        if self.statistics.min_moving_support <= 0:
            raise ValueError("min_moving_support must be positive")
        if self.statistics.angular_bins < 4:
            raise ValueError("angular_bins must be at least four")
        if not 0 <= self.statistics.low_R1 <= self.statistics.high_R1 <= 1:
            raise ValueError("R1 diagnostic thresholds must be ordered in [0, 1]")
        if self.statistics.direction_zero_tolerance < 0:
            raise ValueError("direction_zero_tolerance must be nonnegative")
        if not 0 < self.branches.transport_percentile < 1:
            raise ValueError("transport_percentile must lie in (0, 1)")
        if self.branches.ridge_field not in {"raw", "smoothed"}:
            raise ValueError("ridge_field must be 'raw' or 'smoothed'")
        if self.branches.transverse_scale_grid <= 0:
            raise ValueError("transverse_scale_grid must be positive")
        if self.branches.smoothing_window_cells != 3:
            raise ValueError("the validated optional smoothing window is 3x3")
        if (
            min(
                self.branches.interpolation_weight_tolerance,
                self.branches.ridge_comparison_tolerance,
            )
            < 0
        ):
            raise ValueError("branch numerical tolerances must be nonnegative")
        if not (
            0
            <= self.branches.orientation_ambiguous_R1
            <= self.branches.orientation_reliable_R1
            <= 1
        ):
            raise ValueError(
                "branch R1 diagnostic thresholds must be ordered in [0, 1]"
            )
        if any(
            not 0 <= value <= 180
            for value in (
                self.branches.direction_disagreement_degrees,
                self.branches.abrupt_tangent_mismatch_degrees,
            )
        ):
            raise ValueError("branch angular diagnostics must lie in [0, 180]")
        if any(
            not 0 <= value <= 1
            for value in (
                self.directional.minimum_P_move,
                self.directional.minimum_R1,
                self.directional.minimum_strength,
            )
        ):
            raise ValueError("directional probability/strength thresholds must be in [0, 1]")
        if any(
            not 0 <= value <= 90
            for value in (
                self.directional.maximum_neighbor_direction_difference_degrees,
                self.directional.maximum_step_direction_mismatch_degrees,
            )
        ):
            raise ValueError("directional local-angle thresholds must be in [0, 90]")
        if self.directional.minimum_component_cells < 1:
            raise ValueError("minimum_component_cells must be positive")
        if self.directional.transverse_scale_grid <= 0:
            raise ValueError("directional transverse scale must be positive")
        if self.edges.half_width_grid_scales < 2:
            raise ValueError("cross-section half-width must be at least two cells")
        if self.edges.sampling_interval_grid_scales <= 0:
            raise ValueError("cross-section sampling interval must be positive")
        if not 0 < self.edges.core_refinement_grid_scales <= 1:
            raise ValueError("core refinement must be within one grid scale")
        if (
            self.edges.robust_median_window_samples < 1
            or self.edges.robust_median_window_samples % 2 == 0
        ):
            raise ValueError("robust median window must be a positive odd integer")
        if self.edges.composite_half_window_sections < 0:
            raise ValueError("composite half-width must be nonnegative")
        if self.edges.minimum_persistent_neighbor_sections < 1:
            raise ValueError("minimum persistent neighbors must be positive")
        if not 0 <= self.edges.minimum_persistent_fraction <= 1:
            raise ValueError("minimum persistent fraction must lie in [0, 1]")
        if (
            min(
                self.validation.normalization_atol,
                self.validation.probability_rtol,
                self.validation.probability_atol,
                self.validation.center_atol,
                self.validation.gradient_zero_tolerance,
                self.validation.interpolation_weight_tolerance,
                self.validation.gradient_search_radius_grid_scales,
                self.validation.local_background_radius_grid_scales,
                self.validation.direct_sample_atol_grid_cells,
            )
            < 0
        ):
            raise ValueError("validation tolerances and radii must be nonnegative")
        if self.plotting.projection not in {"PlateCarree", "SouthPolarStereo"}:
            raise ValueError("unsupported plotting projection")
        if self.plotting.dpi <= 0:
            raise ValueError("plotting dpi must be positive")
        if self.plotting.vector_stride_cells < 1:
            raise ValueError("vector_stride_cells must be positive")
        if self.plotting.vector_reference <= 0:
            raise ValueError("vector_reference must be positive")
        if not 0 < self.plotting.directional_vector_reference <= 1:
            raise ValueError("directional_vector_reference must lie in (0, 1]")
        if not 0 < self.plotting.structure_map_max_percentile <= 100:
            raise ValueError("structure_map_max_percentile must lie in (0, 100]")
        if self.plotting.debug_plots and not self.run_validation:
            raise ValueError("debug validation plots require run_validation=true")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.geometry.coordinate_system == "cartesian":
            data["geometry"].pop("ellipsoid", None)
        return data

    @property
    def rate_unit(self) -> str:
        return f"{self.geometry.length_unit}/{self.input.time_unit}"

    @property
    def rate_suffix(self) -> str:
        return self.geometry.rate_suffix(self.input.time_unit)

    @property
    def area_rate_suffix(self) -> str:
        return self.geometry.area_rate_suffix(self.input.time_unit)

    @property
    def rate_gradient_suffix(self) -> str:
        return self.geometry.rate_gradient_suffix(self.input.time_unit)

    @property
    def geometry_metadata(self) -> dict[str, str]:
        return {
            "coordinate_system": self.geometry.coordinate_system,
            "coordinate_unit": (
                "degree"
                if self.geometry.coordinate_system == "geographic"
                else self.geometry.length_unit
            ),
            "length_unit": self.geometry.length_unit,
            "time_unit": self.input.time_unit,
            "rate_unit": self.rate_unit,
            "bearing_convention": (
                "degrees clockwise from positive y/north; "
                "0=+y/north, 90=+x/east"
            ),
            "geometry_backend": (
                f"pyproj.Geod({self.geometry.ellipsoid})"
                if self.geometry.coordinate_system == "geographic"
                else "Euclidean planar"
            ),
        }


def _construct(cls, values: Mapping[str, Any], section: str):
    if not isinstance(values, Mapping):
        raise TypeError(f"{section} must be a YAML mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} keys: {', '.join(unknown)}")
    try:
        return cls(**dict(values))
    except TypeError as exc:
        raise ValueError(f"invalid {section} configuration: {exc}") from exc


def load_config(path: str | Path) -> CompactConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("configuration root must be a YAML mapping")
    nested = {
        "input",
        "output",
        "geometry",
        "grid",
        "statistics",
        "branches",
        "directional",
        "edges",
        "validation",
        "plotting",
    }
    scalar = {"write_debug_outputs", "run_validation", "analysis_version"}
    generated = {"resolved_geometry"}
    unknown = sorted(set(raw) - nested - scalar - generated)
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
    missing = sorted({"input", "output", "geometry", "grid"} - set(raw))
    if missing:
        raise ValueError(f"missing configuration sections: {', '.join(missing)}")

    input_config = _construct(InputConfig, raw["input"], "input")
    output_config = _construct(OutputConfig, raw["output"], "output")
    geometry = _construct(SpatialGeometryConfig, raw["geometry"], "geometry")
    grid_type = (
        GridConfig
        if geometry.coordinate_system == "geographic"
        else CartesianGridConfig
        if geometry.coordinate_system == "cartesian"
        else None
    )
    if grid_type is None:
        raise ValueError(
            "geometry.coordinate_system must be 'geographic' or 'cartesian'"
        )
    grid = _construct(grid_type, raw["grid"], "grid")
    config = CompactConfig(
        input=input_config,
        output=output_config,
        geometry=geometry,
        grid=grid,
        statistics=_construct(
            StatisticsConfig, raw.get("statistics", {}), "statistics"
        ),
        branches=_construct(BranchConfig, raw.get("branches", {}), "branches"),
        directional=_construct(
            DirectionalConfig, raw.get("directional", {}), "directional"
        ),
        edges=_construct(EdgeConfig, raw.get("edges", {}), "edges"),
        validation=_construct(
            ValidationConfig, raw.get("validation", {}), "validation"
        ),
        plotting=_construct(PlotConfig, raw.get("plotting", {}), "plotting"),
        **{key: raw[key] for key in scalar if key in raw},
    )
    for section, key in (("input", "transition_table"), ("output", "root")):
        values = getattr(config, section)
        candidate = Path(getattr(values, key)).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        config = replace(config, **{section: replace(values, **{key: str(candidate)})})
    if "resolved_geometry" in raw and raw["resolved_geometry"] != config.geometry_metadata:
        raise ValueError("resolved_geometry metadata does not match the configuration")
    return config
