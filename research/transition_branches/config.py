"""Compact production configuration for one transition-branch realization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class InputConfig:
    transition_table: str
    matrix_id: str
    timestep_days: float


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_name: str = "lagrangian_currents"


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
        return round((self.lon_max - self.lon_min) / self.dlon)

    @property
    def nlat(self) -> int:
        return round((self.lat_max - self.lat_min) / self.dlat)


@dataclass(frozen=True)
class StatisticsConfig:
    min_moving_support: int = 10
    angular_bins: int = 36
    direction_zero_tolerance_km: float = 1.0e-12
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
    ridge_comparison_tolerance_km_day: float = 1.0e-12
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
    center_atol_degrees: float = 1.0e-9
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
    vector_reference_km_day: float = 5.0
    directional_vector_reference: float = 0.5
    structure_map_max_percentile: float = 100.0
    debug_plots: bool = False


@dataclass(frozen=True)
class CompactConfig:
    input: InputConfig
    output: OutputConfig
    grid: GridConfig = field(default_factory=GridConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    branches: BranchConfig = field(default_factory=BranchConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
    ellipsoid: str = "WGS84"
    write_debug_outputs: bool = False
    run_validation: bool = False
    analysis_version: str = "3.0.0-production"

    def __post_init__(self) -> None:
        if not self.input.matrix_id.strip() or self.input.timestep_days <= 0:
            raise ValueError("input matrix_id and positive timestep_days are required")
        if min(self.grid.dlon, self.grid.dlat) <= 0:
            raise ValueError("grid spacing must be positive")
        if (
            self.grid.lon_max <= self.grid.lon_min
            or self.grid.lat_max <= self.grid.lat_min
        ):
            raise ValueError("grid maxima must exceed minima")
        if self.statistics.min_moving_support <= 0:
            raise ValueError("min_moving_support must be positive")
        if self.statistics.angular_bins < 4:
            raise ValueError("angular_bins must be at least four")
        if not 0 <= self.statistics.low_R1 <= self.statistics.high_R1 <= 1:
            raise ValueError("R1 diagnostic thresholds must be ordered in [0, 1]")
        if self.statistics.direction_zero_tolerance_km < 0:
            raise ValueError("direction_zero_tolerance_km must be nonnegative")
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
                self.branches.ridge_comparison_tolerance_km_day,
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
                self.validation.center_atol_degrees,
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
        if self.plotting.vector_reference_km_day <= 0:
            raise ValueError("vector_reference_km_day must be positive")
        if not 0 < self.plotting.directional_vector_reference <= 1:
            raise ValueError("directional_vector_reference must lie in (0, 1]")
        if not 0 < self.plotting.structure_map_max_percentile <= 100:
            raise ValueError("structure_map_max_percentile must lie in (0, 100]")
        if self.plotting.debug_plots and not self.run_validation:
            raise ValueError("debug validation plots require run_validation=true")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls, values):
    hints = get_type_hints(cls)
    converted = {}
    for name, field_type in hints.items():
        if name not in values:
            continue
        value = values[name]
        if hasattr(field_type, "__dataclass_fields__"):
            value = _construct(field_type, value)
        converted[name] = value
    return cls(**converted)


def load_config(path: str | Path) -> CompactConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _construct(CompactConfig, raw)
    data = config.to_dict()
    for section, key in (("input", "transition_table"), ("output", "root")):
        candidate = Path(data[section][key]).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        data[section][key] = str(candidate)
    return _construct(CompactConfig, data)
