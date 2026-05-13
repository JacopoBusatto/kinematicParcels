from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetCoordinatesConfig:
    trajectory: str = "trajectory"
    obs: str = "obs"
    time: str = "time"
    lon: str = "lon"
    lat: str = "lat"
    z: str | None = "z"


@dataclass(frozen=True)
class DatasetConfig:
    input_path: str
    coordinates: DatasetCoordinatesConfig = field(default_factory=DatasetCoordinatesConfig)


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Defines which analyses should be executed.

    The analyses are executed in the order specified.
    """
    types: tuple[str, ...] = ("trajectories",)


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "./outputs/postprocessing"


@dataclass(frozen=True)
class ExportsConfig:
    save_trajectory_table: bool = False
    save_particle_summary: bool = True
    table_format: str = "parquet"


@dataclass(frozen=True)
class GridConfig:
    mode: str = "from_initial_centers"
    lon_min: float | None = None
    lon_max: float | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    dlon: float = 0.025
    dlat: float = 0.025

    def __post_init__(self) -> None:
        if self.mode not in {"explicit_edges", "from_initial_centers"}:
            raise ValueError(
                "grid.mode must be one of: 'explicit_edges', 'from_initial_centers'"
            )

        if self.dlon <= 0 or self.dlat <= 0:
            raise ValueError("grid.dlon and grid.dlat must be positive.")

        if None in (self.lon_min, self.lon_max, self.lat_min, self.lat_max):
            raise ValueError(
                "grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max must be provided."
            )

        if self.lon_max <= self.lon_min:
            raise ValueError("grid.lon_max must be greater than grid.lon_min.")

        if self.lat_max <= self.lat_min:
            raise ValueError("grid.lat_max must be greater than grid.lat_min.")


@dataclass(frozen=True)
class DensityConfig:
    lon_col: str = "lon"
    lat_col: str = "lat"
    time_col: str = "time"
    normalize_active: bool = True
    normalize_total: bool = True
    fill_ever_active_empty_with_zero: bool = False
    # None = all members (default); int = only that group_member value
    group_member: int | None = None

    animate: bool = False
    animation_var: str = "particle_count"
    animation_label: str = "particle_count"
    animation_fps: int = 8
    animation_every_n: int = 1
    animation_vmin: float | None = None
    animation_vmax: float | None = None
    show_time_bar: bool = True


@dataclass(frozen=True)
class BeachingTimesConfig:
    lon_col: str = "lon0"
    lat_col: str = "lat0"
    value_col: str = "lifetime_seconds"
    statistic: str = "min"
    plot: bool = False


@dataclass(frozen=True)
class FSLEConfig:
    pair_mode: str = "center_pairs"
    min_scale: float = 5.0e-3
    max_scale: float = 1.0e4
    rho_increment: float = 2 ** 0.5
    save_crossing_events: bool = False
    plot: bool = True
    reference_slopes: tuple[str, ...] = ("delta^-2/3", "delta^-1", "delta^-2")
    reference_slope_anchor_scales: dict[str, float] = field(default_factory=dict)
    x_min: float | None = 1.0
    x_max: float | None = 2500.0
    y_min: float | None = 1.0e-2
    y_max: float | None = 1.0

    def __post_init__(self) -> None:
        if self.pair_mode not in {"center_pairs", "all_pairs"}:
            raise ValueError("fsle.pair_mode must be 'center_pairs' or 'all_pairs'.")

        if self.min_scale <= 0 or self.max_scale <= 0:
            raise ValueError("fsle.min_scale and fsle.max_scale must be positive.")

        if self.max_scale <= self.min_scale:
            raise ValueError("fsle.max_scale must be greater than fsle.min_scale.")

        if self.rho_increment <= 1.0:
            raise ValueError("fsle.rho_increment must be greater than 1.")

        valid_reference_slopes = {"delta^-2/3", "delta^-1", "delta^-2"}
        invalid_reference_slopes = [s for s in self.reference_slopes if s not in valid_reference_slopes]
        if invalid_reference_slopes:
            raise ValueError(
                "fsle.reference_slopes contains unsupported values: "
                f"{invalid_reference_slopes}. Supported: {sorted(valid_reference_slopes)}"
            )

        invalid_anchor_slopes = [
            slope for slope in self.reference_slope_anchor_scales if slope not in valid_reference_slopes
        ]
        if invalid_anchor_slopes:
            raise ValueError(
                "fsle.reference_slope_anchor_scales contains unsupported slope keys: "
                f"{invalid_anchor_slopes}. Supported: {sorted(valid_reference_slopes)}"
            )

        invalid_anchor_values = [
            value for value in self.reference_slope_anchor_scales.values() if value <= 0
        ]
        if invalid_anchor_values:
            raise ValueError("fsle.reference_slope_anchor_scales values must be positive.")


@dataclass(frozen=True)
class TrajectoriesConfig:
    plot: bool = True
    title: str = "Trajectories"
    show_start: bool = True
    show_end: bool = True
    alpha: float = 0.7
    plot_color_by: str | None = None  # None = auto (group_member if present)
    plot_cmap: str | None = None  # None = auto (viridis for numeric, tab10/20/hsv for categorical)
    plot_cmap_mode: str = "auto"  # "auto" | "categorical" | "numeric"

    animate: bool = False
    animation_fps: int = 8
    animation_every_n: int = 1
    animation_color_by: str = "lat0"
    animation_cmap: str | None = None  # None = auto (viridis for numeric, tab10/20/hsv for categorical)
    animation_cmap_mode: str = "auto"  # "auto" | "categorical" | "numeric"
    animation_vmin: float | None = None
    animation_vmax: float | None = None
    animation_label: str = "value"
    show_time_bar: bool = True
    trail: bool = True
    trail_steps: int | None = None

    # Grouped particle support
    max_group_member: int | None = None  # None = plot all members


@dataclass(frozen=True)
class StartEndRegionsConfig:
    region_labels: tuple[str, ...] | None = None
    how_many: str = "priority_max"
    priority_level: int | None = None
    priority_mode: str = "exact"
    input_lon_mode: str = "-180_180"
    plot: bool = False

    # Connectivity outputs: trajectory/segment plots coloured by region.
    # plot_connectivity activates both the trajectories-by-region PNG and
    # the dual-coloured connectivity map PNG.
    # connectivity_segments=True (default) draws straight start→end segments;
    # False loads full trajectory paths (heavier but accurate).
    plot_connectivity: bool = False
    animate_connectivity: bool = False
    connectivity_segments: bool = True
    connectivity_color_by: str = "start_region"
    connectivity_label: str = "region"
    connectivity_title: str = "Trajectories by region"
    connectivity_show_start: bool = True
    connectivity_show_end: bool = True
    connectivity_alpha: float | None = None
    connectivity_max_group_member: int | None = None
    connectivity_animation_fps: int | None = None
    connectivity_animation_show_tracer: bool | None = None
    connectivity_trail: bool | None = None
    connectivity_trail_steps: int | None = None


@dataclass(frozen=True)
class ReleaseConfig:
    """
    Describes how particles were released in the simulation.

    Used by the postprocessing pipeline to decide whether grid-based outputs
    (start/end region maps, NetCDF) are meaningful.
    Grid outputs are only produced when mode is 'region_grid' and continuous
    release is disabled.
    """
    mode: str = "region_grid"
    continuous: bool = False


@dataclass(frozen=True)
class PlottingConfig:
    projection: str = "PlateCarree"


@dataclass(frozen=True)
class CleaningConfig:
    truncate_stagnant: bool = False
    stagnant_tol: float = 1.0e-6
    stagnant_min_consecutive: int = 2


@dataclass(frozen=True)
class PostprocessConfig:
    dataset: DatasetConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    exports: ExportsConfig = field(default_factory=ExportsConfig)
    grid: GridConfig | None = None
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    beaching_times: BeachingTimesConfig = field(default_factory=BeachingTimesConfig)
    fsle: FSLEConfig = field(default_factory=FSLEConfig)
    trajectories: TrajectoriesConfig = field(default_factory=TrajectoriesConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    start_end_regions: StartEndRegionsConfig = field(default_factory=StartEndRegionsConfig)


@dataclass(frozen=True)
class ParcelsSchema:
    trajectory_dim: str
    obs_dim: str
    time_var: str
    lon_var: str
    lat_var: str
    z_var: str | None = None