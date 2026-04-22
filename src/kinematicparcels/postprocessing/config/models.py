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
class TrajectoriesConfig:
    plot: bool = True
    title: str = "Trajectories"
    show_start: bool = True
    show_end: bool = True
    alpha: float = 0.7

    animate: bool = False
    animation_fps: int = 8
    animation_color_by: str = "lat0"
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