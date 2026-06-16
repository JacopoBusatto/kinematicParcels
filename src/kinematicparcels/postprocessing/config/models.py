from __future__ import annotations

from dataclasses import dataclass, field


def _validate_plot_limits(name: str, vmin: float | None, vmax: float | None) -> None:
    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError(f"{name}.vmin must be less than or equal to {name}.vmax.")


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
class BeachingTimesPlottingConfig:
    enabled: bool = False
    vmin: float | None = None
    vmax: float | None = None

    def __post_init__(self) -> None:
        _validate_plot_limits("beaching_times.plotting", self.vmin, self.vmax)


@dataclass(frozen=True)
class BeachingTimesConfig:
    lon_col: str = "lon0"
    lat_col: str = "lat0"
    value_col: str = "lifetime_seconds"
    statistic: str = "min"
    plotting: BeachingTimesPlottingConfig = field(default_factory=BeachingTimesPlottingConfig)


@dataclass(frozen=True)
class FSLEConfig:
    pair_mode: str = "center_pairs"
    meridional_only: bool = False
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
class ExponentMapPlotConfig:
    enabled: bool = False
    average_on_time: bool = True
    vmin: float | None = None
    vmax: float | None = None
    min_mask_value: float | None = None
    log_scale: bool = False
    cmap: str = "viridis"

    def __post_init__(self) -> None:
        _validate_plot_limits("exponent_maps.plot", self.vmin, self.vmax)

        if self.min_mask_value is not None and self.log_scale and self.min_mask_value <= 0:
            raise ValueError(
                "exponent_maps plot min_mask_value must be > 0 when log_scale is enabled."
            )


@dataclass(frozen=True)
class ExponentMapsFSLEConfig:
    enabled: bool = False
    scales_km: tuple[float, ...] = ()
    mask_zeros: bool = False
    plot: ExponentMapPlotConfig = field(default_factory=ExponentMapPlotConfig)

    def __post_init__(self) -> None:
        if any(scale <= 0 for scale in self.scales_km):
            raise ValueError("exponent_maps.fsle.scales_km must contain only positive values.")


@dataclass(frozen=True)
class ExponentMapsFTLEConfig:
    enabled: bool = False
    scales_days: tuple[float, ...] = ()
    sampling_mode: str = "last_before_or_at"
    mask_short_windows: bool = True
    mask_zeros: bool = False
    plot: ExponentMapPlotConfig = field(default_factory=ExponentMapPlotConfig)

    def __post_init__(self) -> None:
        if any(scale <= 0 for scale in self.scales_days):
            raise ValueError("exponent_maps.ftle.scales_days must contain only positive values.")

        if self.sampling_mode not in {"last_before_or_at", "max_within_window"}:
            raise ValueError(
                "exponent_maps.ftle.sampling_mode must be 'last_before_or_at' or 'max_within_window'."
            )


@dataclass(frozen=True)
class ExponentMapsConfig:
    distance: str = "geodesical"
    require_grouped_regular_grid: bool = True
    fsle: ExponentMapsFSLEConfig = field(default_factory=ExponentMapsFSLEConfig)
    ftle: ExponentMapsFTLEConfig = field(default_factory=ExponentMapsFTLEConfig)

    def __post_init__(self) -> None:
        if self.distance not in {"geodesical", "meridional"}:
            raise ValueError(
                "exponent_maps.distance must be 'geodesical' or 'meridional'."
            )

        if self.fsle.enabled and len(self.fsle.scales_km) == 0:
            raise ValueError("exponent_maps.fsle.scales_km cannot be empty when FSLE is enabled.")

        if self.ftle.enabled and len(self.ftle.scales_days) == 0:
            raise ValueError("exponent_maps.ftle.scales_days cannot be empty when FTLE is enabled.")

        if not self.fsle.enabled and not self.ftle.enabled:
            raise ValueError("At least one of exponent_maps.fsle or exponent_maps.ftle must be enabled.")


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
    animation_color_by: str = "group_member"
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

    # Discrete start/end map styling
    discrete_cmap: str | None = None
    colorbar_label_mode: str = "numeric"  # "numeric" | "region_label" | "region_name"
    show_region_labels: bool = False


@dataclass(frozen=True)
class TransitionProbabilityConfig:
    @dataclass(frozen=True)
    class PlottingConfig:
        enabled: bool = False
        x_log_scale: bool = False
        y_log_scale: bool = False
        colormap: str | None = None
        x_limit_min: float | None = None
        x_limit_max: float | None = None

    region_labels: tuple[str, ...] = ()
    time_step_stride: int = 1
    how_many: str = "priority_max"
    priority_level: int | None = None
    priority_mode: str = "exact"
    input_lon_mode: str = "-180_180"
    min_life_days: float = 0.0
    trimming_age_days: float | None = None
    max_group_member: int | None = None
    filter_isolated: bool = False
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    def __post_init__(self) -> None:
        if len(self.region_labels) == 0:
            raise ValueError("transition_probability.region_labels cannot be empty.")

        if self.time_step_stride < 1:
            raise ValueError("transition_probability.time_step_stride must be >= 1.")

        if self.how_many not in {"first", "last", "all", "priority_min", "priority_max"}:
            raise ValueError(
                "transition_probability.how_many must be one of: "
                "'first', 'last', 'all', 'priority_min', 'priority_max'."
            )

        if self.priority_mode not in {"exact", "atleast", "atmost"}:
            raise ValueError(
                "transition_probability.priority_mode must be one of: "
                "'exact', 'atleast', 'atmost'."
            )

        if self.input_lon_mode not in {"-180_180", "0_360"}:
            raise ValueError(
                "transition_probability.input_lon_mode must be '-180_180' or '0_360'."
            )

        if self.min_life_days < 0:
            raise ValueError("transition_probability.min_life_days must be >= 0.")

        if self.trimming_age_days is not None and self.trimming_age_days < 0:
            raise ValueError("transition_probability.trimming_age_days must be >= 0 or null.")

        if self.max_group_member is not None and self.max_group_member <= 0:
            raise ValueError(
                "transition_probability.max_group_member must be an integer > 0 or null."
            )

        if self.plotting.x_limit_min is not None and self.plotting.x_limit_min < 0:
            raise ValueError(
                "transition_probability.plotting.x_limit_min must be >= 0 or null."
            )

        if self.plotting.x_limit_max is not None and self.plotting.x_limit_max < 0:
            raise ValueError(
                "transition_probability.plotting.x_limit_max must be >= 0 or null."
            )

        if (
            self.plotting.x_limit_min is not None
            and self.plotting.x_limit_max is not None
            and self.plotting.x_limit_min >= self.plotting.x_limit_max
        ):
            raise ValueError(
                "transition_probability.plotting.x_limit_min must be smaller than "
                "transition_probability.plotting.x_limit_max."
            )

        if self.plotting.x_log_scale:
            if self.plotting.x_limit_min is not None and self.plotting.x_limit_min <= 0:
                raise ValueError(
                    "transition_probability.plotting.x_limit_min must be > 0 when "
                    "transition_probability.plotting.x_log_scale is true."
                )
            if self.plotting.x_limit_max is not None and self.plotting.x_limit_max <= 0:
                raise ValueError(
                    "transition_probability.plotting.x_limit_max must be > 0 when "
                    "transition_probability.plotting.x_log_scale is true."
                )


@dataclass(frozen=True)
class MeridionalCrossingSegmentationConfig:
    lat_filter: str = "rolling_mean"
    filter_window: int = 5
    direction_threshold_deg: float | str = "auto"
    min_segment_duration_days: float = 1.5
    min_segment_displacement_deg: float | str = "auto"
    valid_if: str = "duration_or_displacement"

    def __post_init__(self) -> None:
        if self.lat_filter not in {"rolling_mean", "rolling_median", "none"}:
            raise ValueError(
                "meridional_crossing.segmentation.lat_filter must be one of: "
                "'rolling_mean', 'rolling_median', 'none'."
            )

        if self.filter_window < 1:
            raise ValueError("meridional_crossing.segmentation.filter_window must be >= 1.")

        if isinstance(self.direction_threshold_deg, str):
            if self.direction_threshold_deg != "auto":
                raise ValueError(
                    "meridional_crossing.segmentation.direction_threshold_deg must be numeric or 'auto'."
                )
        elif self.direction_threshold_deg < 0:
            raise ValueError(
                "meridional_crossing.segmentation.direction_threshold_deg must be >= 0."
            )

        if self.min_segment_duration_days < 0:
            raise ValueError(
                "meridional_crossing.segmentation.min_segment_duration_days must be >= 0."
            )

        if isinstance(self.min_segment_displacement_deg, str):
            if self.min_segment_displacement_deg != "auto":
                raise ValueError(
                    "meridional_crossing.segmentation.min_segment_displacement_deg must be numeric or 'auto'."
                )
        elif self.min_segment_displacement_deg < 0:
            raise ValueError(
                "meridional_crossing.segmentation.min_segment_displacement_deg must be >= 0."
            )

        if self.valid_if != "duration_or_displacement":
            raise ValueError(
                "meridional_crossing.segmentation.valid_if must be 'duration_or_displacement'."
            )


@dataclass(frozen=True)
class MeridionalCrossingCrossingConfig:
    crossing_latitude_reference: str = "center"
    count_once_per_segment_per_lat_bin: bool = True

    def __post_init__(self) -> None:
        if self.crossing_latitude_reference not in {"center", "edge"}:
            raise ValueError(
                "meridional_crossing.crossing.crossing_latitude_reference must be "
                "'center' or 'edge'."
            )


@dataclass(frozen=True)
class MeridionalCrossingOutputConfig:
    save_netcdf: bool = True
    save_grid_table: bool = True
    save_figures: bool = True


@dataclass(frozen=True)
class MeridionalCrossingMapPlottingConfig:
    enabled: bool = True
    vmin: float | None = None
    vmax: float | None = None
    as_percent: bool = False

    def __post_init__(self) -> None:
        _validate_plot_limits("meridional_crossing.plotting", self.vmin, self.vmax)


@dataclass(frozen=True)
class MeridionalCrossingPlottingConfig:
    enabled: bool = True
    probability: MeridionalCrossingMapPlottingConfig = field(
        default_factory=MeridionalCrossingMapPlottingConfig
    )
    count: MeridionalCrossingMapPlottingConfig = field(
        default_factory=lambda: MeridionalCrossingMapPlottingConfig(enabled=False)
    )


@dataclass(frozen=True)
class MeridionalCrossingConfig:
    direction: str = "both"
    segmentation: MeridionalCrossingSegmentationConfig = field(
        default_factory=MeridionalCrossingSegmentationConfig
    )
    crossing: MeridionalCrossingCrossingConfig = field(
        default_factory=MeridionalCrossingCrossingConfig
    )
    output: MeridionalCrossingOutputConfig = field(
        default_factory=MeridionalCrossingOutputConfig
    )
    plotting: MeridionalCrossingPlottingConfig = field(
        default_factory=MeridionalCrossingPlottingConfig
    )

    def __post_init__(self) -> None:
        if self.direction not in {"northward", "southward", "both"}:
            raise ValueError(
                "meridional_crossing.direction must be one of: 'northward', 'southward', 'both'."
            )


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
    title_fontsize: int | None = None
    colorbar_fontsize: int | None = None
    colorbar_tick_fontsize: int | None = None
    axis_tick_fontsize: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("title_fontsize", self.title_fontsize),
            ("colorbar_fontsize", self.colorbar_fontsize),
            ("colorbar_tick_fontsize", self.colorbar_tick_fontsize),
            ("axis_tick_fontsize", self.axis_tick_fontsize),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(
                    f"plotting.{name} must be a non-negative integer or null."
                )


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
    exponent_maps: ExponentMapsConfig | None = None
    trajectories: TrajectoriesConfig = field(default_factory=TrajectoriesConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    start_end_regions: StartEndRegionsConfig = field(default_factory=StartEndRegionsConfig)
    transition_probability: TransitionProbabilityConfig = field(
        default_factory=lambda: TransitionProbabilityConfig(region_labels=("sesc-mod", "sesc-sir"))
    )
    meridional_crossing: MeridionalCrossingConfig = field(default_factory=MeridionalCrossingConfig)


@dataclass(frozen=True)
class ParcelsSchema:
    trajectory_dim: str
    obs_dim: str
    time_var: str
    lon_var: str
    lat_var: str
    z_var: str | None = None