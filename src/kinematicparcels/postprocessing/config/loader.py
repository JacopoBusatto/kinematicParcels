from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    AnalysisConfig,
    CleaningConfig,
    DatasetConfig,
    DatasetCoordinatesConfig,
    BeachingTimesPlottingConfig,
    DensityConfig,
    ExponentMapPlotConfig,
    ExponentMapsConfig,
    ExponentMapsFSLEConfig,
    ExponentMapsFTLEConfig,
    ExportsConfig,
    FSLEConfig,
    GridConfig,
    OutputConfig,
    PostprocessConfig,
    BeachingTimesConfig,
    ReleaseConfig,
    TrajectoriesConfig,
    PlottingConfig,
    StartEndRegionsConfig,
    TransitionProbabilityConfig,
    MeridionalCrossingConfig,
    MeridionalCrossingCrossingConfig,
    MeridionalCrossingMapPlottingConfig,
    MeridionalCrossingOutputConfig,
    MeridionalCrossingPlottingConfig,
    MeridionalCrossingSegmentationConfig,
)


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    """
    Ensure that a YAML section is a dictionary.
    """
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping/dictionary.")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    """
    Ensure that a config field is a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _require_number(value: Any, name: str) -> float:
    """
    Ensure that a config field is numeric.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number.")
    return float(value)


def _parse_optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _require_number(value, name)


def _parse_coordinates(section: dict[str, Any] | None) -> DatasetCoordinatesConfig:
    """
    Parse the optional dataset.coordinates section.

    If the section is missing, use Parcels default names.
    """
    if section is None:
        return DatasetCoordinatesConfig()

    section = _require_dict(section, "dataset.coordinates")

    trajectory = _require_nonempty_string(
        section.get("trajectory", "trajectory"),
        "dataset.coordinates.trajectory",
    )
    obs = _require_nonempty_string(
        section.get("obs", "obs"),
        "dataset.coordinates.obs",
    )
    time = _require_nonempty_string(
        section.get("time", "time"),
        "dataset.coordinates.time",
    )
    lon = _require_nonempty_string(
        section.get("lon", "lon"),
        "dataset.coordinates.lon",
    )
    lat = _require_nonempty_string(
        section.get("lat", "lat"),
        "dataset.coordinates.lat",
    )

    z_raw = section.get("z", "z")
    if z_raw is None:
        z = None
    else:
        z = _require_nonempty_string(z_raw, "dataset.coordinates.z")

    return DatasetCoordinatesConfig(
        trajectory=trajectory,
        obs=obs,
        time=time,
        lon=lon,
        lat=lat,
        z=z,
    )


def _parse_dataset(section: dict[str, Any] | None) -> DatasetConfig:
    """
    Parse the required dataset section.
    """
    section = _require_dict(section, "dataset")

    input_path = _require_nonempty_string(
        section.get("input_path"),
        "dataset.input_path",
    )

    coordinates = _parse_coordinates(section.get("coordinates"))

    return DatasetConfig(
        input_path=input_path,
        coordinates=coordinates,
    )


def _parse_analysis(section: dict[str, Any] | None) -> AnalysisConfig:
    """
    Parse analysis configuration.
    """
    if section is None:
        return AnalysisConfig()

    section = _require_dict(section, "analysis")

    raw_types = section.get("types")

    if raw_types is None:
        raise ValueError("analysis.types must be provided.")

    if not isinstance(raw_types, list):
        raise ValueError("analysis.types must be a list.")

    types: list[str] = []

    for i, item in enumerate(raw_types):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"analysis.types[{i}] must be a non-empty string.")
        types.append(item.strip())

    if len(types) == 0:
        raise ValueError("analysis.types cannot be empty.")

    return AnalysisConfig(types=tuple(types))


def _parse_output(section: dict[str, Any] | None) -> OutputConfig:
    """
    Parse the optional output section.
    """
    if section is None:
        return OutputConfig()

    section = _require_dict(section, "output")

    output_dir = _require_nonempty_string(
        section.get("output_dir", "./outputs/postprocessing"),
        "output.output_dir",
    )

    return OutputConfig(output_dir=output_dir)


def _parse_exports(section: dict[str, Any] | None) -> ExportsConfig:
    """
    Parse the optional exports section.
    """
    if section is None:
        return ExportsConfig()

    section = _require_dict(section, "exports")

    save_trajectory_table = bool(section.get("save_trajectory_table", False))
    save_particle_summary = bool(section.get("save_particle_summary", True))
    table_format = _require_nonempty_string(
        section.get("table_format", "parquet"),
        "exports.table_format",
    )

    return ExportsConfig(
        save_trajectory_table=save_trajectory_table,
        save_particle_summary=save_particle_summary,
        table_format=table_format,
    )


def _parse_cleaning(section: dict[str, Any] | None) -> CleaningConfig:
    """
    Parse the optional cleaning section.
    """
    if section is None:
        return CleaningConfig()

    section = _require_dict(section, "cleaning")

    truncate_stagnant = bool(section.get("truncate_stagnant", False))
    stagnant_tol = _require_number(
        section.get("stagnant_tol", 1.0e-6),
        "cleaning.stagnant_tol",
    )
    stagnant_min_consecutive_raw = section.get("stagnant_min_consecutive", 2)
    if not isinstance(stagnant_min_consecutive_raw, int) or stagnant_min_consecutive_raw < 1:
        raise ValueError("'cleaning.stagnant_min_consecutive' must be an integer >= 1.")

    return CleaningConfig(
        truncate_stagnant=truncate_stagnant,
        stagnant_tol=stagnant_tol,
        stagnant_min_consecutive=stagnant_min_consecutive_raw,
    )


def _parse_grid(section: dict[str, Any] | None) -> GridConfig | None:
    """
    Parse the optional grid section.
    """
    if section is None:
        return None

    section = _require_dict(section, "grid")

    mode = _require_nonempty_string(
        section.get("mode", "from_initial_centers"),
        "grid.mode",
    )

    lon_min = _require_number(section.get("lon_min"), "grid.lon_min")
    lon_max = _require_number(section.get("lon_max"), "grid.lon_max")
    lat_min = _require_number(section.get("lat_min"), "grid.lat_min")
    lat_max = _require_number(section.get("lat_max"), "grid.lat_max")
    dlon = _require_number(section.get("dlon"), "grid.dlon")
    dlat = _require_number(section.get("dlat"), "grid.dlat")

    return GridConfig(
        mode=mode,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        dlon=dlon,
        dlat=dlat,
    )


def _parse_density(section: dict[str, Any] | None) -> DensityConfig:
    """
    Parse the optional density section.
    """
    if section is None:
        return DensityConfig()

    section = _require_dict(section, "density")

    lon_col = _require_nonempty_string(
        section.get("lon_col", "lon"),
        "density.lon_col",
    )
    lat_col = _require_nonempty_string(
        section.get("lat_col", "lat"),
        "density.lat_col",
    )
    time_col = _require_nonempty_string(
        section.get("time_col", "time"),
        "density.time_col",
    )

    normalize_active = bool(section.get("normalize_active", True))
    normalize_total = bool(section.get("normalize_total", True))
    fill_ever_active_empty_with_zero = bool(section.get("fill_ever_active_empty_with_zero", False))

    group_member_raw = section.get("group_member", None)
    if group_member_raw is None:
        group_member = None
    else:
        if not isinstance(group_member_raw, int) or group_member_raw <= 0:
            raise ValueError("'density.group_member' must be an integer > 0 or null.")
        group_member = group_member_raw

    animate = bool(section.get("animate", False))
    animation_var = _require_nonempty_string(
        section.get("animation_var", "particle_count"),
        "density.animation_var",
    )
    animation_label = _require_nonempty_string(
        section.get("animation_label", "particle_count"),
        "density.animation_label",
    )

    animation_fps_raw = section.get("animation_fps", 8)
    if not isinstance(animation_fps_raw, int) or animation_fps_raw <= 0:
        raise ValueError("'density.animation_fps' must be an integer > 0.")
    animation_fps = animation_fps_raw

    animation_every_n_raw = section.get("animation_every_n", 1)
    if not isinstance(animation_every_n_raw, int) or animation_every_n_raw < 1:
        raise ValueError("'density.animation_every_n' must be an integer >= 1.")
    animation_every_n_density = animation_every_n_raw

    animation_vmin_raw = section.get("animation_vmin", None)
    if animation_vmin_raw is None:
        animation_vmin = None
    else:
        animation_vmin = _require_number(animation_vmin_raw, "density.animation_vmin")

    animation_vmax_raw = section.get("animation_vmax", None)
    if animation_vmax_raw is None:
        animation_vmax = None
    else:
        animation_vmax = _require_number(animation_vmax_raw, "density.animation_vmax")

    show_time_bar = bool(section.get("show_time_bar", True))

    return DensityConfig(
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
        normalize_active=normalize_active,
        normalize_total=normalize_total,
        group_member=group_member,
        animate=animate,
        animation_var=animation_var,
        animation_label=animation_label,
        animation_fps=animation_fps,
        animation_every_n=animation_every_n_density,
        animation_vmin=animation_vmin,
        animation_vmax=animation_vmax,
        show_time_bar=show_time_bar,
        fill_ever_active_empty_with_zero=fill_ever_active_empty_with_zero
    )


def _parse_beaching_times(section: dict[str, Any] | None) -> BeachingTimesConfig:
    """
    Parse the optional beaching_times section.
    """
    if section is None:
        return BeachingTimesConfig()

    section = _require_dict(section, "beaching_times")

    lon_col = _require_nonempty_string(
        section.get("lon_col", "lon0"),
        "beaching_times.lon_col",
    )
    lat_col = _require_nonempty_string(
        section.get("lat_col", "lat0"),
        "beaching_times.lat_col",
    )
    value_col = _require_nonempty_string(
        section.get("value_col", "lifetime_seconds"),
        "beaching_times.value_col",
    )
    statistic = _require_nonempty_string(
        section.get("statistic", "min"),
        "beaching_times.statistic",
    )

    plotting_section = section.get("plotting", None)
    if plotting_section is None:
        plotting = BeachingTimesPlottingConfig(
            enabled=bool(section.get("plot", False)),
        )
    else:
        plotting_section = _require_dict(plotting_section, "beaching_times.plotting")
        plotting = BeachingTimesPlottingConfig(
            enabled=bool(plotting_section.get("enabled", section.get("plot", False))),
            vmin=_parse_optional_number(
                plotting_section.get("vmin", None),
                "beaching_times.plotting.vmin",
            ),
            vmax=_parse_optional_number(
                plotting_section.get("vmax", None),
                "beaching_times.plotting.vmax",
            ),
        )

    return BeachingTimesConfig(
        lon_col=lon_col,
        lat_col=lat_col,
        value_col=value_col,
        statistic=statistic,
        plotting=plotting,
    )


def _parse_fsle(section: dict[str, Any] | None) -> FSLEConfig:
    """
    Parse the optional fsle section.
    """
    if section is None:
        return FSLEConfig()

    section = _require_dict(section, "fsle")

    pair_mode = _require_nonempty_string(
        section.get("pair_mode", "center_pairs"),
        "fsle.pair_mode",
    )
    meridional_only = bool(section.get("meridional_only", False))
    min_scale = _require_number(section.get("min_scale", 5.0e-3), "fsle.min_scale")
    max_scale = _require_number(section.get("max_scale", 1.0e4), "fsle.max_scale")
    rho_increment = _require_number(
        section.get("rho_increment", 2 ** 0.5),
        "fsle.rho_increment",
    )
    save_crossing_events = bool(section.get("save_crossing_events", False))
    plot = bool(section.get("plot", True))

    reference_slopes_raw = section.get(
        "reference_slopes",
        ["delta^-2/3", "delta^-1", "delta^-2"],
    )
    if not isinstance(reference_slopes_raw, list):
        raise ValueError("'fsle.reference_slopes' must be a list.")
    reference_slopes: list[str] = []
    for i, item in enumerate(reference_slopes_raw):
        reference_slopes.append(
            _require_nonempty_string(item, f"fsle.reference_slopes[{i}]")
        )

    reference_anchor_scales_raw = section.get("reference_slope_anchor_scales", {})
    if not isinstance(reference_anchor_scales_raw, dict):
        raise ValueError("'fsle.reference_slope_anchor_scales' must be a mapping/dictionary.")
    reference_slope_anchor_scales: dict[str, float] = {}
    for raw_key, raw_value in reference_anchor_scales_raw.items():
        key = _require_nonempty_string(raw_key, "fsle.reference_slope_anchor_scales key")
        reference_slope_anchor_scales[key] = _require_number(
            raw_value,
            f"fsle.reference_slope_anchor_scales[{key}]",
        )

    x_min_raw = section.get("x_min", 1.0)
    x_min = None if x_min_raw is None else _require_number(x_min_raw, "fsle.x_min")

    x_max_raw = section.get("x_max", 2500.0)
    x_max = None if x_max_raw is None else _require_number(x_max_raw, "fsle.x_max")

    y_min_raw = section.get("y_min", 1.0e-2)
    y_min = None if y_min_raw is None else _require_number(y_min_raw, "fsle.y_min")

    y_max_raw = section.get("y_max", 1.0)
    y_max = None if y_max_raw is None else _require_number(y_max_raw, "fsle.y_max")

    return FSLEConfig(
        pair_mode=pair_mode,
        meridional_only=meridional_only,
        min_scale=min_scale,
        max_scale=max_scale,
        rho_increment=rho_increment,
        save_crossing_events=save_crossing_events,
        plot=plot,
        reference_slopes=tuple(reference_slopes),
        reference_slope_anchor_scales=reference_slope_anchor_scales,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def _parse_exponent_map_plot(
    section: dict[str, Any] | None,
    *,
    name: str,
) -> ExponentMapPlotConfig:
    if section is None:
        return ExponentMapPlotConfig()

    section = _require_dict(section, name)

    cmap = _require_nonempty_string(
        section.get("cmap", "viridis"),
        f"{name}.cmap",
    )

    return ExponentMapPlotConfig(
        enabled=bool(section.get("enable", section.get("enabled", False))),
        average_on_time=bool(section.get("average_on_time", True)),
        vmin=_parse_optional_number(section.get("vmin", None), f"{name}.vmin"),
        vmax=_parse_optional_number(section.get("vmax", None), f"{name}.vmax"),
        min_mask_value=_parse_optional_number(
            section.get("min_mask_value", None),
            f"{name}.min_mask_value",
        ),
        log_scale=bool(section.get("log_scale", False)),
        cmap=cmap,
    )


def _parse_positive_number_list(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        return ()

    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a list.")

    parsed: list[float] = []
    for i, item in enumerate(value):
        parsed.append(_require_number(item, f"{name}[{i}]"))
    return tuple(parsed)


def _parse_exponent_maps(section: dict[str, Any] | None) -> ExponentMapsConfig | None:
    if section is None:
        return None

    section = _require_dict(section, "exponent_maps")

    distance = _require_nonempty_string(
        section.get("distance", "geodesical"),
        "exponent_maps.distance",
    )
    require_grouped_regular_grid = bool(section.get("require_grouped_regular_grid", True))

    fsle_section = _require_dict(section.get("fsle", {}), "exponent_maps.fsle")
    ftle_section = _require_dict(section.get("ftle", {}), "exponent_maps.ftle")

    fsle = ExponentMapsFSLEConfig(
        enabled=bool(fsle_section.get("enable", fsle_section.get("enabled", False))),
        scales_km=_parse_positive_number_list(
            fsle_section.get("scale", fsle_section.get("scales_km", [])),
            "exponent_maps.fsle.scale",
        ),
        mask_zeros=bool(fsle_section.get("mask_zeros", False)),
        plot=_parse_exponent_map_plot(
            fsle_section.get("plot", None),
            name="exponent_maps.fsle.plot",
        ),
    )

    ftle = ExponentMapsFTLEConfig(
        enabled=bool(ftle_section.get("enable", ftle_section.get("enabled", False))),
        scales_days=_parse_positive_number_list(
            ftle_section.get("scale", ftle_section.get("scales_days", [])),
            "exponent_maps.ftle.scale",
        ),
        sampling_mode=_require_nonempty_string(
            ftle_section.get("sampling_mode", "last_before_or_at"),
            "exponent_maps.ftle.sampling_mode",
        ),
        mask_short_windows=bool(ftle_section.get("mask_short_windows", True)),
        mask_zeros=bool(ftle_section.get("mask_zeros", False)),
        plot=_parse_exponent_map_plot(
            ftle_section.get("plot", None),
            name="exponent_maps.ftle.plot",
        ),
    )

    return ExponentMapsConfig(
        distance=distance,
        require_grouped_regular_grid=require_grouped_regular_grid,
        fsle=fsle,
        ftle=ftle,
    )


def _parse_trajectories(section: dict[str, Any] | None) -> TrajectoriesConfig:
    """
    Parse the optional trajectories section.
    """
    if section is None:
        return TrajectoriesConfig()

    section = _require_dict(section, "trajectories")

    plot = bool(section.get("plot", True))
    title = _require_nonempty_string(
        section.get("title", "Trajectories"),
        "trajectories.title",
    )
    show_start = bool(section.get("show_start", True))
    show_end = bool(section.get("show_end", True))

    plot_color_by_raw = section.get("plot_color_by", None)
    if plot_color_by_raw is None:
        plot_color_by = None
    else:
        plot_color_by = _require_nonempty_string(plot_color_by_raw, "trajectories.plot_color_by")

    plot_cmap_raw = section.get("plot_cmap", None)
    if plot_cmap_raw is None:
        plot_cmap = None
    else:
        plot_cmap = _require_nonempty_string(plot_cmap_raw, "trajectories.plot_cmap")

    plot_cmap_mode = str(section.get("plot_cmap_mode", "auto"))
    if plot_cmap_mode not in ("auto", "categorical", "numeric"):
        raise ValueError(
            "'trajectories.plot_cmap_mode' must be 'auto', 'categorical', or 'numeric'."
        )

    alpha_raw = section.get("alpha", 0.7)
    alpha = _require_number(alpha_raw, "trajectories.alpha")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("'trajectories.alpha' must be between 0 and 1.")

    animate = bool(section.get("animate", False))

    animation_fps_raw = section.get("animation_fps", 8)
    if not isinstance(animation_fps_raw, int) or animation_fps_raw <= 0:
        raise ValueError("'trajectories.animation_fps' must be an integer > 0.")
    animation_fps = animation_fps_raw

    animation_every_n_raw = section.get("animation_every_n", 1)
    if not isinstance(animation_every_n_raw, int) or animation_every_n_raw < 1:
        raise ValueError("'trajectories.animation_every_n' must be an integer >= 1.")
    animation_every_n_traj = animation_every_n_raw

    animation_color_by = _require_nonempty_string(
        section.get("animation_color_by", "lat0"),
        "trajectories.animation_color_by",
    )

    animation_cmap_raw = section.get("animation_cmap", None)
    if animation_cmap_raw is None:
        animation_cmap = None
    else:
        animation_cmap = _require_nonempty_string(animation_cmap_raw, "trajectories.animation_cmap")

    animation_cmap_mode = str(section.get("animation_cmap_mode", "auto"))
    if animation_cmap_mode not in ("auto", "categorical", "numeric"):
        raise ValueError(
            "'trajectories.animation_cmap_mode' must be 'auto', 'categorical', or 'numeric'."
        )

    animation_vmin_raw = section.get("animation_vmin", None)
    if animation_vmin_raw is None:
        animation_vmin = None
    else:
        animation_vmin = _require_number(
            animation_vmin_raw,
            "trajectories.animation_vmin",
        )

    animation_vmax_raw = section.get("animation_vmax", None)
    if animation_vmax_raw is None:
        animation_vmax = None
    else:
        animation_vmax = _require_number(
            animation_vmax_raw,
            "trajectories.animation_vmax",
        )

    animation_label = _require_nonempty_string(
        section.get("animation_label", "value"),
        "trajectories.animation_label",
    )

    show_time_bar = bool(section.get("show_time_bar", True))
    trail = bool(section.get("trail", True))

    trail_steps_raw = section.get("trail_steps", None)
    if trail_steps_raw is None:
        trail_steps = None
    else:
        if not isinstance(trail_steps_raw, int) or trail_steps_raw <= 0:
            raise ValueError("'trajectories.trail_steps' must be an integer > 0 or null.")
        trail_steps = trail_steps_raw

    max_group_member_raw = section.get("max_group_member", None)
    if max_group_member_raw is None:
        max_group_member = None
    else:
        if not isinstance(max_group_member_raw, int) or max_group_member_raw <= 0:
            raise ValueError("'trajectories.max_group_member' must be an integer > 0 or null.")
        max_group_member = max_group_member_raw

    return TrajectoriesConfig(
        plot=plot,
        title=title,
        show_start=show_start,
        show_end=show_end,
        alpha=alpha,
        plot_color_by=plot_color_by,
        plot_cmap=plot_cmap,
        plot_cmap_mode=plot_cmap_mode,
        animate=animate,
        animation_fps=animation_fps,
        animation_every_n=animation_every_n_traj,
        animation_color_by=animation_color_by,
        animation_cmap=animation_cmap,
        animation_cmap_mode=animation_cmap_mode,
        animation_vmin=animation_vmin,
        animation_vmax=animation_vmax,
        animation_label=animation_label,
        show_time_bar=show_time_bar,
        trail=trail,
        trail_steps=trail_steps,
        max_group_member=max_group_member,
    )


def _parse_plotting(section: dict[str, Any] | None) -> PlottingConfig:
    if section is None:
        return PlottingConfig()

    section = _require_dict(section, "plotting")

    projection = _require_nonempty_string(
        section.get("projection", "PlateCarree"),
        "plotting.projection",
    )

    return PlottingConfig(projection=projection)


def _parse_release(section: dict[str, Any] | None) -> ReleaseConfig:
    """
    Parse the optional release section.
    """
    if section is None:
        return ReleaseConfig()

    section = _require_dict(section, "release")

    mode = _require_nonempty_string(
        section.get("mode", "region_grid"),
        "release.mode",
    )
    continuous = bool(section.get("continuous", False))

    return ReleaseConfig(mode=mode, continuous=continuous)


def _parse_start_end_regions(section: dict[str, Any] | None) -> StartEndRegionsConfig:
    """
    Parse the optional start_end_regions section.
    """
    if section is None:
        return StartEndRegionsConfig()

    section = _require_dict(section, "start_end_regions")

    region_labels_raw = section.get("region_labels", None)
    region_labels: tuple[str, ...] | None

    if region_labels_raw is None:
        region_labels = None
    else:
        if not isinstance(region_labels_raw, list):
            raise ValueError("'start_end_regions.region_labels' must be a list or null.")
        parsed_labels: list[str] = []
        for i, item in enumerate(region_labels_raw):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"start_end_regions.region_labels[{i}] must be a non-empty string."
                )
            parsed_labels.append(item.strip())
        region_labels = tuple(parsed_labels)

    how_many = _require_nonempty_string(
        section.get("how_many", "priority_max"),
        "start_end_regions.how_many",
    )
    priority_mode = _require_nonempty_string(
        section.get("priority_mode", "exact"),
        "start_end_regions.priority_mode",
    )
    input_lon_mode = _require_nonempty_string(
        section.get("input_lon_mode", "-180_180"),
        "start_end_regions.input_lon_mode",
    )
    plot = bool(section.get("plot", False))
    plot_connectivity = bool(section.get("plot_connectivity", False))
    animate_connectivity = bool(section.get("animate_connectivity", False))
    connectivity_segments = bool(section.get("connectivity_segments", True))
    connectivity_color_by = _require_nonempty_string(
        section.get("connectivity_color_by", "start_region"),
        "start_end_regions.connectivity_color_by",
    )
    connectivity_label = _require_nonempty_string(
        section.get("connectivity_label", "region"),
        "start_end_regions.connectivity_label",
    )
    connectivity_title = _require_nonempty_string(
        section.get("connectivity_title", "Trajectories by region"),
        "start_end_regions.connectivity_title",
    )
    connectivity_show_start = bool(section.get("connectivity_show_start", True))
    connectivity_show_end = bool(section.get("connectivity_show_end", True))

    connectivity_alpha_raw = section.get("connectivity_alpha", None)
    if connectivity_alpha_raw is None:
        connectivity_alpha = None
    else:
        connectivity_alpha = _require_number(connectivity_alpha_raw, "start_end_regions.connectivity_alpha")
        if not (0.0 <= connectivity_alpha <= 1.0):
            raise ValueError("'start_end_regions.connectivity_alpha' must be between 0 and 1 or null.")

    connectivity_max_group_member_raw = section.get("connectivity_max_group_member", None)
    if connectivity_max_group_member_raw is None:
        connectivity_max_group_member = None
    else:
        if not isinstance(connectivity_max_group_member_raw, int) or connectivity_max_group_member_raw <= 0:
            raise ValueError("'start_end_regions.connectivity_max_group_member' must be an integer > 0 or null.")
        connectivity_max_group_member = connectivity_max_group_member_raw

    connectivity_animation_fps_raw = section.get("connectivity_animation_fps", None)
    if connectivity_animation_fps_raw is None:
        connectivity_animation_fps = None
    else:
        if not isinstance(connectivity_animation_fps_raw, int) or connectivity_animation_fps_raw <= 0:
            raise ValueError("'start_end_regions.connectivity_animation_fps' must be an integer > 0 or null.")
        connectivity_animation_fps = connectivity_animation_fps_raw

    connectivity_animation_show_tracer_raw = section.get("connectivity_animation_show_tracer", None)
    if connectivity_animation_show_tracer_raw is None:
        connectivity_animation_show_tracer = None
    else:
        connectivity_animation_show_tracer = bool(connectivity_animation_show_tracer_raw)

    connectivity_trail_raw = section.get("connectivity_trail", None)
    if connectivity_trail_raw is None:
        connectivity_trail = None
    else:
        connectivity_trail = bool(connectivity_trail_raw)

    connectivity_trail_steps_raw = section.get("connectivity_trail_steps", None)
    if connectivity_trail_steps_raw is None:
        connectivity_trail_steps = None
    else:
        if not isinstance(connectivity_trail_steps_raw, int) or connectivity_trail_steps_raw <= 0:
            raise ValueError("'start_end_regions.connectivity_trail_steps' must be an integer > 0 or null.")
        connectivity_trail_steps = connectivity_trail_steps_raw

    priority_level_raw = section.get("priority_level", None)
    if priority_level_raw is None:
        priority_level = None
    else:
        if not isinstance(priority_level_raw, int):
            raise ValueError("'start_end_regions.priority_level' must be an integer or null.")
        priority_level = priority_level_raw

    return StartEndRegionsConfig(
        region_labels=region_labels,
        how_many=how_many,
        priority_level=priority_level,
        priority_mode=priority_mode,
        input_lon_mode=input_lon_mode,
        plot=plot,
        plot_connectivity=plot_connectivity,
        animate_connectivity=animate_connectivity,
        connectivity_segments=connectivity_segments,
        connectivity_color_by=connectivity_color_by,
        connectivity_label=connectivity_label,
        connectivity_title=connectivity_title,
        connectivity_show_start=connectivity_show_start,
        connectivity_show_end=connectivity_show_end,
        connectivity_alpha=connectivity_alpha,
        connectivity_max_group_member=connectivity_max_group_member,
        connectivity_animation_fps=connectivity_animation_fps,
        connectivity_animation_show_tracer=connectivity_animation_show_tracer,
        connectivity_trail=connectivity_trail,
        connectivity_trail_steps=connectivity_trail_steps,
    )


def _parse_transition_probability(section: dict[str, Any] | None) -> TransitionProbabilityConfig:
    if section is None:
        return TransitionProbabilityConfig(region_labels=("sesc-mod", "sesc-sir"))

    section = _require_dict(section, "transition_probability")

    region_labels_raw = section.get("region_labels", None)
    if not isinstance(region_labels_raw, list) or len(region_labels_raw) == 0:
        raise ValueError("'transition_probability.region_labels' must be a non-empty list.")
    region_labels = tuple(
        _require_nonempty_string(item, f"transition_probability.region_labels[{i}]")
        for i, item in enumerate(region_labels_raw)
    )

    time_step_stride_raw = section.get("time_step_stride", 1)
    if not isinstance(time_step_stride_raw, int) or time_step_stride_raw < 1:
        raise ValueError("'transition_probability.time_step_stride' must be an integer >= 1.")

    how_many = _require_nonempty_string(
        section.get("how_many", "priority_max"),
        "transition_probability.how_many",
    )
    priority_level_raw = section.get("priority_level", None)
    if priority_level_raw is None:
        priority_level = None
    else:
        if not isinstance(priority_level_raw, int):
            raise ValueError("'transition_probability.priority_level' must be an integer or null.")
        priority_level = priority_level_raw

    priority_mode = _require_nonempty_string(
        section.get("priority_mode", "exact"),
        "transition_probability.priority_mode",
    )
    input_lon_mode = _require_nonempty_string(
        section.get("input_lon_mode", "-180_180"),
        "transition_probability.input_lon_mode",
    )
    min_life_days = _require_number(
        section.get("min_life_days", 0),
        "transition_probability.min_life_days",
    )
    trimming_age_days = _parse_optional_number(
        section.get("trimming_age_days", None),
        "transition_probability.trimming_age_days",
    )

    max_group_member_raw = section.get("max_group_member", None)
    if max_group_member_raw is None:
        max_group_member = None
    else:
        if not isinstance(max_group_member_raw, int) or max_group_member_raw <= 0:
            raise ValueError(
                "'transition_probability.max_group_member' must be an integer > 0 or null."
            )
        max_group_member = max_group_member_raw

    filter_isolated = bool(section.get("filter_isolated", False))

    return TransitionProbabilityConfig(
        region_labels=region_labels,
        time_step_stride=time_step_stride_raw,
        how_many=how_many,
        priority_level=priority_level,
        priority_mode=priority_mode,
        input_lon_mode=input_lon_mode,
        min_life_days=min_life_days,
        trimming_age_days=trimming_age_days,
        max_group_member=max_group_member,
        filter_isolated=filter_isolated,
    )


def _parse_meridional_crossing(section: dict[str, Any] | None) -> MeridionalCrossingConfig:
    """
    Parse the optional meridional_crossing section.
    """
    if section is None:
        return MeridionalCrossingConfig()

    section = _require_dict(section, "meridional_crossing")

    direction = _require_nonempty_string(
        section.get("direction", "both"),
        "meridional_crossing.direction",
    )

    segmentation_section = section.get("segmentation", None)
    if segmentation_section is None:
        segmentation = MeridionalCrossingSegmentationConfig()
    else:
        segmentation_section = _require_dict(
            segmentation_section,
            "meridional_crossing.segmentation",
        )
        lat_filter = _require_nonempty_string(
            segmentation_section.get("lat_filter", "rolling_mean"),
            "meridional_crossing.segmentation.lat_filter",
        )
        filter_window_raw = segmentation_section.get("filter_window", 5)
        if not isinstance(filter_window_raw, int) or filter_window_raw < 1:
            raise ValueError(
                "'meridional_crossing.segmentation.filter_window' must be an integer >= 1."
            )

        direction_threshold_raw = segmentation_section.get("direction_threshold_deg", "auto")
        if isinstance(direction_threshold_raw, str):
            direction_threshold_deg: float | str = _require_nonempty_string(
                direction_threshold_raw,
                "meridional_crossing.segmentation.direction_threshold_deg",
            )
        else:
            direction_threshold_deg = _require_number(
                direction_threshold_raw,
                "meridional_crossing.segmentation.direction_threshold_deg",
            )

        min_segment_duration_days = _require_number(
            segmentation_section.get("min_segment_duration_days", 1.5),
            "meridional_crossing.segmentation.min_segment_duration_days",
        )

        min_segment_displacement_raw = segmentation_section.get(
            "min_segment_displacement_deg",
            "auto",
        )
        if isinstance(min_segment_displacement_raw, str):
            min_segment_displacement_deg: float | str = _require_nonempty_string(
                min_segment_displacement_raw,
                "meridional_crossing.segmentation.min_segment_displacement_deg",
            )
        else:
            min_segment_displacement_deg = _require_number(
                min_segment_displacement_raw,
                "meridional_crossing.segmentation.min_segment_displacement_deg",
            )

        valid_if = _require_nonempty_string(
            segmentation_section.get("valid_if", "duration_or_displacement"),
            "meridional_crossing.segmentation.valid_if",
        )

        segmentation = MeridionalCrossingSegmentationConfig(
            lat_filter=lat_filter,
            filter_window=filter_window_raw,
            direction_threshold_deg=direction_threshold_deg,
            min_segment_duration_days=min_segment_duration_days,
            min_segment_displacement_deg=min_segment_displacement_deg,
            valid_if=valid_if,
        )

    crossing_section = section.get("crossing", None)
    if crossing_section is None:
        crossing = MeridionalCrossingCrossingConfig()
    else:
        crossing_section = _require_dict(crossing_section, "meridional_crossing.crossing")
        crossing_latitude_reference = _require_nonempty_string(
            crossing_section.get("crossing_latitude_reference", "center"),
            "meridional_crossing.crossing.crossing_latitude_reference",
        )
        count_once_per_segment_per_lat_bin = bool(
            crossing_section.get("count_once_per_segment_per_lat_bin", True)
        )
        crossing = MeridionalCrossingCrossingConfig(
            crossing_latitude_reference=crossing_latitude_reference,
            count_once_per_segment_per_lat_bin=count_once_per_segment_per_lat_bin,
        )

    output_section = section.get("output", None)
    if output_section is None:
        output = MeridionalCrossingOutputConfig()
    else:
        output_section = _require_dict(output_section, "meridional_crossing.output")
        output = MeridionalCrossingOutputConfig(
            save_netcdf=bool(output_section.get("save_netcdf", True)),
            save_grid_table=bool(output_section.get("save_grid_table", True)),
            save_figures=bool(output_section.get("save_figures", True)),
        )

    plotting_section = section.get("plotting", None)
    if plotting_section is None:
        plotting = MeridionalCrossingPlottingConfig(
            probability=MeridionalCrossingMapPlottingConfig(
                enabled=bool(section.get("show_probability", True))
            ),
            count=MeridionalCrossingMapPlottingConfig(
                enabled=bool(section.get("show_counts", False))
            ),
        )
    else:
        plotting_section = _require_dict(plotting_section, "meridional_crossing.plotting")
        probability_section = plotting_section.get("probability", None)
        if probability_section is None:
            probability = MeridionalCrossingMapPlottingConfig(
                enabled=bool(plotting_section.get("show_probability", True)),
            )
        else:
            probability_section = _require_dict(
                probability_section,
                "meridional_crossing.plotting.probability",
            )
            probability = MeridionalCrossingMapPlottingConfig(
                enabled=bool(
                    probability_section.get(
                        "enabled",
                        plotting_section.get("show_probability", True),
                    )
                ),
                vmin=_parse_optional_number(
                    probability_section.get("vmin", None),
                    "meridional_crossing.plotting.probability.vmin",
                ),
                vmax=_parse_optional_number(
                    probability_section.get("vmax", None),
                    "meridional_crossing.plotting.probability.vmax",
                ),
            )

        count_section = plotting_section.get("count", None)
        if count_section is None:
            count = MeridionalCrossingMapPlottingConfig(
                enabled=bool(plotting_section.get("show_counts", False)),
            )
        else:
            count_section = _require_dict(count_section, "meridional_crossing.plotting.count")
            count = MeridionalCrossingMapPlottingConfig(
                enabled=bool(
                    count_section.get(
                        "enabled",
                        plotting_section.get("show_counts", False),
                    )
                ),
                vmin=_parse_optional_number(
                    count_section.get("vmin", None),
                    "meridional_crossing.plotting.count.vmin",
                ),
                vmax=_parse_optional_number(
                    count_section.get("vmax", None),
                    "meridional_crossing.plotting.count.vmax",
                ),
            )

        plotting = MeridionalCrossingPlottingConfig(
            enabled=bool(plotting_section.get("enabled", True)),
            probability=probability,
            count=count,
        )

    return MeridionalCrossingConfig(
        direction=direction,
        segmentation=segmentation,
        crossing=crossing,
        output=output,
        plotting=plotting,
    )


def load_postprocess_config(path: str | Path) -> PostprocessConfig:
    """
    Load the post-processing YAML config.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Configuration file is empty.")

    raw = _require_dict(raw, "root")

    dataset = _parse_dataset(raw.get("dataset"))
    analysis = _parse_analysis(raw.get("analysis"))
    output = _parse_output(raw.get("output"))
    exports = _parse_exports(raw.get("exports"))
    grid = _parse_grid(raw.get("grid"))
    release = _parse_release(raw.get("release"))
    density = _parse_density(raw.get("density"))
    cleaning = _parse_cleaning(raw.get("cleaning"))
    beaching_times = _parse_beaching_times(raw.get("beaching_times"))
    fsle = _parse_fsle(raw.get("fsle"))
    exponent_maps = _parse_exponent_maps(raw.get("exponent_maps"))
    trajectories = _parse_trajectories(raw.get("trajectories"))
    plotting = _parse_plotting(raw.get("plotting"))
    start_end_regions = _parse_start_end_regions(raw.get("start_end_regions"))
    transition_probability = _parse_transition_probability(raw.get("transition_probability"))
    meridional_crossing = _parse_meridional_crossing(raw.get("meridional_crossing"))

    return PostprocessConfig(
        dataset=dataset,
        analysis=analysis,
        output=output,
        exports=exports,
        grid=grid,
        release=release,
        density=density,
        cleaning=cleaning,
        beaching_times=beaching_times,
        fsle=fsle,
        exponent_maps=exponent_maps,
        trajectories=trajectories,
        plotting=plotting,
        start_end_regions=start_end_regions,
        transition_probability=transition_probability,
        meridional_crossing=meridional_crossing,
    )