from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    AnalysisConfig,
    CleaningConfig,
    DatasetConfig,
    DatasetCoordinatesConfig,
    DensityConfig,
    ExportsConfig,
    GridConfig,
    OutputConfig,
    PostprocessConfig,
    BeachingTimesConfig,
    TrajectoriesConfig,
    PlottingConfig,
    StartEndRegionsConfig,
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
        animate=animate,
        animation_var=animation_var,
        animation_label=animation_label,
        animation_fps=animation_fps,
        animation_vmin=animation_vmin,
        animation_vmax=animation_vmax,
        show_time_bar=show_time_bar,
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
    plot = bool(section.get("plot", False))
    return BeachingTimesConfig(
        lon_col=lon_col,
        lat_col=lat_col,
        value_col=value_col,
        statistic=statistic,
        plot=plot,
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

    animate = bool(section.get("animate", False))

    animation_fps_raw = section.get("animation_fps", 8)
    if not isinstance(animation_fps_raw, int) or animation_fps_raw <= 0:
        raise ValueError("'trajectories.animation_fps' must be an integer > 0.")
    animation_fps = animation_fps_raw

    animation_color_by = _require_nonempty_string(
        section.get("animation_color_by", "lat0"),
        "trajectories.animation_color_by",
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

    return TrajectoriesConfig(
        plot=plot,
        title=title,
        show_start=show_start,
        show_end=show_end,
        animate=animate,
        animation_fps=animation_fps,
        animation_color_by=animation_color_by,
        animation_vmin=animation_vmin,
        animation_vmax=animation_vmax,
        animation_label=animation_label,
        show_time_bar=show_time_bar,
        trail=trail,
        trail_steps=trail_steps,
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
    density = _parse_density(raw.get("density"))
    cleaning = _parse_cleaning(raw.get("cleaning"))
    beaching_times = _parse_beaching_times(raw.get("beaching_times"))
    trajectories = _parse_trajectories(raw.get("trajectories"))
    plotting = _parse_plotting(raw.get("plotting"))
    start_end_regions = _parse_start_end_regions(raw.get("start_end_regions"))

    return PostprocessConfig(
        dataset=dataset,
        analysis=analysis,
        output=output,
        exports=exports,
        grid=grid,
        density=density,
        cleaning=cleaning,
        beaching_times=beaching_times,
        trajectories=trajectories,
        plotting=plotting,
        start_end_regions=start_end_regions
    )