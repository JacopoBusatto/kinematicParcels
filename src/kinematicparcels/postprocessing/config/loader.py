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
    Parse the optional analysis section.
    """
    if section is None:
        return AnalysisConfig()

    section = _require_dict(section, "analysis")

    analysis_type = _require_nonempty_string(
        section.get("type", "trajectories"),
        "analysis.type",
    )

    return AnalysisConfig(type=analysis_type)


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

    lon_min = _require_number(section.get("lon_min"), "grid.lon_min")
    lon_max = _require_number(section.get("lon_max"), "grid.lon_max")
    lat_min = _require_number(section.get("lat_min"), "grid.lat_min")
    lat_max = _require_number(section.get("lat_max"), "grid.lat_max")
    dlon = _require_number(section.get("dlon"), "grid.dlon")
    dlat = _require_number(section.get("dlat"), "grid.dlat")

    return GridConfig(
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

    return DensityConfig(
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
        normalize_active=normalize_active,
        normalize_total=normalize_total,
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

    return PostprocessConfig(
        dataset=dataset,
        analysis=analysis,
        output=output,
        exports=exports,
        grid=grid,
        density=density,
        cleaning=cleaning,
    )