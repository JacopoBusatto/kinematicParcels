"""Production table and reproducibility-file writing."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import CompactConfig

STANDARD_TABLES = (
    "cell_statistics.parquet",
    "branch_cores.parquet",
    "fronts.parquet",
)
DIRECTIONAL_TABLES = (
    "directional_corridors.parquet",
    "directional_fronts.parquet",
    "structure_comparison.parquet",
    "structure_component_comparison.parquet",
)


_COORDINATE_PREFIXES = (
    "start",
    "end",
    "core",
    "front",
    "candidate",
    "corridor",
    "ridge",
    "refined_core",
    "refined_axis",
    "sample",
    "flank",
    "local_gradient_max",
    "transverse_minus",
    "transverse_plus",
    "transverse_left",
    "transverse_right",
)


def _public_column_name(name: str, config: CompactConfig) -> str:
    """Translate a unit-neutral internal field to its stable public schema."""
    if not any(
        marker in name
        for marker in ("_length", "_rate", "_area_rate", "_rate_per_length")
    ):
        if name in {"S_transport", "U_out_all", "U_out_move"} or name.startswith(
            ("U_parallel_", "U_core", "U_inner", "U_outer")
        ) or (
            any(
                token in name
                for token in (
                    "absolute_drop",
                    "absolute_transport_loss",
                    "outer_recovery",
                    "profile_total_variation",
                )
            )
            and not name.startswith("spearman_")
        ):
            name = f"{name}_rate"
        elif (
            any(
                token in name
                for token in (
                    "dS_dx",
                    "dS_dy",
                    "G_perp",
                    "G_parallel",
                    "gradient_magnitude",
                )
            )
            and not name.startswith("spearman_")
            and not name.endswith(("_ratio", "_fraction", "_percentile"))
        ):
            name = f"{name}_rate_per_length"
    replacements = (
        ("_rate_per_length", f"_{config.rate_gradient_suffix}"),
        ("_area_rate", f"_{config.area_rate_suffix}"),
        ("_rate", f"_{config.rate_suffix}"),
        ("_length", f"_{config.geometry.length_suffix}"),
    )
    for internal, public in replacements:
        name = name.replace(internal, public)

    if config.geometry.coordinate_system != "geographic":
        return name

    # Vector components retain the established east/north vocabulary in
    # geographic output, while Cartesian output uses x/y.
    vector_prefixes = (
        "U_out_",
        "D_out_",
        "mu_out_",
        "mu_in_",
        "moment_identity_",
        "directional_identity_",
    )
    if name.startswith(vector_prefixes):
        name = re.sub(r"(?<=_)x(?=_|$)", "east", name)
        name = re.sub(r"(?<=_)y(?=_|$)", "north", name)

    coordinate_names = {"x": "lon", "y": "lat", "x_bin": "lon_bin", "y_bin": "lat_bin"}
    if name in coordinate_names:
        return coordinate_names[name]
    for prefix in _COORDINATE_PREFIXES:
        if name == f"{prefix}_x":
            return f"{prefix}_lon"
        if name == f"{prefix}_y":
            return f"{prefix}_lat"
        if name == f"{prefix}_x_bin":
            return f"{prefix}_lon_bin"
        if name == f"{prefix}_y_bin":
            return f"{prefix}_lat_bin"
        if name == f"{prefix}_x_center":
            return f"{prefix}_lon_center"
        if name == f"{prefix}_y_center":
            return f"{prefix}_lat_center"
    special = {
        f"grid_x_scale_{config.geometry.length_suffix}": f"grid_zonal_scale_{config.geometry.length_suffix}",
        f"grid_y_scale_{config.geometry.length_suffix}": f"grid_meridional_scale_{config.geometry.length_suffix}",
        "x_span": "longitude_span_degrees",
        "y_span": "latitude_span_degrees",
        "centroid_x": "centroid_lon_circular",
        "centroid_y": "centroid_lat",
    }
    return special.get(name, name)


def externalize_table(table: pd.DataFrame, config: CompactConfig) -> pd.DataFrame:
    """Return a copy with coordinate- and unit-specific public column names."""
    renamed = {name: _public_column_name(str(name), config) for name in table.columns}
    duplicates = [name for name in set(renamed.values()) if list(renamed.values()).count(name) > 1]
    if duplicates:
        raise ValueError(f"public output column collision: {sorted(duplicates)}")
    return table.rename(columns=renamed).copy()


def _externalize_mapping(value: Any, config: CompactConfig) -> Any:
    if isinstance(value, dict):
        return {
            _public_column_name(str(key), config): _externalize_mapping(item, config)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_externalize_mapping(item, config) for item in value]
    if isinstance(value, tuple):
        return [_externalize_mapping(item, config) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(config: CompactConfig) -> Path:
    output_root = Path(config.output.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{config.output.run_name}_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def write_scientific_tables(
    run_dir: Path,
    *,
    cells: pd.DataFrame,
    cores: pd.DataFrame,
    fronts: pd.DataFrame,
    config: CompactConfig,
) -> None:
    for name, table in zip(STANDARD_TABLES, (cells, cores, fronts), strict=True):
        externalize_table(table, config).to_parquet(run_dir / name, index=False)


def write_directional_tables(
    run_dir: Path,
    *,
    corridors: pd.DataFrame,
    fronts: pd.DataFrame,
    comparison: pd.DataFrame,
    component_comparison: pd.DataFrame,
    config: CompactConfig,
) -> None:
    tables = (corridors, fronts, comparison, component_comparison)
    for name, table in zip(DIRECTIONAL_TABLES, tables, strict=True):
        externalize_table(table, config).to_parquet(run_dir / name, index=False)


def write_debug_tables(
    run_dir: Path,
    *,
    candidate_drops: pd.DataFrame,
    cross_sections: pd.DataFrame,
    section_summaries: pd.DataFrame,
    components: pd.DataFrame,
    segment_fronts: pd.DataFrame,
    config: CompactConfig,
) -> list[str]:
    outputs = {
        "candidate_drop_zones.parquet": candidate_drops,
        "raw_cross_sections.parquet": cross_sections,
        "section_composites.parquet": section_summaries,
        "component_graph_details.parquet": components,
        "segment_front_candidates.parquet": segment_fronts,
    }
    for name, table in outputs.items():
        externalize_table(table, config).to_parquet(run_dir / name, index=False)
    return list(outputs)


def write_directional_debug_tables(
    run_dir: Path,
    *,
    candidate_drops: pd.DataFrame,
    cross_sections: pd.DataFrame,
    section_summaries: pd.DataFrame,
    components: pd.DataFrame,
    graph_edges: pd.DataFrame,
    config: CompactConfig,
) -> list[str]:
    outputs = {
        "directional_candidate_drop_zones.parquet": candidate_drops,
        "directional_raw_cross_sections.parquet": cross_sections,
        "directional_section_composites.parquet": section_summaries,
        "directional_corridor_components.parquet": components,
        "directional_corridor_graph_edges.parquet": graph_edges,
    }
    for name, table in outputs.items():
        externalize_table(table, config).to_parquet(run_dir / name, index=False)
    return list(outputs)


def write_validation_table(
    run_dir: Path, validation: pd.DataFrame, config: CompactConfig
) -> str:
    name = "gradient_validation.parquet"
    externalize_table(validation, config).to_parquet(run_dir / name, index=False)
    return name


def write_reproducibility_files(
    run_dir: Path,
    *,
    config: CompactConfig,
    input_path: Path,
    transport_threshold_rate: float,
    counts: dict[str, int],
    transition_validation_summary: dict[str, Any],
    gradient_validation_summary: dict[str, Any] | None,
    directional_corridor_summary: dict[str, Any],
    directional_front_summary: dict[str, Any],
    structure_comparison_summary: dict[str, Any],
    figures: list[Path],
    optional_outputs: list[str],
) -> None:
    config_name = "resolved_config.yaml"
    resolved_config = config.to_dict()
    resolved_config["resolved_geometry"] = config.geometry_metadata
    (run_dir / config_name).write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    inventory = [
        *STANDARD_TABLES,
        *DIRECTIONAL_TABLES,
        config_name,
        "manifest.json",
    ]
    inventory.extend(path.relative_to(run_dir).as_posix() for path in figures)
    inventory.extend(optional_outputs)
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "analysis_version": config.analysis_version,
            "python": platform.python_version(),
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "matrix_id": config.input.matrix_id,
            "timestep": config.input.timestep,
            "time_unit": config.input.time_unit,
            "normalized_source_probability_contract": True,
        },
        "geometry": config.geometry_metadata,
        "selection": {
            "min_moving_support": config.statistics.min_moving_support,
            "transport_percentile": config.branches.transport_percentile,
            "ridge_field": config.branches.ridge_field,
            f"transport_threshold_{config.rate_suffix}": transport_threshold_rate,
            "directional": {
                "minimum_P_move": config.directional.minimum_P_move,
                "minimum_R1": config.directional.minimum_R1,
                "minimum_strength": config.directional.minimum_strength,
                "maximum_neighbor_direction_difference_degrees": (
                    config.directional.maximum_neighbor_direction_difference_degrees
                ),
                "maximum_step_direction_mismatch_degrees": (
                    config.directional.maximum_step_direction_mismatch_degrees
                ),
                "minimum_component_cells": config.directional.minimum_component_cells,
            },
        },
        "counts": counts,
        "transition_matrix_validation": _externalize_mapping(
            transition_validation_summary, config
        ),
        "gradient_validation": _externalize_mapping(
            gradient_validation_summary, config
        ),
        "directional_corridors": _externalize_mapping(
            directional_corridor_summary, config
        ),
        "directional_fronts": _externalize_mapping(
            directional_front_summary, config
        ),
        "transport_directional_comparison": _externalize_mapping(
            structure_comparison_summary, config
        ),
        "output_inventory": inventory,
        "options": {
            "write_debug_outputs": config.write_debug_outputs,
            "run_validation": config.run_validation,
            "debug_plots": config.plotting.debug_plots,
        },
        "continuous_fronts_created": False,
        "topology_classification_performed": False,
        "physical_identification_performed": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
