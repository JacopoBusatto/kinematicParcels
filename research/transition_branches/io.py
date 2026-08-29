"""Production table and reproducibility-file writing."""

from __future__ import annotations

import hashlib
import json
import platform
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
) -> None:
    for name, table in zip(STANDARD_TABLES, (cells, cores, fronts), strict=True):
        table.to_parquet(run_dir / name, index=False)


def write_debug_tables(
    run_dir: Path,
    *,
    candidate_drops: pd.DataFrame,
    cross_sections: pd.DataFrame,
    section_summaries: pd.DataFrame,
    components: pd.DataFrame,
    segment_fronts: pd.DataFrame,
) -> list[str]:
    outputs = {
        "candidate_drop_zones.parquet": candidate_drops,
        "raw_cross_sections.parquet": cross_sections,
        "section_composites.parquet": section_summaries,
        "component_graph_details.parquet": components,
        "segment_front_candidates.parquet": segment_fronts,
    }
    for name, table in outputs.items():
        table.to_parquet(run_dir / name, index=False)
    return list(outputs)


def write_validation_table(run_dir: Path, validation: pd.DataFrame) -> str:
    name = "gradient_validation.parquet"
    validation.to_parquet(run_dir / name, index=False)
    return name


def write_reproducibility_files(
    run_dir: Path,
    *,
    config: CompactConfig,
    input_path: Path,
    transport_threshold_km_day: float,
    counts: dict[str, int],
    transition_validation_summary: dict[str, Any],
    gradient_validation_summary: dict[str, Any] | None,
    figures: list[Path],
    optional_outputs: list[str],
) -> None:
    config_name = "resolved_config.yaml"
    (run_dir / config_name).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    inventory = [*STANDARD_TABLES, config_name, "manifest.json"]
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
            "timestep_days": config.input.timestep_days,
            "normalized_source_probability_contract": True,
        },
        "selection": {
            "min_moving_support": config.statistics.min_moving_support,
            "transport_percentile": config.branches.transport_percentile,
            "ridge_field": config.branches.ridge_field,
            "transport_threshold_km_day": transport_threshold_km_day,
        },
        "counts": counts,
        "transition_matrix_validation": transition_validation_summary,
        "gradient_validation": gradient_validation_summary,
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
