"""Parallel transport and distance-free directional structure workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .comparison import compare_transport_and_directional_structures
from .config import CompactConfig, load_config
from .cores import compute_current_cores
from .directional_corridors import compute_directional_corridors
from .directional_fronts import compute_probable_directional_fronts
from .fronts import compute_probable_fronts
from .io import (
    create_run_directory,
    write_debug_tables,
    write_directional_debug_tables,
    write_directional_tables,
    write_reproducibility_files,
    write_scientific_tables,
    write_validation_table,
)
from .plotting import create_standard_figures
from .statistics import compact_cell_table, compute_transition_statistics
from .validation import compute_validation


def run(config: CompactConfig) -> Path:
    input_path = Path(config.input.transition_table).resolve()
    run_dir = create_run_directory(config)
    matrix = pd.read_parquet(input_path)

    statistics = compute_transition_statistics(matrix, config)
    directional_corridors = compute_directional_corridors(statistics.cells, config)
    directional_fronts = compute_probable_directional_fronts(
        statistics.cells, directional_corridors, config
    )
    cores = compute_current_cores(statistics.cells, config)
    fronts = compute_probable_fronts(statistics.cells, cores, config)
    comparison = compare_transport_and_directional_structures(
        statistics.cells, cores, directional_corridors, config
    )
    validation = (
        compute_validation(statistics.cells, fronts, config)
        if config.run_validation
        else None
    )
    cells = compact_cell_table(statistics.cells)
    write_scientific_tables(
        run_dir, cells=cells, cores=cores.cores, fronts=fronts.fronts
    )
    write_directional_tables(
        run_dir,
        corridors=directional_corridors.corridors,
        fronts=directional_fronts.fronts,
        comparison=comparison.cells,
        component_comparison=comparison.components,
    )

    optional_outputs: list[str] = []
    if config.write_debug_outputs:
        optional_outputs.extend(
            write_debug_tables(
                run_dir,
                candidate_drops=fronts.candidate_drops,
                cross_sections=fronts.cross_sections,
                section_summaries=fronts.section_summaries,
                components=cores.components,
                segment_fronts=fronts.segment_fronts,
            )
        )
        optional_outputs.extend(
            write_directional_debug_tables(
                run_dir,
                candidate_drops=directional_fronts.candidate_drops,
                cross_sections=directional_fronts.cross_sections,
                section_summaries=directional_fronts.section_summaries,
                components=directional_corridors.components,
                graph_edges=directional_corridors.edges,
            )
        )
    if validation is not None:
        optional_outputs.append(write_validation_table(run_dir, validation.validation))

    figures = create_standard_figures(
        statistics.cells,
        cores,
        fronts,
        config,
        run_dir / "figures",
        directional_corridors=directional_corridors,
        directional_fronts=directional_fronts,
        validation=validation,
    )
    status_counts = fronts.fronts.front_status.value_counts()
    directional_status_counts = directional_fronts.fronts.front_status.value_counts()
    write_reproducibility_files(
        run_dir,
        config=config,
        input_path=input_path,
        transport_threshold_km_day=cores.threshold_km_day,
        counts={
            "cells": len(cells),
            "current_core_cells": len(cores.cores),
            "core_sides": len(fronts.fronts),
            "probable_transport_fronts": int(
                status_counts.get("probable_transport_front", 0)
            ),
            "observable_sides_without_front": int(
                status_counts.get("observable_no_retained_front", 0)
            ),
            "unobservable_sides": int(status_counts.get("side_not_observable", 0)),
            "directional_corridor_cells": len(directional_corridors.corridors),
            "directional_corridor_components": len(directional_corridors.components),
            "directional_sides": len(directional_fronts.fronts),
            "probable_directional_fronts": int(
                directional_status_counts.get("probable_directional_front", 0)
            ),
            "observable_sides_without_directional_front": int(
                directional_status_counts.get(
                    "observable_no_retained_directional_front", 0
                )
            ),
            "unobservable_directional_sides": int(
                directional_status_counts.get("side_not_observable", 0)
            ),
        },
        transition_validation_summary=statistics.validation_summary,
        gradient_validation_summary=(validation.summary if validation else None),
        directional_corridor_summary=directional_corridors.summary,
        directional_front_summary=directional_fronts.summary,
        structure_comparison_summary=comparison.summary,
        figures=figures,
        optional_outputs=optional_outputs,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect transport cores/fronts and independent directional corridors/fronts"
        )
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
