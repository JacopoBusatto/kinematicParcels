"""Transition matrix to Lagrangian transport, current cores, and fronts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CompactConfig, load_config
from .cores import compute_current_cores
from .fronts import compute_probable_fronts
from .io import (
    create_run_directory,
    write_debug_tables,
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
    cores = compute_current_cores(statistics.cells, config)
    fronts = compute_probable_fronts(statistics.cells, cores, config)
    validation = (
        compute_validation(statistics.cells, fronts, config)
        if config.run_validation
        else None
    )
    cells = compact_cell_table(statistics.cells)
    write_scientific_tables(
        run_dir, cells=cells, cores=cores.cores, fronts=fronts.fronts
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
    if validation is not None:
        optional_outputs.append(write_validation_table(run_dir, validation.validation))

    figures = create_standard_figures(
        statistics.cells,
        cores,
        fronts,
        config,
        run_dir / "figures",
        validation=validation,
    )
    status_counts = fronts.fronts.front_status.value_counts()
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
        },
        transition_validation_summary=statistics.validation_summary,
        gradient_validation_summary=(validation.summary if validation else None),
        figures=figures,
        optional_outputs=optional_outputs,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect Lagrangian current cores and probable transport fronts"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
