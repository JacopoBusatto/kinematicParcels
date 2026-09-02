"""Neutral overlap diagnostics for independent transport/directional selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import CompactConfig
from .cores import CoreSolution
from .directional_corridors import DirectionalCorridorSolution


@dataclass(frozen=True)
class StructureComparison:
    cells: pd.DataFrame
    components: pd.DataFrame
    summary: dict[str, Any]


def compare_transport_and_directional_structures(
    cells: pd.DataFrame,
    transport: CoreSolution,
    directional: DirectionalCorridorSolution,
    config: CompactConfig,
) -> StructureComparison:
    """Classify supported cells without matching unlike component geometries."""
    supported = cells.loc[
        cells.N_out_move.ge(config.statistics.min_moving_support),
        [
            "cell_id",
            "lon_bin",
            "lat_bin",
            "lon",
            "lat",
            "N_out_move",
            "U_out_all_magnitude_km_day",
            "D_out_all_magnitude",
        ],
    ].copy()
    transport_component = transport.cores.set_index("cell_id").component_id
    directional_component = directional.corridors.set_index("cell_id").component_id
    supported["transport_component_id"] = supported.cell_id.map(transport_component)
    supported["directional_component_id"] = supported.cell_id.map(
        directional_component
    )
    supported["transport_core"] = supported.transport_component_id.notna()
    supported["directional_corridor"] = supported.directional_component_id.notna()
    supported["structure_class"] = np.select(
        [
            supported.transport_core & supported.directional_corridor,
            supported.directional_corridor,
            supported.transport_core,
        ],
        ["transport_and_directional", "directional_only", "transport_only"],
        default="neither",
    )
    supported = supported.rename(
        columns={"lon_bin": "start_lon_bin", "lat_bin": "start_lat_bin"}
    )

    component_records: list[dict[str, Any]] = []
    for structure_type, identifier_field, selected_field, overlap_field in (
        (
            "transport",
            "transport_component_id",
            "transport_core",
            "directional_corridor",
        ),
        (
            "directional",
            "directional_component_id",
            "directional_corridor",
            "transport_core",
        ),
    ):
        selected = supported.loc[supported[selected_field]]
        for component_id, group in selected.groupby(identifier_field, sort=True):
            overlap = int(group[overlap_field].sum())
            component_records.append(
                {
                    "structure_type": structure_type,
                    "component_id": component_id,
                    "n_selected_cells": len(group),
                    "n_overlap_cells": overlap,
                    "overlap_fraction": overlap / len(group),
                }
            )
    component_table = pd.DataFrame.from_records(component_records)
    counts = supported.structure_class.value_counts()
    total = len(supported)
    summary = {
        "comparison_population": (
            f"N_out_move >= configured threshold {config.statistics.min_moving_support}"
        ),
        "supported_cells": total,
        "transport_and_directional": int(
            counts.get("transport_and_directional", 0)
        ),
        "directional_only": int(counts.get("directional_only", 0)),
        "transport_only": int(counts.get("transport_only", 0)),
        "neither": int(counts.get("neither", 0)),
        "transport_and_directional_fraction": (
            float(counts.get("transport_and_directional", 0) / total)
            if total
            else np.nan
        ),
        "directional_only_fraction": (
            float(counts.get("directional_only", 0) / total) if total else np.nan
        ),
        "transport_only_fraction": (
            float(counts.get("transport_only", 0) / total) if total else np.nan
        ),
        "neither_fraction": (
            float(counts.get("neither", 0) / total) if total else np.nan
        ),
        "component_matching_performed": False,
    }
    return StructureComparison(supported.reset_index(drop=True), component_table, summary)
