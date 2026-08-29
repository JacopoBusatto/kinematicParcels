"""Independent global-gradient validation for compact retained flanks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._validation_kernel import compute_global_gradient_fields, compute_stage7_fields
from .config import CompactConfig
from .fronts import FrontSolution


@dataclass(frozen=True)
class ValidationSolution:
    global_fields: pd.DataFrame
    validation: pd.DataFrame
    summary: dict


def compute_validation(
    cells: pd.DataFrame, fronts: FrontSolution, config: CompactConfig
) -> ValidationSolution:
    global_result = compute_global_gradient_fields(
        cells,
        config.grid,
        ellipsoid=config.ellipsoid,
        zero_tolerance=config.validation.gradient_zero_tolerance,
    )
    result = compute_stage7_fields(
        cells,
        fronts.segment_fronts,
        fronts.section_summaries,
        config.grid,
        config=config.validation,
        ellipsoid=config.ellipsoid,
        precomputed_global=global_result,
        primary_experiment_id=fronts.experiment_id,
    )
    validation_fields = [
        "unique_comparison_id",
        "ridge_cell_id",
        "side",
        "ridge_type",
        "flank_lon",
        "flank_lat",
        "flank_distance_km",
        "absolute_transport_loss",
        "relative_transport_loss",
        "stage6_persistence",
        "G_perp_at_flank",
        "abs_G_perp_at_flank",
        "G_parallel_at_flank",
        "gradient_magnitude_at_flank",
        "abs_G_perp_at_core",
        "local_max_abs_G_perp",
        "distance_to_local_gradient_max_km",
        "distance_to_local_gradient_max_L_eff",
        "local_abs_G_perp_percentile",
        "nearby_branch_contamination",
        "gradient_observability",
        "n_segment_records",
        "candidate_distance_spread_km",
        "duplicate_flank_disagreement",
        "comparison_record_ids",
        "component_ids",
        "segment_ids",
        "section_ids",
        "quality_flags",
    ]
    validation = result.unique_ridge_cell_side_comparison[
        [
            name
            for name in validation_fields
            if name in result.unique_ridge_cell_side_comparison
        ]
    ].copy()
    return ValidationSolution(
        global_fields=result.global_gradient_fields,
        validation=validation,
        summary=result.summary,
    )
