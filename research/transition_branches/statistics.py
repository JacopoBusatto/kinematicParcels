"""Single-pass transition statistics used by the compact workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._statistics_kernel import (
    GeometryConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    compute_stage1_fields,
    compute_stage2_fields,
    compute_stage3_fields,
    compute_support_fields,
    validate_transition_table,
)
from .config import CompactConfig


@dataclass(frozen=True)
class TransitionStatistics:
    links: pd.DataFrame
    cells: pd.DataFrame
    validation_summary: dict


def compute_transition_statistics(
    table: pd.DataFrame, config: CompactConfig
) -> TransitionStatistics:
    """Validate the normalized matrix and calculate each retained statistic once."""
    validation = validate_transition_table(
        table,
        config.grid,
        config.validation,
    )
    if validation.errors:
        raise ValueError(f"Transition-matrix validation failed: {validation.errors}")
    support = compute_support_fields(
        validation.links, config.grid, (config.statistics.min_moving_support,)
    )
    stage1_config = Stage1Config(
        primary_visualization_min_moving_count=config.statistics.min_moving_support,
        sensitivity_visualization_min_moving_count=config.statistics.min_moving_support,
        direction_zero_tolerance_km=config.statistics.direction_zero_tolerance_km,
    )
    outward = compute_stage1_fields(
        validation.links,
        support.cells,
        config.grid,
        timestep_days=config.input.timestep_days,
        geometry=GeometryConfig(ellipsoid=config.ellipsoid),
        config=stage1_config,
    )
    stage2_config = Stage2Config(
        angular_bins=config.statistics.angular_bins,
        harmonic_zero_tolerance=config.statistics.direction_zero_tolerance_km,
        high_R1=config.statistics.high_R1,
        low_R1=config.statistics.low_R1,
    )
    outgoing = compute_stage2_fields(
        outward.links,
        outward.cells,
        config.grid,
        stage1=stage1_config,
        config=stage2_config,
    )
    incoming = compute_stage3_fields(
        outward.links,
        outgoing.cells,
        config.grid,
        stage1=stage1_config,
        stage2=stage2_config,
        config=Stage3Config(
            angular_bins=config.statistics.angular_bins,
            harmonic_zero_tolerance=config.statistics.direction_zero_tolerance_km,
        ),
    )
    cells, links = incoming.cells, incoming.links
    cells = cells.copy()
    cells["S_transport_km_day"] = cells.U_out_all_magnitude_km_day
    if {"theta1_out", "theta1_in_motion_destination"} <= set(cells):
        cells["delta_theta_io"] = (
            np.remainder(
                cells.theta1_out - cells.theta1_in_motion_destination + 180.0, 360.0
            )
            - 180.0
        )
    else:
        cells["delta_theta_io"] = np.nan
    cells["delta_theta_io_1"] = cells.delta_theta_io
    if "theta_mu_in_motion_destination" in cells:
        cells["delta_theta_io_mu"] = (
            np.remainder(
                cells.theta_mu_out - cells.theta_mu_in_motion_destination + 180.0,
                360.0,
            )
            - 180.0
        )
    else:
        cells["delta_theta_io_mu"] = np.nan
    return TransitionStatistics(
        links=links, cells=cells, validation_summary=validation.summary
    )


COMPACT_CELL_FIELDS = (
    "cell_id",
    "lon_bin",
    "lat_bin",
    "lon",
    "lat",
    "N_out_move",
    "N_in_move",
    "P_stay",
    "P_move",
    "U_out_all_east_km_day",
    "U_out_all_north_km_day",
    "U_out_all_magnitude_km_day",
    "theta_mu_out",
    "R1_out",
    "R2_out",
    "angular_entropy_out",
    "R1_in",
    "R2_in",
    "theta1_in_motion_destination",
    "theta_mu_in_motion_destination",
    "delta_theta_mu1_in",
    "delta_theta_io",
)


def compact_cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical scientist-facing cell table."""
    selected = [name for name in COMPACT_CELL_FIELDS if name in cells]
    return cells[selected].rename(
        columns={"lon_bin": "start_lon_bin", "lat_bin": "start_lat_bin"}
    )
