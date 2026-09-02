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


def _directional_vector_fields(
    cells: pd.DataFrame, *, zero_tolerance: float
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Expose the distance-free first harmonic as moving/all-population vectors."""
    required = {
        "M1_out_real",
        "M1_out_imag",
        "P_move",
        "R1_out",
        "theta1_out",
    }
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"Directional first-harmonic fields missing columns: {missing}")

    output = cells.copy()
    # Bearings are clockwise from north, so the harmonic's imaginary component
    # is eastward and its real component is northward.
    output["D_out_move_east"] = output.M1_out_imag
    output["D_out_move_north"] = output.M1_out_real
    output["D_out_move_magnitude"] = np.hypot(
        output.D_out_move_east, output.D_out_move_north
    )
    output["D_out_all_east"] = output.P_move * output.D_out_move_east
    output["D_out_all_north"] = output.P_move * output.D_out_move_north
    output["D_out_all_magnitude"] = np.hypot(
        output.D_out_all_east, output.D_out_all_north
    )

    # A source population containing only stay transitions has a zero
    # full-population directional vector, while its moving-conditioned vector
    # and bearing remain undefined.
    no_movement = output.P_move.eq(0.0) & output.P_move.notna()
    output.loc[
        no_movement,
        ["D_out_all_east", "D_out_all_north", "D_out_all_magnitude"],
    ] = 0.0

    output["directional_identity_east_residual"] = (
        output.D_out_all_east - output.P_move * output.D_out_move_east
    )
    output["directional_identity_north_residual"] = (
        output.D_out_all_north - output.P_move * output.D_out_move_north
    )
    output["directional_move_magnitude_minus_R1"] = (
        output.D_out_move_magnitude - output.R1_out
    )
    output["directional_all_magnitude_minus_P_move_R1"] = (
        output.D_out_all_magnitude - output.P_move * output.R1_out
    )

    move_bearing = np.full(len(output), np.nan, dtype=float)
    move_defined = output.D_out_move_magnitude.gt(zero_tolerance)
    move_bearing[move_defined] = np.remainder(
        np.rad2deg(
            np.arctan2(
                output.loc[move_defined, "D_out_move_east"],
                output.loc[move_defined, "D_out_move_north"],
            )
        ),
        360.0,
    )
    all_bearing = np.full(len(output), np.nan, dtype=float)
    all_defined = output.D_out_all_magnitude.gt(zero_tolerance)
    all_bearing[all_defined] = np.remainder(
        np.rad2deg(
            np.arctan2(
                output.loc[all_defined, "D_out_all_east"],
                output.loc[all_defined, "D_out_all_north"],
            )
        ),
        360.0,
    )
    output["D_out_move_bearing_residual_degrees"] = np.nan
    output["D_out_all_bearing_residual_degrees"] = np.nan
    theta_defined = output.theta1_out.notna()
    move_compare = move_defined & theta_defined
    all_compare = all_defined & theta_defined
    output.loc[move_compare, "D_out_move_bearing_residual_degrees"] = np.remainder(
        move_bearing[move_compare] - output.loc[move_compare, "theta1_out"] + 180.0,
        360.0,
    ) - 180.0
    output.loc[all_compare, "D_out_all_bearing_residual_degrees"] = np.remainder(
        all_bearing[all_compare] - output.loc[all_compare, "theta1_out"] + 180.0,
        360.0,
    ) - 180.0

    def maximum_absolute(field: str) -> float:
        values = output[field].dropna().abs()
        return float(values.max()) if not values.empty else np.nan

    summary: dict[str, float | int] = {
        "cells_with_D_out_move": int(output.D_out_move_magnitude.notna().sum()),
        "cells_with_D_out_all": int(output.D_out_all_magnitude.notna().sum()),
        "cells_with_defined_directional_bearing": int(theta_defined.sum()),
        "D_out_all_equals_P_move_D_out_move_max_abs_east": maximum_absolute(
            "directional_identity_east_residual"
        ),
        "D_out_all_equals_P_move_D_out_move_max_abs_north": maximum_absolute(
            "directional_identity_north_residual"
        ),
        "D_out_move_magnitude_equals_R1_max_abs": maximum_absolute(
            "directional_move_magnitude_minus_R1"
        ),
        "D_out_all_magnitude_equals_P_move_R1_max_abs": maximum_absolute(
            "directional_all_magnitude_minus_P_move_R1"
        ),
        "D_out_move_bearing_equals_theta1_max_abs_degrees": maximum_absolute(
            "D_out_move_bearing_residual_degrees"
        ),
        "D_out_all_bearing_equals_theta1_max_abs_degrees": maximum_absolute(
            "D_out_all_bearing_residual_degrees"
        ),
    }
    return output, summary


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
    cells, directional_summary = _directional_vector_fields(
        cells,
        zero_tolerance=config.statistics.direction_zero_tolerance_km,
    )
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
    validation_summary = dict(validation.summary)
    validation_summary["directional_vector_identities"] = directional_summary
    return TransitionStatistics(
        links=links, cells=cells, validation_summary=validation_summary
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
    "theta1_out",
    "R1_out",
    "R2_out",
    "angular_entropy_out",
    "D_out_move_east",
    "D_out_move_north",
    "D_out_move_magnitude",
    "D_out_all_east",
    "D_out_all_north",
    "D_out_all_magnitude",
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
