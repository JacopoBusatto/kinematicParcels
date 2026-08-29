from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import KEY_COLUMNS, REQUIRED_COLUMNS
from .config import GridConfig, ValidationConfig


@dataclass(frozen=True)
class ValidationResult:
    transitions: pd.DataFrame
    row_normalization: pd.DataFrame
    duplicates: pd.DataFrame
    summary: dict
    errors: tuple[str, ...]


def validate_transition_matrix(
    table: pd.DataFrame, grid: GridConfig, config: ValidationConfig
) -> ValidationResult:
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        return ValidationResult(table.copy(), pd.DataFrame(), pd.DataFrame(),
                                {"missing_columns": missing},
                                (f"missing_columns:{','.join(missing)}",))

    df = table.loc[:, REQUIRED_COLUMNS].copy()
    df = df.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    df.insert(0, "transition_id", np.arange(len(df), dtype=np.int64))
    df["start_cell_id"] = df.start_lat_bin * grid.nlon + df.start_lon_bin
    df["end_cell_id"] = df.end_lat_bin * grid.nlon + df.end_lon_bin
    errors: list[str] = []

    int_cols = [*KEY_COLUMNS, "transition_count"]
    for column in int_cols:
        if not pd.api.types.is_integer_dtype(df[column]):
            errors.append(f"invalid_dtype:{column}")
    for column in REQUIRED_COLUMNS[4:8] + ("transition_probability",):
        if not pd.api.types.is_float_dtype(df[column]):
            errors.append(f"invalid_dtype:{column}")
    numeric = list(REQUIRED_COLUMNS[4:])
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        errors.append("non_finite_values")
    if (df.transition_count <= 0).any():
        errors.append("nonpositive_transition_count")
    if ((df.transition_probability < 0) | (df.transition_probability > 1)).any():
        errors.append("probability_out_of_range")

    dup_mask = df.duplicated(list(KEY_COLUMNS), keep=False)
    duplicates = df.loc[dup_mask].copy()
    if dup_mask.any():
        errors.append("duplicate_transition_keys")

    bin_valid = (
        df.start_lon_bin.between(0, grid.nlon - 1)
        & df.end_lon_bin.between(0, grid.nlon - 1)
        & df.start_lat_bin.between(0, grid.nlat - 1)
        & df.end_lat_bin.between(0, grid.nlat - 1)
    )
    if not bin_valid.all():
        errors.append("bin_out_of_bounds")

    expected = {
        "start_lon_center": grid.lon_min + (df.start_lon_bin + 0.5) * grid.dlon,
        "start_lat_center": grid.lat_min + (df.start_lat_bin + 0.5) * grid.dlat,
        "end_lon_center": grid.lon_min + (df.end_lon_bin + 0.5) * grid.dlon,
        "end_lat_center": grid.lat_min + (df.end_lat_bin + 0.5) * grid.dlat,
    }
    center_error = np.zeros(len(df), dtype=bool)
    for name, values in expected.items():
        center_error |= ~np.isclose(
            df[name].to_numpy(float), values.to_numpy(float), rtol=0,
            atol=config.center_atol_degrees,
        )
    if center_error.any():
        errors.append("grid_center_mismatch")

    grouped = df.groupby(["start_lon_bin", "start_lat_bin", "start_cell_id"], sort=True)
    row = grouped.agg(
        N_i=("transition_count", "sum"),
        sum_probability=("transition_probability", "sum"),
        n_destinations=("transition_id", "size"),
    ).reset_index()
    row["sum_count"] = row.N_i
    row["normalization_residual"] = row.sum_probability - 1.0
    stay = df.start_cell_id.eq(df.end_cell_id)
    p_stay = df.loc[stay].groupby("start_cell_id").transition_probability.sum()
    row["P_stay"] = row.start_cell_id.map(p_stay).fillna(0.0)
    row["normalization_valid"] = row.normalization_residual.abs().le(config.normalization_atol)

    expected_probability = df.transition_count / grouped.transition_count.transform("sum")
    probability_valid = np.isclose(
        df.transition_probability, expected_probability,
        rtol=config.probability_rtol, atol=config.probability_atol,
    )
    if not row.normalization_valid.all():
        errors.append("row_normalization_failure")
    if not probability_valid.all():
        errors.append("count_probability_identity_failure")
    probability_by_cell = pd.Series(probability_valid, index=df.index).groupby(df.start_cell_id).all()
    row["count_probability_valid"] = row.start_cell_id.map(probability_by_cell).to_numpy(bool)
    row["validation_flags"] = np.where(
        row.normalization_valid & row.count_probability_valid, "", "normalization"
    )

    summary = {
        "n_sparse_transitions": len(df),
        "total_transition_count": int(df.transition_count.sum()),
        "populated_start_cells": len(row),
        "duplicate_rows": len(duplicates),
        "center_mismatch_rows": int(center_error.sum()),
        "normalization_failure_cells": int((~row.normalization_valid).sum()),
        "probability_identity_failure_rows": int((~probability_valid).sum()),
        "normalization_residual_max_abs": float(row.normalization_residual.abs().max()),
        "N_i_min": int(row.N_i.min()),
        "N_i_max": int(row.N_i.max()),
        "errors": errors,
    }
    return ValidationResult(df, row, duplicates, summary, tuple(errors))
