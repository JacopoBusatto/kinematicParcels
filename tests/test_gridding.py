from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinematicparcels.postprocessing.core.gridding import (
    build_release_grid_from_summary,
    infer_regular_spacing_from_centers,
)


def test_infer_regular_spacing_uses_native_step_for_masked_grid() -> None:
    values = np.array(
        [
            0.0,
            0.0018,
            0.0036,
            0.0126,
            0.0144,
            0.0162,
            0.0262,
            0.0362,
            0.0462,
        ]
    )

    assert infer_regular_spacing_from_centers(values) == pytest.approx(0.0018)


def test_infer_regular_spacing_falls_back_to_smallest_gap_when_sparse() -> None:
    values = np.array([0.0, 0.0036, 0.0126])

    assert infer_regular_spacing_from_centers(values) == pytest.approx(0.0036)


def test_infer_regular_spacing_tolerates_small_float_jitter() -> None:
    values = np.array([0.0, 0.00180004, 0.00359997, 0.00540003])

    assert infer_regular_spacing_from_centers(values) == pytest.approx(0.0018)


def test_build_release_grid_from_summary_uses_native_masked_spacing() -> None:
    summary = pd.DataFrame(
        {
            "trajectory": range(9),
            "time0": pd.Timestamp("2026-01-01"),
            "lon0": [
                0.0,
                0.0018,
                0.0036,
                0.0126,
                0.0144,
                0.0162,
                0.0262,
                0.0362,
                0.0462,
            ],
            "lat0": [
                40.0,
                40.0018,
                40.0036,
                40.0126,
                40.0144,
                40.0162,
                40.0262,
                40.0362,
                40.0462,
            ],
        }
    )

    grid = build_release_grid_from_summary(summary)

    assert grid.dlon == pytest.approx(0.0018)
    assert grid.dlat == pytest.approx(0.0018)
