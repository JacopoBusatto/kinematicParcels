from __future__ import annotations

import numpy as np

from kinematicparcels.postprocessing.plotting.colorbar import infer_colorbar_extend


def test_infer_colorbar_extend_uses_strict_clipping_bounds() -> None:
    values = np.array([0.0, 1.0, 2.0])

    assert infer_colorbar_extend(values, vmin=0.0, vmax=2.0) == "neither"


def test_infer_colorbar_extend_detects_lower_clipping() -> None:
    values = np.array([-0.1, 0.0, 1.0])

    assert infer_colorbar_extend(values, vmin=0.0, vmax=None) == "min"


def test_infer_colorbar_extend_detects_upper_clipping() -> None:
    values = np.array([0.0, 1.0, 2.1])

    assert infer_colorbar_extend(values, vmin=None, vmax=2.0) == "max"


def test_infer_colorbar_extend_detects_both_clipping_sides() -> None:
    values = np.array([-0.1, 0.0, 2.1, np.nan])

    assert infer_colorbar_extend(values, vmin=0.0, vmax=2.0) == "both"
