from __future__ import annotations

import numpy as np


def infer_colorbar_extend(
    values,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Return the colorbar extend mode required by finite values outside limits.
    """
    data = np.asarray(values)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "neither"

    has_under = vmin is not None and bool(np.any(finite < vmin))
    has_over = vmax is not None and bool(np.any(finite > vmax))

    if has_under and has_over:
        return "both"
    if has_under:
        return "min"
    if has_over:
        return "max"
    return "neither"


def colorbar_extend_from_limits(
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Return the colorbar extend mode implied by explicit plot limits.

    This is useful when a configured vmin/vmax should be shown as an open-ended
    scale even if the current data do not contain values outside those bounds.
    """
    if vmin is not None and vmax is not None:
        return "both"
    if vmin is not None:
        return "min"
    if vmax is not None:
        return "max"
    return "neither"
