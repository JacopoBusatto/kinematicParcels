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
