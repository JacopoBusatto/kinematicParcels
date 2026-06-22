from __future__ import annotations

import xarray as xr


def mask_values_below(
    da: xr.DataArray,
    min_mask_value: float | None,
) -> xr.DataArray:
    if min_mask_value is None:
        return da
    return da.where(da >= min_mask_value)
