from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr


_SUPPORTED_TABLE_FORMATS = {"parquet", "csv"}


def _normalize_netcdf_attr_value(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, tuple):
        return tuple(_normalize_netcdf_attr_value(item) for item in value)

    if isinstance(value, list):
        return [_normalize_netcdf_attr_value(item) for item in value]

    return value


def _prepare_dataset_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy(deep=False)
    out.attrs = {
        key: _normalize_netcdf_attr_value(value)
        for key, value in out.attrs.items()
    }

    for var_name in out.variables:
        out[var_name].attrs = {
            key: _normalize_netcdf_attr_value(value)
            for key, value in out[var_name].attrs.items()
        }

    return out


def _write_eager_dataset_netcdf(ds: xr.Dataset, path: Path) -> None:
    """
    Write an eager dataset without probing for a distributed Dask scheduler.

    Recent xarray versions perform that probe even when every variable is
    NumPy-backed. Importing ``distributed`` can initialize unrelated network
    and SSL dependencies, so it is both unnecessary and potentially fragile
    for an eager local write. Lazy datasets do not use this helper.
    """
    with ExitStack() as stack:
        for module_name in (
            "xarray.backends.writers",
            "xarray.backends.api",
            "xarray.backends.locks",
        ):
            try:
                module = __import__(module_name, fromlist=["get_dask_scheduler"])
            except ImportError:
                continue
            if hasattr(module, "get_dask_scheduler"):
                stack.enter_context(
                    patch.object(module, "get_dask_scheduler", return_value=None)
                )
        ds.to_netcdf(path)


def save_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    format: str = "parquet",
    index: bool = False,
) -> Path:
    """
    Save a tabular product to disk.

    Supported formats:
    - parquet
    - csv
    """
    fmt = format.lower().strip()
    if fmt not in _SUPPORTED_TABLE_FORMATS:
        raise ValueError(
            f"Unsupported table format '{format}'. "
            f"Supported formats: {sorted(_SUPPORTED_TABLE_FORMATS)}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=index)
    elif fmt == "csv":
        df.to_csv(path, index=index)

    return path


def save_trajectory_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    format: str = "parquet",
    index: bool = False,
) -> Path:
    """
    Save a canonical trajectory table to disk.
    """
    return save_table(df, path, format=format, index=index)


def save_particle_summary(
    df: pd.DataFrame,
    path: str | Path,
    *,
    format: str = "parquet",
    index: bool = False,
) -> Path:
    """
    Save a particle summary table to disk.
    """
    return save_table(df, path, format=format, index=index)


def save_grid_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    format: str = "parquet",
    index: bool = False,
) -> Path:
    """
    Save an aggregated grid table to disk.
    """
    return save_table(df, path, format=format, index=index)


def save_dataset_netcdf(
    ds: xr.Dataset,
    path: str | Path,
) -> Path:
    """
    Save an xarray.Dataset to NetCDF.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ds_to_save = _prepare_dataset_for_netcdf(ds)
    is_eager = all(variable.chunks is None for variable in ds_to_save.variables.values())
    if is_eager:
        _write_eager_dataset_netcdf(ds_to_save, path)
    else:
        ds_to_save.to_netcdf(path)

    return path
