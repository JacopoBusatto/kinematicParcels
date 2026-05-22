from __future__ import annotations

from pathlib import Path

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
    ds_to_save.to_netcdf(path)

    return path