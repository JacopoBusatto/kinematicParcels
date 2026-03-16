from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


_SUPPORTED_TABLE_FORMATS = {"parquet", "csv"}


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

    ds.to_netcdf(path)

    return path