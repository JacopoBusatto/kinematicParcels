from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from ..config.models import DatasetCoordinatesConfig, ParcelsSchema


def open_parcels_dataset(
    path: str | Path,
    *,
    engine: str | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Open a Parcels output dataset from Zarr or NetCDF.

    Parameters
    ----------
    path
        Path to the input dataset (.zarr, .nc, .nc4).
    engine
        Optional xarray engine override.
    chunks
        Optional chunk mapping for lazy/dask-backed opening.

    Returns
    -------
    xr.Dataset
        Opened xarray dataset.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".zarr":
        return xr.open_zarr(path, chunks=chunks)

    if suffix in {".nc", ".nc4"}:
        open_kwargs: dict = {}
        if engine is not None:
            open_kwargs["engine"] = engine
        if chunks is not None:
            open_kwargs["chunks"] = chunks
        return xr.open_dataset(path, **open_kwargs)

    raise ValueError(
        f"Unsupported dataset format for '{path}'. "
        "Expected one of: .zarr, .nc, .nc4"
    )


def resolve_parcels_schema(
    ds: xr.Dataset,
    *,
    coordinates: DatasetCoordinatesConfig,
) -> ParcelsSchema:
    """
    Resolve the dataset coordinate/variable names into the internal schema.

    Parameters
    ----------
    ds
        Opened Parcels dataset.
    coordinates
        Coordinate/variable names from the post-processing configuration.

    Returns
    -------
    ParcelsSchema
        Resolved internal schema.

    Raises
    ------
    KeyError
        If a required dimension or variable is missing.
    """
    if coordinates.trajectory not in ds.dims and coordinates.trajectory not in ds.coords:
        raise KeyError(
            f"Trajectory dimension/coordinate '{coordinates.trajectory}' not found in dataset."
        )

    if coordinates.obs not in ds.dims and coordinates.obs not in ds.coords:
        raise KeyError(
            f"Observation dimension/coordinate '{coordinates.obs}' not found in dataset."
        )

    if coordinates.time not in ds.variables:
        raise KeyError(f"Time variable '{coordinates.time}' not found in dataset.")

    if coordinates.lon not in ds.variables:
        raise KeyError(f"Longitude variable '{coordinates.lon}' not found in dataset.")

    if coordinates.lat not in ds.variables:
        raise KeyError(f"Latitude variable '{coordinates.lat}' not found in dataset.")

    z_var: str | None = None
    if coordinates.z is not None:
        if coordinates.z not in ds.variables:
            raise KeyError(f"Vertical variable '{coordinates.z}' not found in dataset.")
        z_var = coordinates.z

    return ParcelsSchema(
        trajectory_dim=coordinates.trajectory,
        obs_dim=coordinates.obs,
        time_var=coordinates.time,
        lon_var=coordinates.lon,
        lat_var=coordinates.lat,
        z_var=z_var,
    )


def build_trajectory_table(
    ds: xr.Dataset,
    *,
    schema: ParcelsSchema,
    extra_vars: list[str] | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Build the canonical trajectory table from a Parcels dataset.

    Parameters
    ----------
    ds
        Opened Parcels dataset.
    schema
        Resolved Parcels schema.
    extra_vars
        Optional list of additional variables to include in the output table.
    sort
        If True, sort the output table by trajectory and obs.

    Returns
    -------
    pd.DataFrame
        Canonical trajectory table with at least the columns:
        trajectory, obs, time, lon, lat
        and optionally z plus any requested extra variables.

    Raises
    ------
    KeyError
        If a requested extra variable is missing.
    ValueError
        If one or more mandatory output columns are missing after conversion.
    """
    var_names = [
        schema.time_var,
        schema.lon_var,
        schema.lat_var,
    ]

    if schema.z_var is not None:
        var_names.append(schema.z_var)

    if extra_vars is not None:
        for var in extra_vars:
            if var not in ds.variables:
                raise KeyError(f"Requested extra variable '{var}' not found in dataset.")
            if var not in var_names:
                var_names.append(var)

    ds_sel = ds[var_names]

    df = ds_sel.to_dataframe().reset_index()

    rename_map = {
        schema.trajectory_dim: "trajectory",
        schema.obs_dim: "obs",
        schema.time_var: "time",
        schema.lon_var: "lon",
        schema.lat_var: "lat",
    }
    if schema.z_var is not None:
        rename_map[schema.z_var] = "z"

    df = df.rename(columns=rename_map)

    required_columns = ["trajectory", "obs", "time", "lon", "lat"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Trajectory table is missing required columns after conversion: "
            f"{missing}"
        )

    if sort:
        df = df.sort_values(["trajectory", "obs"]).reset_index(drop=True)

    return df


def sanitize_trajectories(
    df: pd.DataFrame,
    *,
    drop_empty: bool = True,
    truncate_at_first_invalid: bool = True,
) -> pd.DataFrame:
    """
    Clean the canonical trajectory table according to basic validity rules.

    A point is considered valid if both lon and lat are not NaN.

    Rules
    -----
    - Trajectories with no valid points are dropped.
    - If `truncate_at_first_invalid=True`, each trajectory is truncated
      at the first invalid point.
    - The output is sorted by trajectory and obs.

    Parameters
    ----------
    df
        Canonical trajectory table.
    drop_empty
        If True, remove trajectories with no valid points.
    truncate_at_first_invalid
        If True, truncate each trajectory at the first invalid point.

    Returns
    -------
    pd.DataFrame
        Cleaned trajectory table.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    required_columns = ["trajectory", "obs", "lon", "lat"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            "sanitize_trajectories requires these columns: "
            f"{required_columns}. Missing: {missing}"
        )

    if df.empty:
        return df.copy()

    df = df.sort_values(["trajectory", "obs"]).reset_index(drop=True).copy()
    is_valid = df["lon"].notna() & df["lat"].notna()
    df["_is_valid"] = is_valid

    cleaned_groups: list[pd.DataFrame] = []

    for _, g in df.groupby("trajectory", sort=False):
        valid_mask = g["_is_valid"].to_numpy()

        if not valid_mask.any():
            if drop_empty:
                continue
            cleaned = g.iloc[0:0].copy()
            cleaned_groups.append(cleaned)
            continue

        if truncate_at_first_invalid:
            first_invalid_positions = (~valid_mask).nonzero()[0]
            if len(first_invalid_positions) > 0:
                first_invalid_idx = first_invalid_positions[0]
                g = g.iloc[:first_invalid_idx].copy()
            else:
                g = g.copy()
        else:
            g = g.loc[g["_is_valid"]].copy()

        if g.empty and drop_empty:
            continue

        cleaned_groups.append(g)

    if not cleaned_groups:
        result = df.iloc[0:0].copy()
        return result.drop(columns="_is_valid")

    result = pd.concat(cleaned_groups, ignore_index=True)
    result = result.drop(columns="_is_valid")
    result = result.sort_values(["trajectory", "obs"]).reset_index(drop=True)

    return result


def truncate_stagnant_trajectories(
    df: pd.DataFrame,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    trajectory_col: str = "trajectory",
    obs_col: str = "obs",
    tol: float = 1.0e-6,
    min_consecutive: int = 2,
    drop_fully_stagnant: bool = True,
) -> pd.DataFrame:
    """
    Truncate trajectories when they become stagnant.

    A trajectory is considered stagnant when consecutive positions satisfy:
        abs(dlon) <= tol and abs(dlat) <= tol

    If a stagnant sequence of at least `min_consecutive` consecutive steps is found,
    the trajectory is truncated from the first stagnant point onward.

    Parameters
    ----------
    df
        Canonical trajectory table.
    lon_col, lat_col
        Spatial columns.
    trajectory_col
        Trajectory identifier column.
    obs_col
        Observation index column.
    tol
        Tolerance used to detect near-zero motion.
    min_consecutive
        Minimum number of consecutive stagnant steps required to trigger truncation.
    drop_fully_stagnant
        If True, trajectories stagnant from the beginning are removed entirely.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    required = [trajectory_col, obs_col, lon_col, lat_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"truncate_stagnant_trajectories requires columns {required}. Missing: {missing}"
        )

    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")

    if df.empty:
        return df.copy()

    df = df.sort_values([trajectory_col, obs_col]).reset_index(drop=True).copy()

    cleaned_groups: list[pd.DataFrame] = []

    for _, g in df.groupby(trajectory_col, sort=False):
        g = g.sort_values(obs_col).copy()

        if len(g) <= 1:
            cleaned_groups.append(g)
            continue

        lon = g[lon_col].to_numpy()
        lat = g[lat_col].to_numpy()

        dlon = np.abs(np.diff(lon))
        dlat = np.abs(np.diff(lat))

        stagnant_step = (dlon <= tol) & (dlat <= tol)

        run_len = 0
        start_idx: int | None = None

        for i, is_stagnant in enumerate(stagnant_step):
            if is_stagnant:
                run_len += 1
                if run_len == min_consecutive:
                    # i refers to diff between points i and i+1
                    # if a stagnant run starts at diff index s,
                    # we want to drop from point s onward
                    start_idx = i - min_consecutive + 1
                    break
            else:
                run_len = 0

        if start_idx is None:
            cleaned_groups.append(g)
            continue

        # start_idx refers to the first point of the stagnant segment
        if start_idx == 0:
            if drop_fully_stagnant:
                continue
            cleaned = g.iloc[0:0].copy()
            cleaned_groups.append(cleaned)
            continue

        cleaned = g.iloc[:start_idx].copy()
        if not cleaned.empty:
            cleaned_groups.append(cleaned)

    if not cleaned_groups:
        return df.iloc[0:0].copy()

    out = pd.concat(cleaned_groups, ignore_index=True)
    out = out.sort_values([trajectory_col, obs_col]).reset_index(drop=True)
    return out


def load_trajectory_table(
    path: str | Path,
    *,
    coordinates: DatasetCoordinatesConfig = DatasetCoordinatesConfig(),
    engine: str | None = None,
    chunks: dict | None = None,
    extra_vars: list[str] | None = None,
    sort: bool = True,
    drop_empty: bool = True,
    truncate_at_first_invalid: bool = True,
    truncate_stagnant: bool = False,
    stagnant_tol: float = 1.0e-6,
    stagnant_min_consecutive: int = 2,
) -> pd.DataFrame:
    """
    High-level convenience function to load and clean a trajectory table.

    This function runs the full basic pipeline:
    open dataset -> resolve schema -> build trajectory table -> sanitize.

    Parameters
    ----------
    path
        Path to the input Parcels dataset.
    coordinates
        Dataset coordinate/variable naming configuration.
    engine
        Optional xarray engine override.
    chunks
        Optional chunk mapping for lazy/dask-backed opening.
    extra_vars
        Optional list of additional variables to include.
    sort
        If True, sort the raw trajectory table by trajectory and obs.
    drop_empty
        If True, remove trajectories with no valid points.
    truncate_at_first_invalid
        If True, truncate each trajectory at the first invalid point.

    Returns
    -------
    pd.DataFrame
        Cleaned canonical trajectory table.
    """
    ds = open_parcels_dataset(path, engine=engine, chunks=chunks)
    schema = resolve_parcels_schema(ds, coordinates=coordinates)
    df = build_trajectory_table(ds, schema=schema, extra_vars=extra_vars, sort=sort)
    df = sanitize_trajectories(
        df,
        drop_empty=drop_empty,
        truncate_at_first_invalid=truncate_at_first_invalid,
    )
    if truncate_stagnant:
        df = truncate_stagnant_trajectories(
            df,
            tol=stagnant_tol,
            min_consecutive=stagnant_min_consecutive,
        )
    return df