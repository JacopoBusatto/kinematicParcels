from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_TRAJECTORY_DATASET_ATTRS = {
    "Conventions": "CF-1.6/CF-1.7",
    "feature_type": "trajectory",
    "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
}


def _series_to_fixed_width_array(series: pd.Series, length: int) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        out = np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]")
        values = series.to_numpy(dtype="datetime64[ns]")
        out[: len(values)] = values
        return out

    if pd.api.types.is_numeric_dtype(series.dtype):
        out = np.full(length, np.nan, dtype=np.float64)
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        out[: len(values)] = values
        return out

    values = series.astype(str).to_numpy()
    width = max([1, *(len(value) for value in values)])
    out = np.full(length, "", dtype=f"<U{width}")
    out[: len(values)] = values
    return out


def build_dataset_from_trajectories(
    trajectories: list[pd.DataFrame],
    *,
    trajectory_level_columns: set[str] | None = None,
    dataset_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    if not trajectories:
        raise ValueError("No trajectories were provided to build_dataset_from_trajectories.")

    resolved_trajectory_level_columns = set(trajectory_level_columns or ())

    ordered = sorted(trajectories, key=lambda frame: int(frame["trajectory"].iloc[0]))
    trajectory_ids = np.asarray([int(frame["trajectory"].iloc[0]) for frame in ordered], dtype=np.int64)
    max_obs = max(len(frame) for frame in ordered)
    obs = np.arange(max_obs, dtype=np.int32)

    variable_names = []
    for frame in ordered:
        for column in frame.columns:
            if column not in {"trajectory", "obs"} and column not in variable_names:
                variable_names.append(column)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in variable_names:
        if name in resolved_trajectory_level_columns:
            values = np.asarray([frame[name].iloc[0] for frame in ordered])
            if pd.api.types.is_numeric_dtype(values.dtype):
                values = pd.to_numeric(values, errors="raise").astype(np.int64)
            data_vars[name] = (("trajectory",), values)
            continue

        sample_series = ordered[0][name]
        rows = [_series_to_fixed_width_array(frame[name], max_obs) for frame in ordered]
        values = np.stack(rows, axis=0)

        if pd.api.types.is_datetime64_any_dtype(sample_series.dtype):
            values = values.astype("datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(sample_series.dtype):
            values = values.astype(np.float64)

        data_vars[name] = (("trajectory", "obs"), values)

    attrs = dict(DEFAULT_TRAJECTORY_DATASET_ATTRS)
    if dataset_attrs:
        attrs.update(dataset_attrs)

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "trajectory": trajectory_ids,
            "obs": obs,
        },
        attrs=attrs,
    )


def build_zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    trajectory_size = int(ds.sizes.get("trajectory", 1))
    obs_size = int(ds.sizes.get("obs", 1))
    obs_chunk = 1 if obs_size > 0 else 1

    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in ds.data_vars.items():
        if variable.dims == ("trajectory", "obs"):
            encoding[name] = {"chunks": (trajectory_size, obs_chunk)}
        elif variable.dims == ("trajectory",):
            encoding[name] = {"chunks": (trajectory_size,)}

    return encoding