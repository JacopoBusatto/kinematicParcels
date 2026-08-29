from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

from ..config.models import ParcelsSchema, PostprocessConfig
from ..core import build_particle_summary
from ..io import (
    load_trajectory_table,
    open_parcels_dataset,
    resolve_parcels_schema,
)

_OBSERVATION_VARIABLES_ATTR = "postprocessing_observation_variables"
_TRAJECTORY_METADATA_VARIABLES_ATTR = "postprocessing_trajectory_metadata_variables"
OBSERVATION_VARIABLE_METADATA_CONTEXT_KEY = "observation_variable_metadata"
_GROUP_METADATA_VARIABLES = {
    "circle_id",
    "group_id",
    "group_member",
    "group_size",
}
_MEMBER_COORDINATE_VARIABLES = {
    f"{axis}_{member}"
    for axis in ("lon", "lat")
    for member in (1, 2, 3, 4, 5)
}


@dataclass(frozen=True)
class _OptionalVariableRoles:
    observation: tuple[str, ...] = ()
    trajectory_metadata: tuple[str, ...] = ()

    @property
    def all(self) -> tuple[str, ...]:
        return self.trajectory_metadata + self.observation


def _discover_optional_variables(
    ds: xr.Dataset,
    schema: ParcelsSchema,
) -> _OptionalVariableRoles:
    """Discover compatible non-canonical data variables in a Parcels dataset."""
    canonical = {
        schema.time_var,
        schema.lon_var,
        schema.lat_var,
    }
    if schema.z_var is not None:
        canonical.add(schema.z_var)

    observation: list[str] = []
    trajectory_metadata: list[str] = []
    observation_dims = {schema.trajectory_dim, schema.obs_dim}

    for name, variable in ds.data_vars.items():
        name = str(name)
        if name in canonical:
            continue

        dims = tuple(str(dim) for dim in variable.dims)
        if dims == (schema.trajectory_dim,):
            trajectory_metadata.append(name)
        elif len(dims) == 2 and set(dims) == observation_dims:
            if name in _GROUP_METADATA_VARIABLES:
                trajectory_metadata.append(name)
            else:
                observation.append(name)

    return _OptionalVariableRoles(
        observation=tuple(observation),
        trajectory_metadata=tuple(trajectory_metadata),
    )


def _attach_optional_variable_roles(
    df: pd.DataFrame,
    roles: _OptionalVariableRoles,
) -> pd.DataFrame:
    df.attrs[_OBSERVATION_VARIABLES_ATTR] = tuple(
        name
        for name in roles.observation
        if name in df.columns and name not in _MEMBER_COORDINATE_VARIABLES
    )
    df.attrs[_TRAJECTORY_METADATA_VARIABLES_ATTR] = tuple(
        name for name in roles.trajectory_metadata if name in df.columns
    )
    return df


def _collect_observation_variable_metadata(
    ds: xr.Dataset,
    roles: _OptionalVariableRoles,
) -> dict[str, dict[str, object]]:
    return {
        name: {
            key: ds[name].attrs[key]
            for key in ("units", "long_name", "standard_name")
            if key in ds[name].attrs
        }
        for name in roles.observation
        if name in ds.variables
    }


def _required_cached_trajectory_columns(
    roles: _OptionalVariableRoles,
) -> set[str]:
    has_wide_group_members = {"lon_1", "lat_1"}.issubset(roles.all)
    required = set(roles.all)
    if has_wide_group_members:
        required -= _MEMBER_COORDINATE_VARIABLES
        required.add("group_member")
    return required


def _required_cached_summary_columns(df: pd.DataFrame) -> set[str]:
    required = {
        name
        for name in df.attrs.get(_TRAJECTORY_METADATA_VARIABLES_ATTR, ())
        if name in df.columns
    }
    for name in df.attrs.get(_OBSERVATION_VARIABLES_ATTR, ()):
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name].dtype):
            required.update(
                {
                    f"{name}0",
                    f"{name}f",
                    f"{name}_min",
                    f"{name}_max",
                    f"{name}_mean",
                }
            )
    return required


def _expand_memberwise_rows(df: pd.DataFrame) -> pd.DataFrame:
    has_group_member = "group_member" in df.columns
    has_member_columns = ("lon_1" in df.columns) and ("lat_1" in df.columns)
    if has_group_member or (not has_member_columns):
        return df

    member_chunks: list[pd.DataFrame] = []
    for m in (1, 2, 3, 4, 5):
        lon_col = f"lon_{m}"
        lat_col = f"lat_{m}"
        if lon_col not in df.columns or lat_col not in df.columns:
            continue

        chunk = df.copy()
        if "group_size" in chunk.columns:
            chunk = chunk[chunk["group_size"] >= m]

        if chunk.empty:
            continue

        chunk["source_trajectory"] = chunk["trajectory"]
        chunk["trajectory"] = chunk["trajectory"].astype(str) + f"_m{m}"
        chunk["lon"] = chunk[lon_col]
        chunk["lat"] = chunk[lat_col]
        chunk["group_member"] = m
        member_chunks.append(chunk)

    if not member_chunks:
        return df

    out = pd.concat(member_chunks, ignore_index=True)
    drop_cols = [
        c
        for c in (
            "lon_1",
            "lat_1",
            "lon_2",
            "lat_2",
            "lon_3",
            "lat_3",
            "lon_4",
            "lat_4",
            "lon_5",
            "lat_5",
        )
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out.sort_values(["trajectory", "group_member", "obs"]).reset_index(drop=True)


def _table_path(
    cfg: PostprocessConfig,
    name: str,
) -> Path:
    return Path(cfg.output.output_dir) / f"{name}.{cfg.exports.table_format}"


def _load_exported_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format for '{path}'")


def get_trajectory_table(
    cfg: PostprocessConfig,
    context: dict,
) -> pd.DataFrame:
    """
    Return the trajectory table from:
    1. context
    2. exported file
    3. fresh computation
    """
    if "trajectory_table" in context:
        return context["trajectory_table"]

    path = _table_path(cfg, "trajectory_table")

    ds = open_parcels_dataset(cfg.dataset.input_path)
    try:
        schema = resolve_parcels_schema(ds, coordinates=cfg.dataset.coordinates)
        roles = _discover_optional_variables(ds, schema)
        context[OBSERVATION_VARIABLE_METADATA_CONTEXT_KEY] = (
            _collect_observation_variable_metadata(ds, roles)
        )
    finally:
        ds.close()

    required_cached_columns = _required_cached_trajectory_columns(roles)

    if path.exists():
        df = _expand_memberwise_rows(_load_exported_table(path))
        if required_cached_columns.issubset(df.columns):
            df = _attach_optional_variable_roles(df, roles)
            context["trajectory_table"] = df
            return df

    df = load_trajectory_table(
        cfg.dataset.input_path,
        coordinates=cfg.dataset.coordinates,
        truncate_stagnant=cfg.cleaning.truncate_stagnant,
        stagnant_tol=cfg.cleaning.stagnant_tol,
        stagnant_min_consecutive=cfg.cleaning.stagnant_min_consecutive,
        extra_vars=list(roles.all),
    )
    df = _expand_memberwise_rows(df)
    df = _attach_optional_variable_roles(df, roles)

    context["trajectory_table"] = df
    return df


def get_particle_summary(
    cfg: PostprocessConfig,
    context: dict,
) -> pd.DataFrame:
    """
    Return the particle summary from:
    1. context
    2. exported file
    3. fresh computation
    """
    if "particle_summary" in context:
        return context["particle_summary"]

    traj = get_trajectory_table(cfg, context)
    path = _table_path(cfg, "particle_summary")
    if path.exists():
        df = _load_exported_table(path)
        required_cached_columns = _required_cached_summary_columns(traj)
        if required_cached_columns.issubset(df.columns):
            context["particle_summary"] = df
            return df

    summary = build_particle_summary(traj)
    context["particle_summary"] = summary
    return summary
