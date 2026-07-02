from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels import __version__

from ..config.models import MeridionalExcursionConfig
from ..core.gridding import RegularGrid


_ANCHOR_COLUMNS = {
    "initial_position": ("lon0", "lat0"),
    "southmost_point": ("lon_at_lat_min", "lat_min"),
    "northmost_point": ("lon_at_lat_max", "lat_max"),
}


@dataclass(frozen=True)
class MeridionalExcursionResult:
    table: pd.DataFrame
    grid_table: pd.DataFrame
    dataset: xr.Dataset


def _empty_excursion_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trajectory",
            "n_obs",
            "time0",
            "lon0",
            "lat0",
            "lat_min",
            "lon_at_lat_min",
            "time_at_lat_min",
            "age_at_lat_min_days",
            "lat_max",
            "lon_at_lat_max",
            "time_at_lat_max",
            "age_at_lat_max_days",
            "timef",
            "lonf",
            "latf",
            "duration_days",
            "southward_excursion_deg",
            "northward_excursion_deg",
        ]
    )


def _empty_grid_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "variable",
            "anchor",
            "merge",
            "lon_bin",
            "lat_bin",
            "lon_center",
            "lat_center",
            "value",
            "count",
        ]
    )


def _normalize_longitude_to_grid(lon: float, grid: RegularGrid) -> float:
    width = float(grid.lon_max - grid.lon_min)
    if not np.isfinite(lon):
        return np.nan
    if width <= 0.0 or width > 360.0 + 1.0e-9:
        return lon

    wrapped = ((lon - grid.lon_min) % 360.0) + grid.lon_min
    if wrapped >= grid.lon_max:
        wrapped = np.nextafter(grid.lon_max, grid.lon_min)
    return float(wrapped)


def _normalize_longitudes_to_grid(values: pd.Series, grid: RegularGrid) -> pd.Series:
    return values.astype(float).map(lambda value: _normalize_longitude_to_grid(value, grid))


def _scalarize_identifier(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return _scalarize_identifier(value.item() if value.ndim == 0 else value.reshape(-1)[0])
        return tuple(_scalarize_identifier(v) for v in value.tolist())

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _scalarize_identifier(value[0])
        return tuple(_scalarize_identifier(v) for v in value)

    return value


def _prepare_trajectory(
    group: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    time_col: str,
    obs_col: str | None,
) -> pd.DataFrame:
    work = group.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.loc[
        work[lon_col].notna() & work[lat_col].notna() & work[time_col].notna()
    ].copy()

    if work.empty:
        return work

    sort_cols = [time_col]
    if obs_col is not None and obs_col in work.columns:
        sort_cols.append(obs_col)
    return work.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def compute_meridional_excursion_table(
    df: pd.DataFrame,
    *,
    min_duration_days: float | None = None,
    trajectory_col: str = "trajectory",
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    obs_col: str = "obs",
) -> pd.DataFrame:
    """
    Build one meridional-excursion row per trajectory.

    Ties for the same minimum or maximum latitude use the first occurrence after
    stable time/observation sorting.
    """
    required = [trajectory_col, lon_col, lat_col, time_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    group_cols = [trajectory_col]
    has_group_member = "group_member" in df.columns
    if has_group_member:
        group_cols.append("group_member")

    metadata_cols = [
        col
        for col in ("circle_id", "group_id", "group_size")
        if col in df.columns
    ]

    rows = []
    grouped = df.groupby(group_cols, sort=False, observed=False)
    for group_key, group in grouped:
        work = _prepare_trajectory(
            group,
            lon_col=lon_col,
            lat_col=lat_col,
            time_col=time_col,
            obs_col=obs_col if obs_col in df.columns else None,
        )
        if work.empty:
            continue

        first = work.iloc[0]
        last = work.iloc[-1]
        duration_days = float((last[time_col] - first[time_col]).total_seconds() / 86400.0)
        if min_duration_days is not None and duration_days < min_duration_days:
            continue

        lat = work[lat_col].to_numpy(dtype=float)
        idx_min = int(np.nanargmin(lat))
        idx_max = int(np.nanargmax(lat))
        row_min = work.iloc[idx_min]
        row_max = work.iloc[idx_max]

        if has_group_member:
            traj, group_member = group_key
        else:
            traj = group_key
            group_member = None

        age_min_days = float((row_min[time_col] - first[time_col]).total_seconds() / 86400.0)
        age_max_days = float((row_max[time_col] - first[time_col]).total_seconds() / 86400.0)

        row = {
            "trajectory": _scalarize_identifier(traj),
            "n_obs": int(len(work)),
            "time0": first[time_col],
            "lon0": float(first[lon_col]),
            "lat0": float(first[lat_col]),
            "lat_min": float(row_min[lat_col]),
            "lon_at_lat_min": float(row_min[lon_col]),
            "time_at_lat_min": row_min[time_col],
            "age_at_lat_min_days": age_min_days,
            "lat_max": float(row_max[lat_col]),
            "lon_at_lat_max": float(row_max[lon_col]),
            "time_at_lat_max": row_max[time_col],
            "age_at_lat_max_days": age_max_days,
            "timef": last[time_col],
            "lonf": float(last[lon_col]),
            "latf": float(last[lat_col]),
            "duration_days": duration_days,
            "southward_excursion_deg": float(first[lat_col] - row_min[lat_col]),
            "northward_excursion_deg": float(row_max[lat_col] - first[lat_col]),
        }

        if has_group_member:
            row["group_member"] = _scalarize_identifier(group_member)

        for col in metadata_cols:
            row[col] = first[col]

        rows.append(row)

    if not rows:
        return _empty_excursion_table()

    sort_cols = [col for col in group_cols if col in rows[0]]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def _merge_value_and_count_tables(
    value_table: pd.DataFrame,
    count_table: pd.DataFrame,
    *,
    variable: str,
    anchor: str,
    merge: str,
) -> pd.DataFrame:
    group_cols = ["lon_bin", "lat_bin", "lon_center", "lat_center"]
    if value_table.empty:
        out = _empty_grid_table()
        return out

    out = value_table.merge(
        count_table[group_cols + ["count"]],
        on=group_cols,
        how="left",
    )
    out.insert(0, "merge", merge)
    out.insert(0, "anchor", anchor)
    out.insert(0, "variable", variable)
    return out[
        [
            "variable",
            "anchor",
            "merge",
            "lon_bin",
            "lat_bin",
            "lon_center",
            "lat_center",
            "value",
            "count",
        ]
    ]


def compute_meridional_excursion_grid(
    table: pd.DataFrame,
    *,
    grid: RegularGrid,
    variables: tuple[str, ...],
    anchors: tuple[str, ...],
    merge: str,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Aggregate meridional-excursion values onto a regular lon/lat grid.
    """
    invalid_anchors = [anchor for anchor in anchors if anchor not in _ANCHOR_COLUMNS]
    if invalid_anchors:
        raise ValueError(
            f"Unsupported meridional excursion anchor(s): {invalid_anchors}. "
            f"Supported: {sorted(_ANCHOR_COLUMNS)}"
        )

    missing_variables = [var for var in variables if var not in table.columns]
    if missing_variables:
        raise KeyError(f"Excursion table missing configured variable(s): {missing_variables}")

    grid_tables = []
    data_vars = {}

    for anchor in anchors:
        lon_col, lat_col = _ANCHOR_COLUMNS[anchor]
        missing_anchor_cols = [col for col in (lon_col, lat_col) if col not in table.columns]
        if missing_anchor_cols:
            raise KeyError(f"Excursion table missing anchor column(s): {missing_anchor_cols}")

        work = table.copy()
        work["_grid_lon"] = _normalize_longitudes_to_grid(work[lon_col], grid)
        work["_grid_lat"] = work[lat_col].astype(float)

        for variable in variables:
            value_var = f"{variable}_at_{anchor}_{merge}"
            count_var = f"{variable}_at_{anchor}_count"

            value_table = grid.aggregate(
                work,
                value_col=variable,
                agg=merge,
                lon_col="_grid_lon",
                lat_col="_grid_lat",
                output_col="value",
            )
            count_table = grid.aggregate(
                work,
                value_col=variable,
                agg="count",
                lon_col="_grid_lon",
                lat_col="_grid_lat",
                output_col="count",
            )

            grid_tables.append(
                _merge_value_and_count_tables(
                    value_table,
                    count_table,
                    variable=variable,
                    anchor=anchor,
                    merge=merge,
                )
            )

            value_ds = grid.to_xarray(
                value_table,
                value_col="value",
                dataset_name=value_var,
            )
            count_ds = grid.to_xarray(
                count_table,
                value_col="count",
                dataset_name=count_var,
            )

            data_vars[value_var] = value_ds[value_var]
            data_vars[count_var] = count_ds[count_var]

    grid_table = (
        pd.concat(grid_tables, ignore_index=True)
        if grid_tables
        else _empty_grid_table()
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs={
            "title": "Meridional excursion gridded products",
            "summary": (
                "Per-trajectory meridional extrema and excursions aggregated onto "
                "a regular lon/lat grid. Variable names encode the coordinate anchor "
                "used for binning."
            ),
            "grid_type": "regular_lonlat",
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "dlon": grid.dlon,
            "dlat": grid.dlat,
            "merge": merge,
            "anchors": tuple(anchors),
            "variables": tuple(variables),
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "software_version": __version__,
        },
    )

    for anchor in anchors:
        for variable in variables:
            value_var = f"{variable}_at_{anchor}_{merge}"
            count_var = f"{variable}_at_{anchor}_count"
            if value_var in ds:
                ds[value_var].attrs.update(
                    source_variable=variable,
                    anchor=anchor,
                    aggregation=merge,
                )
            if count_var in ds:
                ds[count_var].attrs.update(
                    source_variable=variable,
                    anchor=anchor,
                    aggregation="count",
                    units="count",
                )

    return grid_table, ds


def compute_meridional_excursion(
    df: pd.DataFrame,
    *,
    grid: RegularGrid,
    cfg: MeridionalExcursionConfig,
    trajectory_col: str = "trajectory",
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    obs_col: str = "obs",
) -> MeridionalExcursionResult:
    table = compute_meridional_excursion_table(
        df,
        min_duration_days=cfg.min_duration_days,
        trajectory_col=trajectory_col,
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
        obs_col=obs_col,
    )
    grid_table, ds = compute_meridional_excursion_grid(
        table,
        grid=grid,
        variables=cfg.gridding.variables,
        anchors=cfg.gridding.over,
        merge=cfg.gridding.merge,
    )
    ds.attrs["min_duration_days"] = "none" if cfg.min_duration_days is None else cfg.min_duration_days
    return MeridionalExcursionResult(table=table, grid_table=grid_table, dataset=ds)
