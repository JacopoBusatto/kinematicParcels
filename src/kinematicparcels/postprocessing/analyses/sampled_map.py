from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from scipy.spatial import cKDTree

from kinematicparcels import __version__

from ..config.models import SampledMapConfig, SampledMapVariableConfig
from ..core.gridding import RegularGrid


_GAUSSIAN_CUTOFF_SIGMA = 3.0
_WGS84 = Geod(ellps="WGS84")
_CANONICAL_TRAJECTORY_COLUMNS = {"trajectory", "obs", "time", "lon", "lat", "z"}


@dataclass(frozen=True)
class SampledMapResult:
    table: pd.DataFrame
    dataset: xr.Dataset


def _normalize_longitudes_to_grid(
    values: pd.Series,
    grid: RegularGrid,
) -> pd.Series:
    width = float(grid.lon_max - grid.lon_min)
    lon = values.astype(float).to_numpy(copy=True)
    if width <= 360.0 + 1.0e-9:
        finite = np.isfinite(lon)
        wrapped = ((lon[finite] - grid.lon_min) % 360.0) + grid.lon_min
        inside_wrapped = wrapped < grid.lon_max
        finite_indexes = np.flatnonzero(finite)
        lon[finite_indexes[inside_wrapped]] = wrapped[inside_wrapped]
    return pd.Series(lon, index=values.index, dtype=float)


def _full_grid_table(grid: RegularGrid) -> pd.DataFrame:
    lon_bin = np.tile(np.arange(grid.nlon, dtype=np.int64), grid.nlat)
    lat_bin = np.repeat(np.arange(grid.nlat, dtype=np.int64), grid.nlon)
    return pd.DataFrame(
        {
            "lon_bin": lon_bin,
            "lat_bin": lat_bin,
            "lon_center": grid.lon_min + (lon_bin + 0.5) * grid.dlon,
            "lat_center": grid.lat_min + (lat_bin + 0.5) * grid.dlat,
        }
    )


def _filtered_binned_points(
    df: pd.DataFrame,
    *,
    variable: str,
    variable_cfg: SampledMapVariableConfig,
    grid: RegularGrid,
) -> pd.DataFrame:
    work = df[["trajectory", "lon", "lat", variable]].copy()
    work["_grid_lon"] = _normalize_longitudes_to_grid(work["lon"], grid)
    work["_grid_lat"] = work["lat"].astype(float)

    values = work[variable].to_numpy(dtype=float)
    lon = work["_grid_lon"].to_numpy(dtype=float)
    lat = work["_grid_lat"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
    if variable_cfg.valid_min is not None:
        valid &= values >= variable_cfg.valid_min
    if variable_cfg.valid_max is not None:
        valid &= values <= variable_cfg.valid_max

    work = work.loc[valid].copy()
    if work.empty:
        return grid.assign_bins(
            work,
            lon_col="_grid_lon",
            lat_col="_grid_lat",
            drop_outside=True,
        )

    return grid.assign_bins(
        work,
        lon_col="_grid_lon",
        lat_col="_grid_lat",
        drop_outside=True,
    )


def _aggregate_variable(
    df: pd.DataFrame,
    *,
    variable: str,
    variable_cfg: SampledMapVariableConfig,
    weighting: str,
    grid: RegularGrid,
) -> dict[str, np.ndarray]:
    shape = (grid.nlat, grid.nlon)
    point_count = np.zeros(shape, dtype=np.int64)
    trajectory_count = np.zeros(shape, dtype=np.int64)
    mean = np.full(shape, np.nan, dtype=float)
    std = np.full(shape, np.nan, dtype=float)

    binned = _filtered_binned_points(
        df,
        variable=variable,
        variable_cfg=variable_cfg,
        grid=grid,
    )
    if binned.empty:
        return {
            "point_count": point_count,
            "trajectory_count": trajectory_count,
            "mean": mean,
            "std": std,
        }

    cell_keys = ["lat_bin", "lon_bin"]
    grouped = binned.groupby(cell_keys, sort=True, observed=False)
    counts = grouped.agg(
        point_count=(variable, "size"),
        trajectory_count=("trajectory", "nunique"),
    )

    if weighting == "points":
        statistics = grouped[variable].agg(mean="mean", std="std")
    else:
        per_trajectory = (
            binned.groupby(
                cell_keys + ["trajectory"], sort=True, observed=False
            )[variable]
            .mean()
            .rename("trajectory_mean")
            .reset_index()
        )
        statistics = per_trajectory.groupby(
            cell_keys, sort=True, observed=False
        )["trajectory_mean"].agg(mean="mean", std="std")

    aggregated = counts.join(statistics, how="left").reset_index()
    lat_idx = aggregated["lat_bin"].to_numpy(dtype=np.int64)
    lon_idx = aggregated["lon_bin"].to_numpy(dtype=np.int64)
    indexes = (lat_idx, lon_idx)
    point_count[indexes] = aggregated["point_count"].to_numpy(dtype=np.int64)
    trajectory_count[indexes] = aggregated["trajectory_count"].to_numpy(
        dtype=np.int64
    )
    mean[indexes] = aggregated["mean"].to_numpy(dtype=float)
    std[indexes] = aggregated["std"].to_numpy(dtype=float)

    supported = (
        (point_count >= variable_cfg.minimum_point_count)
        & (trajectory_count >= variable_cfg.minimum_trajectory_count)
    )
    mean[~supported] = np.nan
    std[~supported] = np.nan
    return {
        "point_count": point_count,
        "trajectory_count": trajectory_count,
        "mean": mean,
        "std": std,
    }


def _cell_areas_km2(grid: RegularGrid, geod: Geod) -> np.ndarray:
    row_areas = np.empty(grid.nlat, dtype=float)
    for lat_bin in range(grid.nlat):
        south = max(-90.0, float(grid.lat_edges[lat_bin]))
        north = min(90.0, float(grid.lat_edges[lat_bin + 1]))
        area_m2, _ = geod.polygon_area_perimeter(
            [0.0, grid.dlon, grid.dlon, 0.0],
            [south, south, north, north],
        )
        row_areas[lat_bin] = abs(float(area_m2)) / 1.0e6
    return np.broadcast_to(row_areas[:, None], (grid.nlat, grid.nlon)).copy()


def _unit_sphere_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        )
    )


def gaussian_smooth_supported_field(
    field: np.ndarray,
    *,
    grid: RegularGrid,
    sigma_km: float,
    geod: Geod = _WGS84,
    cutoff_sigma: float = _GAUSSIAN_CUTOFF_SIGMA,
) -> np.ndarray:
    """Area-weighted geodesic Gaussian smoothing without filling missing cells."""
    field = np.asarray(field, dtype=float)
    expected_shape = (grid.nlat, grid.nlon)
    if field.shape != expected_shape:
        raise ValueError(
            f"Field shape {field.shape} does not match grid shape {expected_shape}."
        )
    if not np.isfinite(sigma_km) or sigma_km <= 0:
        raise ValueError("sigma_km must be positive.")
    if not np.isfinite(cutoff_sigma) or cutoff_sigma <= 0:
        raise ValueError("cutoff_sigma must be positive.")

    output = np.full_like(field, np.nan, dtype=float)
    supported_bins = np.argwhere(np.isfinite(field))
    if supported_bins.size == 0:
        return output

    lat_idx = supported_bins[:, 0]
    lon_idx = supported_bins[:, 1]
    source_lon = grid.lon_centers[lon_idx]
    source_lat = grid.lat_centers[lat_idx]
    source_xyz = _unit_sphere_xyz(source_lon, source_lat)
    source_values = field[lat_idx, lon_idx]
    source_areas = _cell_areas_km2(grid, geod)[lat_idx, lon_idx]

    tree = cKDTree(source_xyz)
    cutoff_km = cutoff_sigma * sigma_km
    # Use a slightly conservative reference radius for candidate discovery;
    # WGS84 geodesic distances below provide the exact final cutoff.
    angular_cutoff = cutoff_km / 6350.0
    if angular_cutoff >= np.pi:
        candidate_lists = [list(range(len(source_xyz)))] * len(source_xyz)
    else:
        chord_radius = 2.0 * np.sin(0.5 * angular_cutoff)
        candidate_lists = tree.query_ball_point(source_xyz, r=chord_radius)

    for target_index, candidates in enumerate(candidate_lists):
        candidate_idx = np.asarray(candidates, dtype=np.int64)
        if candidate_idx.size == 0:
            continue
        target_lon = float(source_lon[target_index])
        target_lat = float(source_lat[target_index])
        _, _, distance_m = geod.inv(
            np.full(candidate_idx.size, target_lon),
            np.full(candidate_idx.size, target_lat),
            source_lon[candidate_idx],
            source_lat[candidate_idx],
        )
        distance_km = np.asarray(distance_m, dtype=float) / 1000.0
        inside = distance_km <= cutoff_km
        if not np.any(inside):
            continue
        candidate_idx = candidate_idx[inside]
        distance_km = distance_km[inside]
        weights = (
            np.exp(-0.5 * np.square(distance_km / sigma_km))
            * source_areas[candidate_idx]
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            continue
        output[lat_idx[target_index], lon_idx[target_index]] = float(
            np.sum(weights * source_values[candidate_idx]) / weight_sum
        )

    return output


def _neighbor(
    array: np.ndarray,
    *,
    axis: int,
    direction: int,
    periodic: bool,
) -> np.ndarray:
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1.")
    if axis == 1 and periodic:
        return np.roll(array, -direction, axis=axis)

    output = np.full(array.shape, np.nan, dtype=float)
    if axis == 0:
        if direction > 0:
            output[:-1, :] = array[1:, :]
        else:
            output[1:, :] = array[:-1, :]
    elif axis == 1:
        if direction > 0:
            output[:, :-1] = array[:, 1:]
        else:
            output[:, 1:] = array[:, :-1]
    else:
        raise ValueError("axis must be 0 or 1.")
    return output


def _geodesic_distance_km(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    *,
    geod: Geod,
) -> np.ndarray:
    _, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)
    return np.asarray(distance_m, dtype=float) / 1000.0


def compute_sphere_aware_gradients(
    field: np.ndarray,
    *,
    grid: RegularGrid,
    geod: Geod = _WGS84,
) -> dict[str, np.ndarray]:
    """Differentiate adjacent grid centers using exact WGS84 distances."""
    field = np.asarray(field, dtype=float)
    expected_shape = (grid.nlat, grid.nlon)
    if field.shape != expected_shape:
        raise ValueError(
            f"Field shape {field.shape} does not match grid shape {expected_shape}."
        )

    lon2d, lat2d = np.meshgrid(grid.lon_centers, grid.lat_centers)
    periodic_lon = bool(
        grid.nlon > 2
        and np.isclose(
            grid.lon_max - grid.lon_min,
            360.0,
            rtol=0.0,
            atol=max(1.0e-9, abs(grid.dlon) * 1.0e-8),
        )
    )

    east = _neighbor(field, axis=1, direction=1, periodic=periodic_lon)
    west = _neighbor(field, axis=1, direction=-1, periodic=periodic_lon)
    north = _neighbor(field, axis=0, direction=1, periodic=False)
    south = _neighbor(field, axis=0, direction=-1, periodic=False)

    east_lon = _neighbor(lon2d, axis=1, direction=1, periodic=periodic_lon)
    west_lon = _neighbor(lon2d, axis=1, direction=-1, periodic=periodic_lon)
    north_lat = _neighbor(lat2d, axis=0, direction=1, periodic=False)
    south_lat = _neighbor(lat2d, axis=0, direction=-1, periodic=False)

    dx_centered = _geodesic_distance_km(
        west_lon, lat2d, east_lon, lat2d, geod=geod
    )
    dx_east = _geodesic_distance_km(
        lon2d, lat2d, east_lon, lat2d, geod=geod
    )
    dx_west = _geodesic_distance_km(
        west_lon, lat2d, lon2d, lat2d, geod=geod
    )
    dy_centered = _geodesic_distance_km(
        lon2d, south_lat, lon2d, north_lat, geod=geod
    )
    dy_north = _geodesic_distance_km(
        lon2d, lat2d, lon2d, north_lat, geod=geod
    )
    dy_south = _geodesic_distance_km(
        lon2d, south_lat, lon2d, lat2d, geod=geod
    )

    finite_center = np.isfinite(field)
    east_ok = np.isfinite(east)
    west_ok = np.isfinite(west)
    north_ok = np.isfinite(north)
    south_ok = np.isfinite(south)

    zonal = np.full_like(field, np.nan, dtype=float)
    zonal_distance = np.full_like(field, np.nan, dtype=float)
    centered = finite_center & east_ok & west_ok & (dx_centered > 0.0)
    east_only = finite_center & east_ok & ~west_ok & (dx_east > 0.0)
    west_only = finite_center & west_ok & ~east_ok & (dx_west > 0.0)
    zonal[centered] = (east[centered] - west[centered]) / dx_centered[centered]
    zonal[east_only] = (east[east_only] - field[east_only]) / dx_east[east_only]
    zonal[west_only] = (field[west_only] - west[west_only]) / dx_west[west_only]
    zonal_distance[centered] = dx_centered[centered]
    zonal_distance[east_only] = dx_east[east_only]
    zonal_distance[west_only] = dx_west[west_only]

    meridional = np.full_like(field, np.nan, dtype=float)
    meridional_distance = np.full_like(field, np.nan, dtype=float)
    centered = finite_center & north_ok & south_ok & (dy_centered > 0.0)
    north_only = finite_center & north_ok & ~south_ok & (dy_north > 0.0)
    south_only = finite_center & south_ok & ~north_ok & (dy_south > 0.0)
    meridional[centered] = (
        north[centered] - south[centered]
    ) / dy_centered[centered]
    meridional[north_only] = (
        north[north_only] - field[north_only]
    ) / dy_north[north_only]
    meridional[south_only] = (
        field[south_only] - south[south_only]
    ) / dy_south[south_only]
    meridional_distance[centered] = dy_centered[centered]
    meridional_distance[north_only] = dy_north[north_only]
    meridional_distance[south_only] = dy_south[south_only]

    magnitude = np.full_like(field, np.nan, dtype=float)
    complete = np.isfinite(zonal) & np.isfinite(meridional)
    magnitude[complete] = np.hypot(zonal[complete], meridional[complete])
    return {
        "zonal_gradient": zonal,
        "meridional_gradient": meridional,
        "gradient_magnitude": magnitude,
        "zonal_gradient_distance_km": zonal_distance,
        "meridional_gradient_distance_km": meridional_distance,
    }


def _source_metadata(
    variable: str,
    variable_metadata: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, str]:
    if variable_metadata is None:
        return {}
    raw = variable_metadata.get(variable, {})
    return {
        key: str(raw[key])
        for key in ("units", "long_name", "standard_name")
        if key in raw and raw[key] is not None and str(raw[key]).strip()
    }


def _attach_variable_metadata(
    ds: xr.Dataset,
    *,
    variable: str,
    variable_cfg: SampledMapVariableConfig,
    weighting: str,
    source_metadata: Mapping[str, str],
    gradients_enabled: bool,
    smoothing_sigma_km: float | None,
) -> None:
    source_units = source_metadata.get("units")
    source_long_name = source_metadata.get("long_name", variable)
    source_standard_name = source_metadata.get("standard_name")

    common = {
        "source_variable": variable,
        "source_long_name": source_long_name,
    }
    if source_units is not None:
        common["source_units"] = source_units
    if source_standard_name is not None:
        common["source_standard_name"] = source_standard_name

    mean_attrs = {
        **common,
        "long_name": f"Grid-cell mean of {source_long_name}",
        "aggregation_weighting": weighting,
    }
    if source_units is not None:
        mean_attrs["units"] = source_units
    if source_standard_name is not None:
        mean_attrs["standard_name"] = source_standard_name
    ds[f"{variable}_mean"].attrs.update(mean_attrs)

    std_attrs = {
        **common,
        "long_name": f"Grid-cell sample standard deviation of {source_long_name}",
        "aggregation_weighting": weighting,
        "ddof": 1,
    }
    if source_units is not None:
        std_attrs["units"] = source_units
    ds[f"{variable}_std"].attrs.update(std_attrs)

    for count_name, long_name in (
        ("point_count", "Valid observation-point count"),
        ("trajectory_count", "Distinct trajectory count"),
    ):
        ds[f"{variable}_{count_name}"].attrs.update(
            {
                **common,
                "long_name": f"{long_name} for {source_long_name}",
                "units": "count",
            }
        )

    for name in ("mean", "std"):
        ds[f"{variable}_{name}"].attrs.update(
            minimum_point_count=variable_cfg.minimum_point_count,
            minimum_trajectory_count=variable_cfg.minimum_trajectory_count,
            valid_min=(
                "none" if variable_cfg.valid_min is None else variable_cfg.valid_min
            ),
            valid_max=(
                "none" if variable_cfg.valid_max is None else variable_cfg.valid_max
            ),
        )

    if not gradients_enabled:
        return

    smoothed_attrs = {
        **common,
        "long_name": f"Gaussian-smoothed grid-cell mean of {source_long_name}",
        "smoothing_sigma_km": float(smoothing_sigma_km),
        "smoothing_cutoff_sigma": _GAUSSIAN_CUTOFF_SIGMA,
        "smoothing_support": "raw supported cells only",
    }
    if source_units is not None:
        smoothed_attrs["units"] = source_units
    ds[f"{variable}_smoothed_mean"].attrs.update(smoothed_attrs)

    gradient_units = f"{source_units} km-1" if source_units else "km-1"
    gradient_names = {
        "zonal_gradient": "Eastward gradient",
        "meridional_gradient": "Northward gradient",
        "gradient_magnitude": "Horizontal gradient magnitude",
    }
    for suffix, label in gradient_names.items():
        ds[f"{variable}_{suffix}"].attrs.update(
            {
                **common,
                "long_name": f"{label} of smoothed {source_long_name}",
                "units": gradient_units,
                "distance_units": "km",
                "smoothing_sigma_km": float(smoothing_sigma_km),
            }
        )

    for suffix, label in (
        ("zonal_gradient_distance_km", "Zonal differentiation distance"),
        (
            "meridional_gradient_distance_km",
            "Meridional differentiation distance",
        ),
    ):
        ds[f"{variable}_{suffix}"].attrs.update(
            {
                **common,
                "long_name": label,
                "units": "km",
            }
        )


def compute_sampled_map(
    df: pd.DataFrame,
    *,
    grid: RegularGrid,
    cfg: SampledMapConfig,
    variable_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> SampledMapResult:
    """Aggregate configured observation variables and derive gridded gradients."""
    required = {"trajectory", "lon", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Sampled-map input missing required columns: {missing}")

    work = df.copy()
    if cfg.max_group_member is not None and "group_member" in work.columns:
        work = work.loc[work["group_member"] <= cfg.max_group_member].copy()

    table = _full_grid_table(grid)
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}

    for variable, variable_cfg in cfg.variables.items():
        if variable in _CANONICAL_TRAJECTORY_COLUMNS:
            raise ValueError(
                f"Configured sampled_map variable {variable!r} is canonical; "
                "sampled_map variables must be additional observation fields."
            )
        if variable not in work.columns:
            raise KeyError(
                f"Configured sampled_map variable {variable!r} is not present "
                "in the trajectory table."
            )
        if (
            not pd.api.types.is_numeric_dtype(work[variable].dtype)
            or pd.api.types.is_bool_dtype(work[variable].dtype)
        ):
            raise TypeError(
                f"Configured sampled_map variable {variable!r} must be numeric."
            )

        fields = _aggregate_variable(
            work,
            variable=variable,
            variable_cfg=variable_cfg,
            weighting=cfg.weighting,
            grid=grid,
        )
        if cfg.gradients.enabled:
            smoothed = gaussian_smooth_supported_field(
                fields["mean"],
                grid=grid,
                sigma_km=float(cfg.gradients.smoothing_sigma_km),
            )
            fields["smoothed_mean"] = smoothed
            fields.update(compute_sphere_aware_gradients(smoothed, grid=grid))

        for suffix, values in fields.items():
            output_name = f"{variable}_{suffix}"
            table[output_name] = values.reshape(-1)
            data_vars[output_name] = (("lat", "lon"), values)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lat": grid.lat_centers, "lon": grid.lon_centers},
        attrs={
            "title": "Sampled trajectory-observation maps",
            "summary": (
                "Configured trajectory observations aggregated into regular "
                "longitude/latitude cells."
            ),
            "grid_type": "regular_lonlat",
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "dlon": grid.dlon,
            "dlat": grid.dlat,
            "variables": tuple(cfg.variables),
            "weighting": cfg.weighting,
            "standard_deviation_ddof": 1,
            "gradients_enabled": cfg.gradients.enabled,
            "gradient_geometry": "WGS84 geodesic adjacent-cell differences",
            "gradient_missing_policy": "centered then one-sided; never zero-filled",
            "smoothing_support": "raw supported cells only",
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "software_version": __version__,
        },
    )
    if cfg.gradients.enabled:
        ds.attrs.update(
            smoothing_method="area-weighted WGS84-distance Gaussian",
            smoothing_sigma_km=float(cfg.gradients.smoothing_sigma_km),
            smoothing_cutoff_sigma=_GAUSSIAN_CUTOFF_SIGMA,
        )

    for variable, variable_cfg in cfg.variables.items():
        _attach_variable_metadata(
            ds,
            variable=variable,
            variable_cfg=variable_cfg,
            weighting=cfg.weighting,
            source_metadata=_source_metadata(variable, variable_metadata),
            gradients_enabled=cfg.gradients.enabled,
            smoothing_sigma_km=cfg.gradients.smoothing_sigma_km,
        )

    return SampledMapResult(table=table, dataset=ds)
