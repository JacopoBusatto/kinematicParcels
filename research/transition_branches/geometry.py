"""Shared grid, angle, interpolation, and physical-scale geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def signed_angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return ``first - second`` wrapped to [-180, 180) degrees."""
    return (
        np.remainder(
            np.asarray(first, dtype=float) - np.asarray(second, dtype=float) + 180.0,
            360.0,
        )
        - 180.0
    )


def grid_array(cells: pd.DataFrame, grid: Any, field: str) -> np.ndarray:
    """Place a cell-table field on its configured two-dimensional grid."""
    values = np.full((grid.nlat, grid.nlon), np.nan, dtype=float)
    valid = cells.lon_bin.between(0, grid.nlon - 1) & cells.lat_bin.between(
        0, grid.nlat - 1
    )
    rows = cells.loc[valid]
    values[rows.lat_bin.to_numpy(np.int64), rows.lon_bin.to_numpy(np.int64)] = rows[
        field
    ].to_numpy(float)
    return values


def support_aware_uniform_3x3(
    values: np.ndarray,
    support: np.ndarray,
    *,
    periodic_longitude: bool,
) -> np.ndarray:
    """Mild uniform smoothing that never fills an unsupported focal cell."""
    values = np.asarray(values, dtype=float)
    support = np.asarray(support, dtype=bool) & np.isfinite(values)
    numerator = np.zeros_like(values, dtype=float)
    denominator = np.zeros_like(values, dtype=float)
    nlat, nlon = values.shape
    for delta_lat in (-1, 0, 1):
        source_lat_start = max(0, -delta_lat)
        source_lat_stop = min(nlat, nlat - delta_lat)
        target_lat_start = source_lat_start + delta_lat
        target_lat_stop = source_lat_stop + delta_lat
        source_values = values[source_lat_start:source_lat_stop]
        source_support = support[source_lat_start:source_lat_stop]
        for delta_lon in (-1, 0, 1):
            if periodic_longitude:
                shifted_values = np.roll(source_values, delta_lon, axis=1)
                shifted_support = np.roll(source_support, delta_lon, axis=1)
                numerator[target_lat_start:target_lat_stop] += np.where(
                    shifted_support, shifted_values, 0.0
                )
                denominator[target_lat_start:target_lat_stop] += shifted_support
            else:
                source_lon_start = max(0, -delta_lon)
                source_lon_stop = min(nlon, nlon - delta_lon)
                target_lon_start = source_lon_start + delta_lon
                target_lon_stop = source_lon_stop + delta_lon
                selected_values = source_values[:, source_lon_start:source_lon_stop]
                selected_support = source_support[:, source_lon_start:source_lon_stop]
                numerator[
                    target_lat_start:target_lat_stop,
                    target_lon_start:target_lon_stop,
                ] += np.where(selected_support, selected_values, 0.0)
                denominator[
                    target_lat_start:target_lat_stop,
                    target_lon_start:target_lon_stop,
                ] += selected_support
    smoothed = np.full_like(values, np.nan, dtype=float)
    valid = support & (denominator > 0)
    smoothed[valid] = numerator[valid] / denominator[valid]
    return smoothed


def bilinear_supported_sample(
    values: np.ndarray,
    support: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    grid: Any,
    *,
    weight_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate only when every nonzero-weight corner is supported."""
    target_lon = np.asarray(target_lon, dtype=float)
    target_lat = np.asarray(target_lat, dtype=float)
    sampled = np.full(target_lon.shape, np.nan, dtype=float)
    boundary = np.zeros(target_lon.shape, dtype=bool)
    missing = np.zeros(target_lon.shape, dtype=bool)
    span = grid.lon_max - grid.lon_min
    for index, (lon, lat) in enumerate(zip(target_lon, target_lat)):
        if not np.isfinite(lon) or not np.isfinite(lat):
            missing[index] = True
            continue
        if grid.periodic_longitude:
            lon = ((lon - grid.lon_min) % span) + grid.lon_min
        x = (lon - grid.lon_min) / grid.dlon - 0.5
        y = (lat - grid.lat_min) / grid.dlat - 0.5
        if y < -weight_tolerance or y > grid.nlat - 1 + weight_tolerance:
            boundary[index] = True
            continue
        if not grid.periodic_longitude and (
            x < -weight_tolerance or x > grid.nlon - 1 + weight_tolerance
        ):
            boundary[index] = True
            continue
        if not grid.periodic_longitude:
            x = min(max(x, 0.0), grid.nlon - 1.0)
        y = min(max(y, 0.0), grid.nlat - 1.0)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        candidates = (
            (y0, x0, (1.0 - fx) * (1.0 - fy)),
            (y0, x0 + 1, fx * (1.0 - fy)),
            (y0 + 1, x0, (1.0 - fx) * fy),
            (y0 + 1, x0 + 1, fx * fy),
        )
        weighted_value = 0.0
        defensible = True
        for lat_index, lon_index, weight in candidates:
            if weight <= weight_tolerance:
                continue
            if lat_index < 0 or lat_index >= grid.nlat:
                boundary[index] = True
                defensible = False
                break
            if grid.periodic_longitude:
                lon_index %= grid.nlon
            elif lon_index < 0 or lon_index >= grid.nlon:
                boundary[index] = True
                defensible = False
                break
            if not support[lat_index, lon_index] or not np.isfinite(
                values[lat_index, lon_index]
            ):
                missing[index] = True
                defensible = False
                break
            weighted_value += weight * values[lat_index, lon_index]
        if defensible:
            sampled[index] = weighted_value
    return sampled, boundary, missing


def physical_cell_scales(
    cells: pd.DataFrame,
    grid: Any,
    geod: Geod,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zonal, meridional, and geometric-mean cell scales in metres."""
    lon = cells.lon.to_numpy(float)
    lat = cells.lat.to_numpy(float)
    _, _, zonal_m = geod.inv(lon - grid.dlon / 2.0, lat, lon + grid.dlon / 2.0, lat)
    _, _, meridional_m = geod.inv(
        lon, lat - grid.dlat / 2.0, lon, lat + grid.dlat / 2.0
    )
    effective_m = np.sqrt(np.asarray(zonal_m) * np.asarray(meridional_m))
    return np.asarray(zonal_m), np.asarray(meridional_m), effective_m


# Private aliases keep the verified Stage-5/6/7 implementations numerically intact.
_signed_difference = signed_angle_difference
_grid_array = grid_array
_bilinear_supported_sample = bilinear_supported_sample
_physical_cell_scales = physical_cell_scales
