from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0


def wrap_lon_delta_deg(lon: np.ndarray | float, target_lon: float) -> np.ndarray:
    """
    Return longitude differences wrapped to [-180, 180) degrees.
    """
    return (np.asarray(lon, dtype=float) - target_lon + 180.0) % 360.0 - 180.0


def haversine_km(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """
    Great-circle distance in kilometers.
    """
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def meridional_distance_km(
    lat1: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """
    North-south distance in kilometers.
    """
    return EARTH_RADIUS_KM * np.abs(np.radians(lat2) - np.radians(lat1))


def local_equirectangular_xy_km(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    *,
    lon0: float,
    lat0: float,
) -> np.ndarray:
    """
    Project lon/lat to local equirectangular x/y coordinates in kilometers.

    This approximation is intended for regional domains and local neighbor
    searches or explicitly approximate Euclidean diagnostics.
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    x = EARTH_RADIUS_KM * np.radians(wrap_lon_delta_deg(lon_arr, lon0)) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_KM * np.radians(lat_arr - lat0)
    return np.column_stack([x, y])


def local_euclidean_km(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
    *,
    lon0: float,
    lat0: float,
) -> np.ndarray:
    """
    Local equirectangular Euclidean distance in kilometers.
    """
    xy1 = local_equirectangular_xy_km(lon1, lat1, lon0=lon0, lat0=lat0)
    xy2 = local_equirectangular_xy_km(lon2, lat2, lon0=lon0, lat0=lat0)
    return np.linalg.norm(xy2 - xy1, axis=1)
