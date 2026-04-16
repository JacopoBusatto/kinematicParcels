"""
Spherical geometry utilities for LKM implementation.

Provides functions for computing group centers on the sphere and projecting
to local tangent-plane coordinates.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_group_center_spherical(
    lons_deg: np.ndarray,
    lats_deg: np.ndarray,
    R_earth: float = 6371000.0,
) -> Tuple[float, float]:
    """
    Compute spherical group center by Cartesian averaging and renormalization.

    This avoids issues with longitude wrapping and provides a geometrically
    consistent center for groups.

    Parameters
    ----------
    lons_deg, lats_deg : array-like
        Particle longitudes and latitudes in degrees
    R_earth : float
        Earth radius in meters (default: 6371 km)

    Returns
    -------
    lon_center_rad, lat_center_rad : float
        Group center in radians
    """
    lons_rad = np.radians(lons_deg)
    lats_rad = np.radians(lats_deg)

    # Convert to Earth-centered Cartesian coordinates
    X = R_earth * np.cos(lats_rad) * np.cos(lons_rad)
    Y = R_earth * np.cos(lats_rad) * np.sin(lons_rad)
    Z = R_earth * np.sin(lats_rad)

    # Average in Cartesian space
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    Z_mean = np.mean(Z)

    # Renormalize to sphere
    norm = np.sqrt(X_mean**2 + Y_mean**2 + Z_mean**2)
    X_c = R_earth * X_mean / norm
    Y_c = R_earth * Y_mean / norm
    Z_c = R_earth * Z_mean / norm

    # Convert back to geographic coordinates
    lon_center_rad = np.arctan2(Y_c, X_c)
    lat_center_rad = np.arcsin(Z_c / R_earth)

    return lon_center_rad, lat_center_rad


def build_tangent_plane_basis(
    lon_center_rad: float,
    lat_center_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build orthonormal east-north basis vectors at the group center.

    Returns unit vectors pointing east and north in Cartesian coordinates.

    Parameters
    ----------
    lon_center_rad, lat_center_rad : float
        Center location in radians

    Returns
    -------
    e_E, e_N : ndarray
        East and north unit vectors (shape: (3,))
    """
    # Radial vector (up from Earth's surface)
    e_R = np.array([
        np.cos(lat_center_rad) * np.cos(lon_center_rad),
        np.cos(lat_center_rad) * np.sin(lon_center_rad),
        np.sin(lat_center_rad)
    ])

    # East vector (tangent, perpendicular to meridian)
    e_E = np.array([
        -np.sin(lon_center_rad),
        np.cos(lon_center_rad),
        0.0
    ])

    # North vector (tangent, perpendicular to parallel)
    e_N = np.array([
        -np.sin(lat_center_rad) * np.cos(lon_center_rad),
        -np.sin(lat_center_rad) * np.sin(lon_center_rad),
        np.cos(lat_center_rad)
    ])

    return e_E, e_N


def project_to_tangent_plane(
    lon_i_deg: float,
    lat_i_deg: float,
    lon_center_rad: float,
    lat_center_rad: float,
    e_E: np.ndarray,
    e_N: np.ndarray,
    R_earth: float = 6371000.0,
) -> Tuple[float, float]:
    """
    Project particle position to local tangent-plane coordinates.

    Parameters
    ----------
    lon_i_deg, lat_i_deg : float
        Particle position in degrees
    lon_center_rad, lat_center_rad : float
        Group center in radians
    e_E, e_N : ndarray
        East and north basis vectors
    R_earth : float
        Earth radius in meters

    Returns
    -------
    x_rel_m, y_rel_m : float
        Relative coordinates in meters (east, north from center)
    """
    # Convert particle to Cartesian
    lon_i_rad = np.radians(lon_i_deg)
    lat_i_rad = np.radians(lat_i_deg)

    X_i = R_earth * np.cos(lat_i_rad) * np.cos(lon_i_rad)
    Y_i = R_earth * np.cos(lat_i_rad) * np.sin(lon_i_rad)
    Z_i = R_earth * np.sin(lat_i_rad)

    # Center in Cartesian
    X_c = R_earth * np.cos(lat_center_rad) * np.cos(lon_center_rad)
    Y_c = R_earth * np.cos(lat_center_rad) * np.sin(lon_center_rad)
    Z_c = R_earth * np.sin(lat_center_rad)

    # Displacement vector
    dX = X_i - X_c
    dY = Y_i - Y_c
    dZ = Z_i - Z_c

    # Project onto tangent-plane basis
    x_rel_m = np.dot([dX, dY, dZ], e_E)
    y_rel_m = np.dot([dX, dY, dZ], e_N)

    return x_rel_m, y_rel_m


def full_group_geom_to_local(
    lons_deg: np.ndarray,
    lats_deg: np.ndarray,
    group_ids: np.ndarray,
    R_earth: float = 6371000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute group centers and local coordinates for all particles.

    Parameters
    ----------
    lons_deg, lats_deg : ndarray
        All particle positions in degrees
    group_ids : ndarray
        Group ID for each particle
    R_earth : float
        Earth radius in meters

    Returns
    -------
    center_lons_rad, center_lats_rad : ndarray
        Group centers in radians (one per particle, repeated for group)
    x_rel_m, y_rel_m : ndarray
        Local coordinates in meters (one per particle)
    """
    unique_groups = np.unique(group_ids)
    n_particles = len(lons_deg)

    # Initialize output arrays
    center_lons_rad = np.zeros(n_particles)
    center_lats_rad = np.zeros(n_particles)
    x_rel_m = np.zeros(n_particles)
    y_rel_m = np.zeros(n_particles)

    for group_id in unique_groups:
        # Get particles in this group
        mask = group_ids == group_id
        group_lons = lons_deg[mask]
        group_lats = lats_deg[mask]
        group_indices = np.where(mask)[0]

        # Compute spherical center
        lon_c_rad, lat_c_rad = compute_group_center_spherical(group_lons, group_lats, R_earth)

        # Build tangent-plane basis
        e_E, e_N = build_tangent_plane_basis(lon_c_rad, lat_c_rad)

        # Project each particle to local coordinates
        for idx in group_indices:
            x_rel, y_rel = project_to_tangent_plane(
                lons_deg[idx], lats_deg[idx],
                lon_c_rad, lat_c_rad,
                e_E, e_N, R_earth
            )
            x_rel_m[idx] = x_rel
            y_rel_m[idx] = y_rel

        # Store center for all particles in group
        center_lons_rad[mask] = lon_c_rad
        center_lats_rad[mask] = lat_c_rad

    return center_lons_rad, center_lats_rad, x_rel_m, y_rel_m