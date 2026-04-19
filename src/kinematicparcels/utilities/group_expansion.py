"""
Grouped-particle release generation.

Transforms validated base release centers into groups of particles,
where each group consists of:
  - Member 1: at the base release center
  - Members 2..n: on a circle of radius radius_km around the center
"""

from __future__ import annotations

import numpy as np
from parcels import FieldSet

from kinematicparcels.utilities.init_checks import mask_inside_domain, mask_inside_ocean


def km_to_degrees(distance_km: float) -> float:
    """
    Convert distance in kilometers to approximate degrees on Earth.
    
    Uses approximate mean radius: 1 degree ≈ 111.32 km.
    This is accurate for modest distances (< 100 km).
    """
    return distance_km / 111.32


def generate_circle_points(
    center_lon: float,
    center_lat: float,
    radius_km: float,
    n_points: int,
    start_angle_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate points on a circle around a center location.
    
    Parameters
    ----------
    center_lon, center_lat : float
        Center point (degrees)
    radius_km : float
        Circle radius (kilometers)
    n_points : int
        Number of points to generate on the circle
    start_angle_deg : float
        Starting angle in degrees (0 = East, 90 = North)
        
    Returns
    -------
    lons, lats : 1D arrays
        Coordinates of points on the circle
        
    Notes
    -----
    Uses Cartesian approximation on the sphere, suitable for small radii.
    For large radii (> 100 km), consider geodetic distance formulas.
    """
    radius_deg = km_to_degrees(radius_km)
    
    # Account for latitude effect on longitude spacing
    lat_rad = np.radians(center_lat)
    lon_scaling = np.cos(lat_rad)
    
    # Generate evenly spaced angles
    angles_deg = np.linspace(
        start_angle_deg,
        start_angle_deg + 360.0,
        n_points,
        endpoint=False,
    )
    angles_rad = np.radians(angles_deg)
    
    # Cartesian offsets
    dx = radius_deg * np.cos(angles_rad)
    dy = radius_deg * np.sin(angles_rad)
    
    # Convert to geographic coordinates
    lons = center_lon + dx / lon_scaling
    lats = center_lat + dy
    
    return lons, lats


def expand_groups(
    lons_base: np.ndarray,
    lats_base: np.ndarray,
    fieldset: FieldSet,
    group_size: int,
    radius_km: float,
    placement: str = "random",
    seed: int | None = None,
    filter_land: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand base release centers into groups of particles.
    
    Each group consists of:
      - Member 1: at the base center
      - Members 2..group_size: on circle of radius radius_km
    
    If any member of a group falls outside the fieldset domain,
    the entire group is discarded.
    
    Parameters
    ----------
    lons_base, lats_base : 1D np.ndarray
        Validated base release centers
    fieldset : parcels.FieldSet
        For domain boundary checking
    group_size : int
        Number of particles per group (>= 1)
    radius_km : float
        Radius (km) of circle for non-center members
    placement : {'random', 'equal_angles'}
        'random': random starting angle per group, then equal spacing
        'equal_angles': fixed starting angle (0 degrees), deterministic spacing
    seed : int or None
        Random seed for reproducibility (used if placement='random')
        
    Returns
    -------
    lons_expanded : 1D np.ndarray
        Longitudes of all particles (flattened across groups)
    lats_expanded : 1D np.ndarray
        Latitudes of all particles (flattened across groups)
    group_id : 1D np.ndarray (dtype int)
        Group index (0, 1, 2, ...) for each particle
    group_member : 1D np.ndarray (dtype int)
        Member index within group (1, 2, ..., group_size) for each particle
    group_size_arr : 1D np.ndarray (dtype int)
        Group size for each particle (constant = group_size)
        
    Raises
    ------
    ValueError
        If group_size < 1 or radius_km < 0
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if radius_km < 0:
        raise ValueError(f"radius_km must be >= 0, got {radius_km}")
    
    lons_base = np.asarray(lons_base, dtype=float).ravel()
    lats_base = np.asarray(lats_base, dtype=float).ravel()
    
    if len(lons_base) != len(lats_base):
        raise ValueError("lons_base and lats_base must have same length")
    
    if group_size == 1:
        # Single mode: no expansion needed
        return (
            lons_base,
            lats_base,
            np.arange(len(lons_base), dtype=int),  # group_id
            np.ones(len(lons_base), dtype=int),    # group_member = 1
            np.ones(len(lons_base), dtype=int),    # group_size = 1
        )
    
    # Generalize case: group_size > 1
    if placement not in ("random", "equal_angles"):
        raise ValueError(
            f"placement must be 'random' or 'equal_angles', got {placement}"
        )
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate all particles and track metadata
    lons_all = []
    lats_all = []
    group_id_all = []
    group_member_all = []
    group_size_all = []
    
    n_circle_members = group_size - 1  # members 2..group_size
    angular_spacing = 360.0 / n_circle_members if n_circle_members > 0 else 360.0
    
    # Track valid groups after domain filtering
    n_base_points = len(lons_base)
    valid_groups = []
    
    for group_idx, (lon_center, lat_center) in enumerate(zip(lons_base, lats_base)):
        # Member 1: center
        lons_group = [lon_center]
        lats_group = [lat_center]
        
        # Members 2..group_size: on circle
        if n_circle_members > 0:
            if placement == "random":
                start_angle = np.random.uniform(0, 360)
            else:  # equal_angles
                start_angle = 0.0
            
            lons_circle, lats_circle = generate_circle_points(
                center_lon=lon_center,
                center_lat=lat_center,
                radius_km=radius_km,
                n_points=n_circle_members,
                start_angle_deg=start_angle,
            )
            lons_group.extend(lons_circle)
            lats_group.extend(lats_circle)
        
        # Check if all members of this group are inside domain
        lons_group = np.asarray(lons_group)
        lats_group = np.asarray(lats_group)
        
        mask = mask_inside_domain(lons_group, lats_group, fieldset, inclusive=False)
        if filter_land:
            mask &= mask_inside_ocean(lons_group, lats_group, fieldset)

        if np.all(mask):
            # All members are valid, keep the group
            lons_all.extend(lons_group)
            lats_all.extend(lats_group)
            
            for member_idx in range(1, group_size + 1):
                group_id_all.append(group_idx)
                group_member_all.append(member_idx)
                group_size_all.append(group_size)
            
            valid_groups.append(group_idx)
        # else: discard entire group
    
    # Report filtering statistics
    n_original = n_base_points
    n_valid = len(valid_groups)
    n_discarded = n_original - n_valid
    
    if n_discarded > 0:
        print(
            f"[group expansion] domain filtering: "
            f"{n_original} base groups -> {n_valid} valid "
            f"({n_discarded} discarded)"
        )
    
    # Flatten results
    lons_expanded = np.asarray(lons_all, dtype=float)
    lats_expanded = np.asarray(lats_all, dtype=float)
    group_id = np.asarray(group_id_all, dtype=int)
    group_member = np.asarray(group_member_all, dtype=int)
    group_size_arr = np.asarray(group_size_all, dtype=int)
    
    if len(lons_expanded) == 0:
        print("[group expansion] WARNING: all groups were filtered out")
    
    return lons_expanded, lats_expanded, group_id, group_member, group_size_arr
