"""
Group dynamics and synchronized LKM updates.

This module handles the orchestration of group-level computations and
stores relative coordinates for the kernel to use.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from kinematicparcels.utilities.geometry import full_group_geom_to_local


def compute_active_group_membership(pset) -> dict[int, list]:
    """
    Build mapping of active particles by group_id.

    Parameters
    ----------
    pset : ParticleSet
        Parcels ParticleSet

    Returns
    -------
    dict
        group_id -> list of particle indices
    """
    group_membership = {}

    for i, particle in enumerate(pset):
        if particle.state == 0:  # Active particle
            group_id = int(particle.group_id)
            if group_id not in group_membership:
                group_membership[group_id] = []
            group_membership[group_id].append(i)

    return group_membership


def update_group_centers_and_relative_coords(
    pset,
    apply_to_group_size_min: int = 2,
) -> None:
    """
    Update group centers and store relative coordinates for LKM calculation.

    This function:
    1. Identifies active particles and groups them by group_id
    2. Computes spherical group centers and local coordinates
    3. Stores relative coordinates (x_rel_m, y_rel_m) on particles
    4. The kernel will compute LKM velocities from these coordinates

    Parameters
    ----------
    pset : ParticleSet
        Parcels ParticleSet with group metadata
    apply_to_group_size_min : int
        Minimum group size to apply LKM (default: 2)
    """
    # Get active particles
    active_mask = np.array([p.state == 0 for p in pset])
    if not np.any(active_mask):
        return  # No active particles

    # Extract positions and group IDs for active particles
    active_indices = np.where(active_mask)[0]
    lons_deg = np.array([pset[i].lon for i in active_indices])
    lats_deg = np.array([pset[i].lat for i in active_indices])
    group_ids = np.array([pset[i].group_id for i in active_indices])

    # Compute group centers and local coordinates
    center_lons_rad, center_lats_rad, x_rel_m, y_rel_m = full_group_geom_to_local(
        lons_deg, lats_deg, group_ids
    )

    # Store coordinates on particles for kernel to use
    for j, particle_idx in enumerate(active_indices):
        particle = pset[particle_idx]

        # Store relative coordinates (kernel will compute LKM from these)
        particle.x_rel_m = float(x_rel_m[j])
        particle.y_rel_m = float(y_rel_m[j])

        # Store group center for diagnostics
        particle.center_lon = np.degrees(center_lons_rad[j])
        particle.center_lat = np.degrees(center_lats_rad[j])


# Backward compatibility alias (old name)
update_group_centers_and_lkm = update_group_centers_and_relative_coords