"""
Post-processing tools for Parcels outputs.
"""

from .core import build_particle_summary
from .io import (
    build_trajectory_table,
    load_trajectory_table,
    open_parcels_dataset,
    resolve_parcels_schema,
    sanitize_trajectories,
)
from .plotting import plot_connectivity_map, plot_trajectories_map

__all__ = [
    "open_parcels_dataset",
    "resolve_parcels_schema",
    "build_trajectory_table",
    "sanitize_trajectories",
    "load_trajectory_table",
    "build_particle_summary",
    "plot_trajectories_map",
    "plot_connectivity_map",
]