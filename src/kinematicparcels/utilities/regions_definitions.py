"""
Compatibility wrapper for the legacy utilities.regions_definitions module.

The canonical region definitions now live under kinematicparcels.regions.definitions.
"""

from kinematicparcels.regions.definitions import REGIONS_DATA, REGIONS_DATA_RECTANGLES

__all__ = [
    "REGIONS_DATA",
    "REGIONS_DATA_RECTANGLES",
]