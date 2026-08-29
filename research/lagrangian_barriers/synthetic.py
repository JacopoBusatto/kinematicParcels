from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import GridConfig


def sparse_transition_table(
    grid: GridConfig,
    transitions: Iterable[tuple[int, int, int, int, int]],
) -> pd.DataFrame:
    """Build a normalized sparse table from (start lon/lat, end lon/lat, count)."""
    rows = []
    for slon, slat, elon, elat, count in transitions:
        rows.append({
            "start_lon_bin": slon, "start_lat_bin": slat,
            "end_lon_bin": elon, "end_lat_bin": elat,
            "start_lon_center": grid.lon_min + (slon + .5) * grid.dlon,
            "start_lat_center": grid.lat_min + (slat + .5) * grid.dlat,
            "end_lon_center": grid.lon_min + (elon + .5) * grid.dlon,
            "end_lat_center": grid.lat_min + (elat + .5) * grid.dlat,
            "transition_count": count,
        })
    frame = pd.DataFrame(rows)
    totals = frame.groupby(["start_lon_bin", "start_lat_bin"]).transition_count.transform("sum")
    frame["transition_probability"] = frame.transition_count / totals
    return frame


def zonal_corridor(grid: GridConfig, *, split: bool = False, gap_lon: int | None = None) -> pd.DataFrame:
    rows = []
    lat = grid.nlat // 2
    for lon in range(1, grid.nlon - 2):
        if lon == gap_lon:
            continue
        rows.append((lon, lat, lon + 1, lat, 24))
        rows.append((lon, lat, lon, lat, 6))
        if split and lon == grid.nlon // 2:
            rows[-2] = (lon, lat, lon + 1, lat - 1, 12)
            rows.append((lon, lat, lon + 1, lat + 1, 12))
    return sparse_transition_table(grid, rows)
