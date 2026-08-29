from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pyproj import Geod
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, unary_union


def annotate_seed_family(
    branch_points: pd.DataFrame,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.DataFrame:
    """Annotate components touching a post-detection geographic seed box."""
    lon = branch_points.lon.to_numpy(float)
    longitude_match = (lon >= lon_min) & (lon <= lon_max) if lon_min <= lon_max else ((lon >= lon_min) | (lon <= lon_max))
    seed = branch_points.loc[longitude_match & branch_points.lat.between(lat_min, lat_max)]
    components = set(seed.component_id)
    out = branch_points[["branch_id", "component_id"]].drop_duplicates().copy()
    out["seed_family_selected"] = out.component_id.isin(components)
    out["annotation_only"] = True
    return out


def annotate_external_front_distances(
    detected_points: pd.DataFrame,
    front_geojson: str | Path,
    *,
    ellipsoid: str = "WGS84",
) -> pd.DataFrame:
    """Measure post-detection point-to-front distances without changing selection."""
    payload = json.loads(Path(front_geojson).read_text(encoding="utf-8"))
    geometries = [shape(feature["geometry"]) for feature in payload.get("features", [])]
    if not geometries:
        raise ValueError("External-front GeoJSON contains no geometries")
    fronts = unary_union(geometries)
    geod = Geod(ellps=ellipsoid)
    rows = []
    for row in detected_points.itertuples(index=False):
        nearest = nearest_points(Point(float(row.lon), float(row.lat)), fronts)[1]
        _, _, distance_m = geod.inv(float(row.lon), float(row.lat), nearest.x, nearest.y)
        rows.append({
            "branch_id": getattr(row, "branch_id", None),
            "barrier_id": getattr(row, "barrier_id", None),
            "point_id": getattr(row, "branch_point_id", getattr(row, "barrier_candidate_id", None)),
            "front_lon": nearest.x, "front_lat": nearest.y,
            "front_distance_km": distance_m / 1000.0, "annotation_only": True,
        })
    return pd.DataFrame(rows)
