from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from .common import unwrap_longitudes
from .config import BranchesConfig, GeometryConfig


@dataclass(frozen=True)
class BranchResult:
    points: pd.DataFrame
    summary: pd.DataFrame
    paths: tuple[tuple[str, ...], ...]


def decompose_directed_graph(graph: nx.DiGraph) -> tuple[tuple[str, ...], ...]:
    visited: set[tuple[str, str]] = set()
    paths: list[tuple[str, ...]] = []
    junctions = {n for n in graph if graph.in_degree(n) != 1 or graph.out_degree(n) != 1}
    for start in sorted(junctions):
        for nxt in sorted(graph.successors(start)):
            if (start, nxt) in visited:
                continue
            path = [start, nxt]
            visited.add((start, nxt))
            current = nxt
            while current not in junctions and graph.out_degree(current) == 1:
                following = next(iter(graph.successors(current)))
                if (current, following) in visited:
                    break
                path.append(following)
                visited.add((current, following))
                current = following
            paths.append(tuple(path))
    for edge in sorted(graph.edges()):
        if edge in visited:
            continue
        start, nxt = edge
        path = [start, nxt]
        visited.add(edge)
        current = nxt
        while current != start and graph.out_degree(current) == 1:
            following = next(iter(graph.successors(current)))
            if (current, following) in visited:
                break
            path.append(following)
            visited.add((current, following))
            current = following
        paths.append(tuple(path))
    return tuple(paths)


def _spherical_center(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.column_stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    mean = xyz.mean(axis=0)
    mean /= np.linalg.norm(mean)
    return float(np.rad2deg(np.arctan2(mean[1], mean[0]))), float(np.rad2deg(np.arctan2(mean[2], np.hypot(mean[0], mean[1]))))


def _resample_path(
    lon: np.ndarray, lat: np.ndarray, spacing: float, geod: Geod
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_lon: list[float] = [float(lon[0])]
    dense_lat: list[float] = [float(lat[0])]
    for lon0, lat0, lon1, lat1 in zip(lon[:-1], lat[:-1], lon[1:], lat[1:]):
        az, _, distance_m = geod.inv(lon0, lat0, lon1, lat1)
        distance = distance_m / 1000.0
        steps = max(1, int(np.ceil(distance / max(spacing / 2, 1.0))))
        for k in range(1, steps + 1):
            lo, la, _ = geod.fwd(lon0, lat0, az, distance_m * k / steps)
            dense_lon.append(float(lo)); dense_lat.append(float(la))
    dense_lon_a = unwrap_longitudes(np.asarray(dense_lon))
    dense_lat_a = np.asarray(dense_lat)
    _, _, distance_m = geod.inv(dense_lon_a[:-1], dense_lat_a[:-1], dense_lon_a[1:], dense_lat_a[1:])
    cumulative = np.r_[0.0, np.cumsum(distance_m / 1000.0)]
    targets = np.arange(0.0, cumulative[-1] + spacing * 0.5, spacing)
    if targets[-1] < cumulative[-1] - 1e-8:
        targets = np.r_[targets, cumulative[-1]]
    else:
        targets[-1] = cumulative[-1]
    return np.interp(targets, cumulative, dense_lon_a), np.interp(targets, cumulative, dense_lat_a), targets


def build_branch_geometry(
    graph: nx.DiGraph, nodes: pd.DataFrame, geometry: GeometryConfig, config: BranchesConfig
) -> BranchResult:
    paths = decompose_directed_graph(graph)
    node_lookup = nodes.set_index("mode_id")
    geod = Geod(ellps=geometry.ellipsoid)
    undirected = graph.to_undirected(as_view=True)
    components = sorted(nx.connected_components(undirected), key=lambda members: min(members))
    component_by_node = {
        node: f"c{component_index:05d}"
        for component_index, members in enumerate(components)
        for node in members
    }
    branch_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for branch_number, path in enumerate(paths):
        raw = node_lookup.loc[list(path)]
        raw_lon = raw.start_lon.to_numpy(float); raw_lat = raw.start_lat.to_numpy(float)
        if len(raw) > 1:
            _, _, raw_distance_m = geod.inv(raw_lon[:-1], raw_lat[:-1], raw_lon[1:], raw_lat[1:])
            raw_s = np.r_[0.0, np.cumsum(raw_distance_m / 1000.0)]
        else:
            raw_s = np.asarray([0.0])
        lon, lat, s = _resample_path(
            raw_lon, raw_lat, config.sample_spacing_km, geod
        )
        nearest_raw = np.abs(s[:, None] - raw_s[None, :]).argmin(axis=1)
        raw_count = raw["transition_count"].to_numpy(float) if "transition_count" in raw else np.ones(len(raw))
        raw_probability = raw["mode_probability_all"].to_numpy(float) if "mode_probability_all" in raw else np.full(len(raw), np.nan)
        support_count = np.interp(s, raw_s, raw_count)
        support_probability = np.interp(s, raw_s, raw_probability)
        lon0, lat0 = _spherical_center(lon, lat)
        center_lon = np.full(len(lon), lon0); center_lat = np.full(len(lat), lat0)
        center_azimuth, _, center_distance_m = geod.inv(center_lon, center_lat, lon, lat)
        raw_x = center_distance_m / 1000.0 * np.sin(np.deg2rad(center_azimuth))
        raw_y = center_distance_m / 1000.0 * np.cos(np.deg2rad(center_azimuth))
        smooth_x, smooth_y = raw_x.copy(), raw_y.copy()
        if len(s) >= config.smoothing_window and config.smoothing_window > config.smoothing_order:
            smooth_x = savgol_filter(raw_x, config.smoothing_window, config.smoothing_order, mode="interp")
            smooth_y = savgol_filter(raw_y, config.smoothing_window, config.smoothing_order, mode="interp")
            smooth_x[[0, -1]] = raw_x[[0, -1]]; smooth_y[[0, -1]] = raw_y[[0, -1]]
        smooth_distance_m = np.hypot(smooth_x, smooth_y) * 1000.0
        smooth_azimuth = np.degrees(np.arctan2(smooth_x, smooth_y))
        smooth_lon, smooth_lat, _ = geod.fwd(center_lon, center_lat, smooth_azimuth, smooth_distance_m)
        dx = np.gradient(smooth_x, s, edge_order=1) if len(s) > 1 else np.asarray([1.0])
        dy = np.gradient(smooth_y, s, edge_order=1) if len(s) > 1 else np.asarray([0.0])
        norm = np.maximum(np.hypot(dx, dy), 1e-12)
        tx, ty = dx / norm, dy / norm
        nx_left, ny_left = -ty, tx
        dtx = np.gradient(tx, s, edge_order=1) if len(s) > 1 else np.asarray([0.0])
        dty = np.gradient(ty, s, edge_order=1) if len(s) > 1 else np.asarray([0.0])
        curvature = np.hypot(dtx, dty)
        radius = np.full_like(curvature, np.inf)
        np.divide(1.0, curvature, out=radius, where=curvature > 1e-12)
        bearing = np.degrees(np.arctan2(tx, ty)) % 360.0
        branch_id = f"b{branch_number:05d}"
        frame = pd.DataFrame({
            "branch_id": branch_id, "component_id": [component_by_node[path[0]]] * len(s),
            "branch_point_id": [f"{branch_id}:p{i:05d}" for i in range(len(s))],
            "point_order": np.arange(len(s)), "s_km": s,
            "lon_unwrapped": unwrap_longitudes(np.asarray(smooth_lon)),
            "lon": ((np.asarray(smooth_lon) + 180) % 360) - 180, "lat": smooth_lat,
            "raw_lon_unwrapped": lon, "raw_lat": lat,
            "metric_x_km": smooth_x, "metric_y_km": smooth_y,
            "raw_metric_x_km": raw_x, "raw_metric_y_km": raw_y,
            "tangent_x": tx, "tangent_y": ty, "normal_x": nx_left, "normal_y": ny_left,
            "bearing_deg": bearing, "curvature_km_inv": curvature,
            "radius_curvature_km": radius,
            "metric_crs_definition": f"local_aeqd_geodesic_polar lon_0={lon0:.12g} lat_0={lat0:.12g} ellps={geometry.ellipsoid}",
            "source_node_start": path[0], "source_node_end": path[-1],
            "node_id": raw.index.to_numpy()[nearest_raw],
            "local_support_count": support_count,
            "local_support_probability": support_probability,
            "junction_start": graph.in_degree(path[0]) != 1 or graph.out_degree(path[0]) != 1,
            "junction_end": graph.in_degree(path[-1]) != 1 or graph.out_degree(path[-1]) != 1,
            "cycle": path[0] == path[-1], "nearest_other_branch_km": np.inf,
            "self_proximity_km": np.inf,
        })
        branch_frames.append(frame)
        summaries.append({
            "branch_id": branch_id, "component_id": frame.component_id.iloc[0],
            "n_graph_nodes": len(path), "n_points": len(frame), "length_km": float(s[-1]),
            "is_cycle": bool(path[0] == path[-1]), "scan_eligible": bool(s[-1] >= config.min_scan_length_km),
            "major_branch": bool(s[-1] >= config.major_branch_length_km),
        })
    if not branch_frames:
        return BranchResult(pd.DataFrame(), pd.DataFrame(summaries), paths)
    points = pd.concat(branch_frames, ignore_index=True)
    lon_r, lat_r = np.deg2rad(points.lon), np.deg2rad(points.lat)
    xyz = np.column_stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    tree = cKDTree(xyz)
    _, neighbors = tree.query(xyz, k=min(32, len(points)))
    earth_radius = 6371.0088
    for idx, candidates in enumerate(np.atleast_2d(neighbors)):
        row = points.iloc[idx]
        other = np.inf; self_distance = np.inf
        for candidate in np.atleast_1d(candidates)[1:]:
            target = points.iloc[int(candidate)]
            chord = np.linalg.norm(xyz[idx] - xyz[int(candidate)])
            distance = 2 * earth_radius * np.arcsin(min(1.0, chord / 2))
            if target.branch_id != row.branch_id:
                other = min(other, distance)
            elif abs(int(target.point_order) - int(row.point_order)) > 3:
                self_distance = min(self_distance, distance)
        points.at[idx, "nearest_other_branch_km"] = other
        points.at[idx, "self_proximity_km"] = self_distance
    return BranchResult(points, pd.DataFrame(summaries), paths)
