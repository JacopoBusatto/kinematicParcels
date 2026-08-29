from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

from .config import GeometryConfig, GridConfig, PermeabilityConfig


@dataclass(frozen=True)
class PermeabilityResult:
    cross_sections: pd.DataFrame
    contributions: pd.DataFrame | None
    summary: pd.DataFrame


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])


def _local_coordinates(geod: Geod, lon0: float, lat0: float, bearing: float,
                       lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    az, _, distance_m = geod.inv(np.full(len(lon), lon0), np.full(len(lat), lat0), lon, lat)
    delta = np.deg2rad((az - bearing + 180) % 360 - 180)
    radius = distance_m / 1000.0
    return radius * np.cos(delta), -radius * np.sin(delta)


def diagnose_cross_branch_permeability(
    branch_points: pd.DataFrame,
    transitions: pd.DataFrame,
    grid: GridConfig,
    geometry: GeometryConfig,
    config: PermeabilityConfig,
    *,
    eligible_branch_ids: set[str] | None = None,
    contribution_sink: Callable[[str, pd.DataFrame], None] | None = None,
) -> PermeabilityResult:
    """Evaluate all requested offsets together in each branch-point local frame."""
    geod = Geod(ellps=geometry.ellipsoid)
    cells = transitions.sort_values("transition_id").drop_duplicates("start_cell_id")[
        ["start_cell_id", "start_lon_center", "start_lat_center"]
    ].reset_index(drop=True)
    cell_xyz = _unit_xyz(cells.start_lon_center.to_numpy(float), cells.start_lat_center.to_numpy(float))
    cell_tree = cKDTree(cell_xyz)
    transition_groups = {int(k): g for k, g in transitions.groupby("start_cell_id", sort=False)}
    offsets = np.arange(config.min_offset_km, config.max_offset_km + config.offset_spacing_km * .5,
                        config.offset_spacing_km)
    scan_rows: list[dict] = []
    all_contributions: list[pd.DataFrame] = []
    earth_radius = 6371.0088
    query_radius = np.hypot(config.source_along_halfwidth_km,
                            config.source_normal_halfwidth_km + max(abs(offsets))) + 25
    chord_radius = 2 * np.sin(query_radius / (2 * earth_radius))

    for branch_id, branch in branch_points.groupby("branch_id", sort=True):
        if eligible_branch_ids is not None and branch_id not in eligible_branch_ids:
            continue
        branch_contributions: list[pd.DataFrame] = []
        for point in branch.sort_values("point_order").itertuples(index=False):
            query = _unit_xyz(np.asarray([point.lon]), np.asarray([point.lat]))[0]
            nearby_indexes = cell_tree.query_ball_point(query, chord_radius)
            if not nearby_indexes:
                continue
            nearby_cells = cells.iloc[nearby_indexes].reset_index(drop=True)
            local = pd.concat([transition_groups[int(cid)] for cid in nearby_cells.start_cell_id], ignore_index=True)
            position_by_cell = {int(cid): i for i, cid in enumerate(nearby_cells.start_cell_id)}
            source_position = local.start_cell_id.map(position_by_cell).to_numpy(int)
            source_along, source_normal = _local_coordinates(
                geod, point.lon, point.lat, point.bearing_deg,
                nearby_cells.start_lon_center.to_numpy(float), nearby_cells.start_lat_center.to_numpy(float),
            )
            _, end_normal = _local_coordinates(
                geod, point.lon, point.lat, point.bearing_deg,
                local.end_lon_center.to_numpy(float), local.end_lat_center.to_numpy(float),
            )
            cell_normal_offset = source_normal[:, None] - offsets[None, :]
            cell_inside = (
                (np.abs(source_along[:, None]) <= config.source_along_halfwidth_km)
                & (np.abs(cell_normal_offset) <= config.source_normal_halfwidth_km)
            )
            cell_side = np.where(cell_normal_offset > config.on_line_tolerance_km, 1,
                                 np.where(cell_normal_offset < -config.on_line_tolerance_km, -1, 0))
            row_inside = cell_inside[source_position]
            row_source_side = cell_side[source_position]
            end_normal_offset = end_normal[:, None] - offsets[None, :]
            row_end_side = np.where(end_normal_offset > config.on_line_tolerance_km, 1,
                                    np.where(end_normal_offset < -config.on_line_tolerance_km, -1, 0))
            weights = local.transition_count.to_numpy(np.int64)[:, None]
            minus = row_inside & (row_source_side == -1); plus = row_inside & (row_source_side == 1)
            on_line = row_inside & (row_source_side == 0)
            m2p = minus & (row_end_side == 1); p2m = plus & (row_end_side == -1)
            stay = local.is_stay.to_numpy(bool)[:, None]
            n_minus = (weights * minus).sum(axis=0).astype(int); n_plus = (weights * plus).sum(axis=0).astype(int)
            c_m2p = (weights * m2p).sum(axis=0).astype(int); c_p2m = (weights * p2m).sum(axis=0).astype(int)
            stay_minus = (weights * minus * stay).sum(axis=0).astype(int)
            stay_plus = (weights * plus * stay).sum(axis=0).astype(int)
            move_minus = n_minus - stay_minus; move_plus = n_plus - stay_plus
            cells_minus = (cell_inside & (cell_side == -1)).sum(axis=0).astype(int)
            cells_plus = (cell_inside & (cell_side == 1)).sum(axis=0).astype(int)
            on_line_counts = (weights * on_line).sum(axis=0).astype(int)

            geometry_limit = max(abs(config.min_offset_km), abs(config.max_offset_km))
            common_flags: list[str] = []
            for value, reason in ((point.radius_curvature_km, "curvature_limited"),
                                  (point.nearest_other_branch_km, "near_branch_limited"),
                                  (point.self_proximity_km, "self_proximity_limited")):
                if np.isfinite(value) and config.geometry_fraction * float(value) < geometry_limit:
                    geometry_limit = config.geometry_fraction * float(value); common_flags.append(reason)
            geometry_valid = np.abs(offsets) <= geometry_limit + 1e-9
            candidate_lon = np.empty(len(offsets)); candidate_lat = np.empty(len(offsets))
            for j, offset in enumerate(offsets):
                candidate_lon[j], candidate_lat[j], _ = geod.fwd(
                    point.lon, point.lat, (point.bearing_deg - 90) % 360, offset * 1000
                )
            domain_valid = (candidate_lat >= grid.lat_min) & (candidate_lat < grid.lat_max)
            geometry_valid &= domain_valid
            effective = offsets[geometry_valid]
            effective_min = float(effective.min()) if len(effective) else np.nan
            effective_max = float(effective.max()) if len(effective) else np.nan

            for j, offset in enumerate(offsets):
                flags = list(common_flags)
                if abs(offset) > geometry_limit + 1e-9: flags.append("outside_geometry_range")
                if not domain_valid[j]: flags.append("outside_domain")
                if cells_minus[j] < config.min_source_cells_per_side: flags.append("few_minus_cells")
                if cells_plus[j] < config.min_source_cells_per_side: flags.append("few_plus_cells")
                if n_minus[j] < config.min_counts_per_side: flags.append("low_minus_count")
                if n_plus[j] < config.min_counts_per_side: flags.append("low_plus_count")
                if move_minus[j] < config.min_moving_counts_per_side: flags.append("low_minus_moving_count")
                if move_plus[j] < config.min_moving_counts_per_side: flags.append("low_plus_moving_count")
                support_valid = bool(geometry_valid[j] and cells_minus[j] >= config.min_source_cells_per_side
                    and cells_plus[j] >= config.min_source_cells_per_side
                    and n_minus[j] >= config.min_counts_per_side and n_plus[j] >= config.min_counts_per_side)
                total_n = int(n_minus[j] + n_plus[j]); total_cross = int(c_m2p[j] + c_p2m[j])
                moving_n = int(move_minus[j] + move_plus[j])
                p_minus = c_m2p[j] / n_minus[j] if n_minus[j] else np.nan
                p_plus = c_p2m[j] / n_plus[j] if n_plus[j] else np.nan
                p_cross = total_cross / total_n if total_n else np.nan
                ci_minus = wilson_interval(int(c_m2p[j]), int(n_minus[j]), config.confidence_level)
                ci_plus = wilson_interval(int(c_p2m[j]), int(n_plus[j]), config.confidence_level)
                ci_cross = wilson_interval(total_cross, total_n, config.confidence_level)
                scan_rows.append({
                    "branch_id": branch_id, "branch_point_id": point.branch_point_id,
                    "point_order": point.point_order, "s_km": point.s_km, "offset_km": float(offset),
                    "candidate_lon": candidate_lon[j], "candidate_lat": candidate_lat[j],
                    "tangent_x": point.tangent_x, "tangent_y": point.tangent_y,
                    "normal_x": point.normal_x, "normal_y": point.normal_y,
                    "requested_min_offset_km": config.min_offset_km, "requested_max_offset_km": config.max_offset_km,
                    "effective_min_offset_km": effective_min, "effective_max_offset_km": effective_max,
                    "source_along_halfwidth_km": config.source_along_halfwidth_km,
                    "source_normal_halfwidth_km": config.source_normal_halfwidth_km,
                    "source_cells_minus": int(cells_minus[j]), "source_cells_plus": int(cells_plus[j]),
                    "counts_minus": int(n_minus[j]), "counts_plus": int(n_plus[j]),
                    "stay_counts_minus": int(stay_minus[j]), "stay_counts_plus": int(stay_plus[j]),
                    "moving_counts_minus": int(move_minus[j]), "moving_counts_plus": int(move_plus[j]),
                    "cross_count_minus_to_plus": int(c_m2p[j]), "cross_count_plus_to_minus": int(c_p2m[j]),
                    "noncross_count_minus": int(n_minus[j] - c_m2p[j]),
                    "noncross_count_plus": int(n_plus[j] - c_p2m[j]),
                    "on_line_source_count": int(on_line_counts[j]),
                    "P_minus_to_plus": p_minus, "P_plus_to_minus": p_plus, "P_cross": p_cross,
                    "P_minus_to_plus_moving": c_m2p[j] / move_minus[j] if move_minus[j] else np.nan,
                    "P_plus_to_minus_moving": c_p2m[j] / move_plus[j] if move_plus[j] else np.nan,
                    "P_cross_moving": total_cross / moving_n if moving_n else np.nan,
                    "directional_asymmetry": p_minus - p_plus if np.isfinite(p_minus) and np.isfinite(p_plus) else np.nan,
                    "P_minus_to_plus_ci_low": ci_minus[0], "P_minus_to_plus_ci_high": ci_minus[1],
                    "P_plus_to_minus_ci_low": ci_plus[0], "P_plus_to_minus_ci_high": ci_plus[1],
                    "P_cross_ci_low": ci_cross[0], "P_cross_ci_high": ci_cross[1],
                    "geometry_valid": bool(geometry_valid[j]), "support_valid": support_valid,
                    "quality_flags": ";".join(sorted(set(flags))),
                })
                if config.save_contributions:
                    denominator = weights[:, 0] * row_inside[:, j] * (row_source_side[:, j] != 0)
                    crossing = weights[:, 0] * (m2p[:, j] | p2m[:, j])
                    aggregated = {
                        "denominator_count": np.bincount(source_position, denominator, minlength=len(nearby_cells)),
                        "crossing_count": np.bincount(source_position, crossing, minlength=len(nearby_cells)),
                        "crossing_minus_to_plus_count": np.bincount(source_position, weights[:, 0] * m2p[:, j], minlength=len(nearby_cells)),
                        "crossing_plus_to_minus_count": np.bincount(source_position, weights[:, 0] * p2m[:, j], minlength=len(nearby_cells)),
                    }
                    used = np.flatnonzero(cell_inside[:, j])
                    if len(used):
                        branch_contributions.append(pd.DataFrame({
                            "branch_id": branch_id, "branch_point_id": point.branch_point_id,
                            "offset_km": float(offset), "start_cell_id": nearby_cells.start_cell_id.to_numpy(int)[used],
                            "source_side": cell_side[used, j],
                            **{name: values[used].astype(np.int64) for name, values in aggregated.items()},
                        }))
        if branch_contributions:
            combined = pd.concat(branch_contributions, ignore_index=True)
            if contribution_sink is not None: contribution_sink(branch_id, combined)
            else: all_contributions.append(combined)

    cross = pd.DataFrame(scan_rows)
    summary = pd.DataFrame() if cross.empty else cross.groupby("branch_id").agg(
        n_cross_sections=("branch_point_id", "nunique"), n_supported=("support_valid", "sum"),
        mean_P_cross=("P_cross", "mean"), median_P_cross=("P_cross", "median"),
        counts_minus=("counts_minus", "sum"), counts_plus=("counts_plus", "sum"),
    ).reset_index()
    contributions = pd.concat(all_contributions, ignore_index=True) if all_contributions else None
    return PermeabilityResult(cross, contributions, summary)
