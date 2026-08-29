from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths

from .config import BarriersConfig, GeometryConfig, PermeabilityConfig


@dataclass(frozen=True)
class BarrierResult:
    candidates_all: pd.DataFrame
    candidates_selected: pd.DataFrame
    points: pd.DataFrame
    summary: pd.DataFrame


def detect_barrier_candidates(
    cross_sections: pd.DataFrame,
    barriers: BarriersConfig,
    permeability: PermeabilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    if cross_sections.empty:
        empty = pd.DataFrame()
        return empty, empty
    min_distance_steps = max(1, int(np.ceil(barriers.min_separation_km / permeability.offset_spacing_km)))
    for (branch_id, point_id), profile in cross_sections.groupby(["branch_id", "branch_point_id"], sort=True):
        profile = profile.sort_values("offset_km").reset_index(drop=True)
        valid = profile.support_valid.to_numpy(bool) & np.isfinite(profile.P_cross.to_numpy(float))
        start = 0
        while start < len(profile):
            while start < len(profile) and not valid[start]: start += 1
            stop = start
            while stop < len(profile) and valid[stop]: stop += 1
            if stop - start >= 3:
                segment = profile.iloc[start:stop].copy()
                values = segment.P_cross.to_numpy(float)
                smooth = gaussian_filter1d(values, barriers.smoothing_sigma_steps, mode="nearest")
                profile.loc[start:stop - 1, "P_cross_smoothed"] = smooth
                peaks, props = find_peaks(-smooth, prominence=0.0)
                widths = peak_widths(-smooth, peaks, rel_height=0.5)[0] if len(peaks) else np.asarray([])
                raw_candidates = list(peaks)
                if smooth[0] < smooth[1]: raw_candidates.insert(0, 0)
                if smooth[-1] < smooth[-2]: raw_candidates.append(len(smooth) - 1)
                accepted_positions: list[int] = []
                candidate_data: list[tuple[int, float, float, int, int]] = []
                for local_idx in raw_candidates:
                    if local_idx in peaks:
                        prop_idx = int(np.flatnonzero(peaks == local_idx)[0])
                        prominence = float(props["prominences"][prop_idx])
                        width_km = float(widths[prop_idx] * permeability.offset_spacing_km)
                        left_base = int(props["left_bases"][prop_idx])
                        right_base = int(props["right_bases"][prop_idx])
                    else:
                        prominence = 0.0; width_km = 0.0
                        left_base = local_idx; right_base = local_idx
                    candidate_data.append((local_idx, prominence, width_km, left_base, right_base))
                for local_idx, prominence, width_km, left_base, right_base in sorted(candidate_data, key=lambda x: -x[1]):
                    absolute_idx = start + local_idx
                    row = profile.iloc[absolute_idx]
                    reasons = []
                    if local_idx in (0, len(smooth) - 1): reasons.append("edge_minimum")
                    if prominence < barriers.min_prominence: reasons.append("low_prominence")
                    if width_km < barriers.min_width_km: reasons.append("narrow_minimum")
                    if any(abs(local_idx - prior) < min_distance_steps for prior in accepted_positions):
                        reasons.append("too_close_to_stronger_minimum")
                    if not reasons: accepted_positions.append(local_idx)
                    ref_idx = left_base if smooth[left_base] >= smooth[right_base] else right_base
                    reference = profile.iloc[start + ref_idx]
                    ci_overlap = bool(row.P_cross_ci_high >= reference.P_cross_ci_low)
                    rows.append({
                        "branch_id": branch_id, "branch_point_id": point_id,
                        "point_order": int(row.point_order), "s_km": float(row.s_km),
                        "offset_km": float(row.offset_km), "lon": float(row.candidate_lon),
                        "lat": float(row.candidate_lat), "P_cross": float(row.P_cross),
                        "tangent_x": float(row.get("tangent_x", np.nan)),
                        "tangent_y": float(row.get("tangent_y", np.nan)),
                        "normal_x": float(row.get("normal_x", np.nan)),
                        "normal_y": float(row.get("normal_y", np.nan)),
                        "P_cross_smoothed": float(smooth[local_idx]),
                        "P_minus_to_plus": float(row.P_minus_to_plus),
                        "P_plus_to_minus": float(row.P_plus_to_minus),
                        "P_cross_moving": float(row.P_cross_moving),
                        "directional_asymmetry": float(row.directional_asymmetry),
                        "prominence": prominence, "width_km": width_km,
                        "reference_P_cross": float(reference.P_cross),
                        "reference_offset_km": float(reference.offset_km),
                        "total_support_count": int(row.counts_minus + row.counts_plus),
                        "P_cross_ci_low": float(row.P_cross_ci_low),
                        "P_cross_ci_high": float(row.P_cross_ci_high),
                        "reference_ci_low": float(reference.P_cross_ci_low),
                        "reference_ci_high": float(reference.P_cross_ci_high),
                        "ci_overlaps_reference": ci_overlap,
                        "accepted": not reasons, "rejection_reasons": ";".join(reasons),
                        "quality_flags": ("ci_overlaps_reference" if ci_overlap else ""),
                    })
            start = max(stop, start + 1)
    all_candidates = pd.DataFrame(rows)
    selected = all_candidates.loc[all_candidates.accepted].copy() if not all_candidates.empty else all_candidates.copy()
    if not selected.empty:
        selected = selected.sort_values(["branch_id", "point_order", "offset_km"]).reset_index(drop=True)
        selected.insert(0, "barrier_candidate_id", [f"bc{i:07d}" for i in range(len(selected))])
        all_candidates = all_candidates.merge(
            selected[["branch_id", "branch_point_id", "offset_km", "barrier_candidate_id"]],
            on=["branch_id", "branch_point_id", "offset_km"], how="left",
        )
    return all_candidates, selected


def _compatible(a: pd.Series, b: pd.Series, config: BarriersConfig, geod: Geod) -> tuple[bool, float, float]:
    offset_gap = abs(float(a.offset_km) - float(b.offset_km))
    _, _, distance_m = geod.inv(float(a.lon), float(a.lat), float(b.lon), float(b.lat))
    distance = distance_m / 1000.0
    sign_ok = (
        abs(float(a.offset_km)) <= config.core_sign_band_km
        or abs(float(b.offset_km)) <= config.core_sign_band_km
        or np.sign(a.offset_km) == np.sign(b.offset_km)
    )
    valid = offset_gap <= config.max_offset_jump_km and distance <= config.max_physical_gap_km and sign_ok
    cost = offset_gap / config.max_offset_jump_km + distance / config.max_physical_gap_km + abs(float(a.P_cross) - float(b.P_cross))
    return bool(valid), float(cost), float(distance)


def connect_barrier_segments(
    selected: pd.DataFrame, geometry: GeometryConfig, config: BarriersConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    geod = Geod(ellps=geometry.ellipsoid)
    output: list[pd.DataFrame] = []
    summaries: list[dict] = []
    barrier_counter = 0
    for branch_id, candidates in selected.groupby("branch_id", sort=True):
        candidates = candidates.sort_values(["point_order", "offset_km"]).copy()
        tracks: list[list[int]] = []
        for order in sorted(candidates.point_order.unique()):
            current_indices = candidates.index[candidates.point_order.eq(order)].tolist()
            active_tracks = [i for i, track in enumerate(tracks)
                             if order - int(candidates.loc[track[-1], "point_order"]) <= config.max_missing_sections + 1]
            matched_current: set[int] = set()
            if active_tracks and current_indices:
                costs = np.full((len(active_tracks), len(current_indices)), 1e6)
                distances = np.full_like(costs, np.nan)
                for i, track_index in enumerate(active_tracks):
                    previous = candidates.loc[tracks[track_index][-1]]
                    for j, current_index in enumerate(current_indices):
                        valid, cost, distance = _compatible(previous, candidates.loc[current_index], config, geod)
                        if valid: costs[i, j] = cost; distances[i, j] = distance
                row_ids, col_ids = linear_sum_assignment(costs)
                for i, j in zip(row_ids, col_ids):
                    if costs[i, j] >= 1e6: continue
                    tracks[active_tracks[i]].append(current_indices[j]); matched_current.add(current_indices[j])
            for current_index in current_indices:
                if current_index not in matched_current: tracks.append([current_index])
        for track in tracks:
            frame = candidates.loc[track].sort_values("point_order").copy()
            barrier_id = f"bar{barrier_counter:05d}"; barrier_counter += 1
            frame["barrier_id"] = barrier_id
            frame["barrier_point_order"] = np.arange(len(frame))
            frame["geometry_part"] = np.r_[0, np.cumsum(np.diff(frame.point_order.to_numpy(int)) > 1)]
            distances = []
            gap_lengths = []
            for previous, current in zip(frame.iloc[:-1].itertuples(), frame.iloc[1:].itertuples()):
                _, _, distance_m = geod.inv(previous.lon, previous.lat, current.lon, current.lat)
                distance = distance_m / 1000.0
                if current.point_order - previous.point_order > 1:
                    gap_lengths.append(distance); distances.append(0.0)
                else:
                    distances.append(distance)
            frame["along_barrier_km"] = np.r_[0.0, np.cumsum(distances)]
            length = float(sum(distances))
            robust = len(frame) >= config.min_segment_points and length >= config.min_segment_length_km
            frame["robust_segment"] = robust
            output.append(frame)
            span = int(frame.point_order.max() - frame.point_order.min() + 1)
            summaries.append({
                "barrier_id": barrier_id, "parent_branch_id": branch_id,
                "n_points": len(frame), "segment_length_km": length,
                "coverage_fraction": len(frame) / span, "gap_count": len(gap_lengths),
                "gap_length_km": float(sum(gap_lengths)), "robust_segment": robust,
                "mean_permeability": float(frame.P_cross.mean()),
                "median_permeability": float(frame.P_cross.median()),
                "min_permeability": float(frame.P_cross.min()),
                "max_permeability": float(frame.P_cross.max()),
                "mean_directional_asymmetry": float(frame.directional_asymmetry.mean()),
            })
    return pd.concat(output, ignore_index=True), pd.DataFrame(summaries)


def analyze_barriers(
    cross_sections: pd.DataFrame, geometry: GeometryConfig,
    permeability: PermeabilityConfig, barriers: BarriersConfig,
) -> BarrierResult:
    all_candidates, selected = detect_barrier_candidates(cross_sections, barriers, permeability)
    points, summary = connect_barrier_segments(selected, geometry, barriers)
    return BarrierResult(all_candidates, selected, points, summary)
