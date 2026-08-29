from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .common import circular_difference_degrees
from .config import ModesConfig


@dataclass(frozen=True)
class ModesResult:
    modes: pd.DataFrame
    membership: pd.DataFrame
    rejected: pd.DataFrame
    cell_mode_summary: pd.DataFrame


def circular_pdf_peaks(bearings: np.ndarray, weights: np.ndarray, config: ModesConfig):
    n = config.angular_bins
    width = 360.0 / n
    hist, _ = np.histogram(bearings % 360.0, bins=n, range=(0.0, 360.0), weights=weights)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    sigma = config.smoothing_bandwidth_degrees / width
    pdf = gaussian_filter1d(hist, sigma=sigma, mode="wrap")
    triple = np.tile(pdf, 3)
    distance = max(1, int(np.ceil(config.min_peak_separation_degrees / width)))
    peaks, props = find_peaks(triple, prominence=0.0, distance=distance)
    central = (peaks >= n) & (peaks < 2 * n)
    peaks = peaks[central] - n
    prominences = props["prominences"][central]
    if len(peaks) == 0 and pdf.max() > 0:
        peaks = np.asarray([int(np.argmax(pdf))])
        prominences = np.asarray([0.0])
    order = np.argsort(peaks)
    return hist, pdf, peaks[order], prominences[order]


def _basin_assignment(pdf: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    n = len(pdf)
    if len(peaks) == 0:
        return np.full(n, -1, dtype=int)
    if len(peaks) == 1:
        return np.zeros(n, dtype=int)
    boundaries: list[int] = []
    for i, peak in enumerate(peaks):
        nxt = int(peaks[(i + 1) % len(peaks)])
        indexes = np.arange(peak, nxt + (n if nxt <= peak else 0) + 1) % n
        values = pdf[indexes]
        minima = indexes[np.isclose(values, values.min(), rtol=0, atol=1e-15)]
        boundaries.append(int(minima[len(minima) // 2]))
    assigned = np.full(n, -1, dtype=int)
    for i in range(len(peaks)):
        start = (boundaries[i - 1] + 1) % n
        stop = boundaries[i]
        idx = start
        while True:
            assigned[idx] = i
            if idx == stop:
                break
            idx = (idx + 1) % n
    return assigned


def detect_transport_modes(transitions: pd.DataFrame, config: ModesConfig) -> ModesResult:
    mode_rows: list[dict] = []
    rejected_rows: list[dict] = []
    membership = transitions[[
        "transition_id", "start_cell_id", "end_cell_id", "start_lon_bin", "start_lat_bin",
        "end_lon_bin", "end_lat_bin", "transition_count", "transition_probability",
        "bearing_deg", "distance_km", "is_stay",
    ]].copy()
    n_records = len(membership)
    mode_assignments = np.full(n_records, None, dtype=object)
    candidate_assignments = np.full(n_records, None, dtype=object)
    assignment_reasons = np.where(membership.is_stay.to_numpy(bool), "stay_excluded", "no_supported_peak").astype(object)
    angular_mismatches = np.full(n_records, np.nan, dtype=float)
    bearing_by_id = np.full(n_records, np.nan, dtype=float)
    bearing_by_id[membership.transition_id.to_numpy(int)] = membership.bearing_deg.to_numpy(float)

    width = 360.0 / config.angular_bins
    for cell_id, group in transitions.groupby("start_cell_id", sort=True):
        moving = group.loc[~group.is_stay]
        if moving.empty:
            continue
        total_count = int(group.transition_count.sum())
        moving_count = int(moving.transition_count.sum())
        weights = moving.conditional_moving_probability.to_numpy(float)
        hist, pdf, peaks, prominences = circular_pdf_peaks(
            moving.bearing_deg.to_numpy(float), weights, config
        )
        basin_by_bin = _basin_assignment(pdf, peaks)
        bearing_bins = np.floor((moving.bearing_deg.to_numpy(float) % 360.0) / width).astype(int)
        basin = basin_by_bin[bearing_bins]
        candidate_records: list[dict] = []
        for candidate_index, peak_bin in enumerate(peaks):
            members = moving.iloc[np.flatnonzero(basin == candidate_index)]
            if members.empty:
                continue
            w = members.conditional_moving_probability.to_numpy(float)
            mass_moving = float(w.sum())
            mass_all = float(members.transition_probability.sum())
            count = int(members.transition_count.sum())
            dx = float(np.sum(w * members.dx_km) / mass_moving)
            dy = float(np.sum(w * members.dy_km) / mass_moving)
            mean_distance = float(np.sum(w * members.distance_km) / mass_moving)
            distance_sd = float(np.sqrt(np.sum(w * (members.distance_km - mean_distance) ** 2) / mass_moving))
            vector_bearing = float(np.degrees(np.arctan2(dx, dy)) % 360.0)
            peak_bearing = float((peak_bin + 0.5) * width)
            delta = ((members.bearing_deg.to_numpy(float) - vector_bearing + 180) % 360) - 180
            angular_sd = float(np.sqrt(np.sum(w * delta ** 2) / mass_moving))
            prominence = float(prominences[candidate_index])
            relative_prominence = prominence / float(pdf.max()) if pdf.max() > 0 else 0.0
            reasons = []
            if total_count < config.min_start_count: reasons.append("low_start_count")
            if moving_count < config.min_moving_count: reasons.append("low_moving_count")
            if count < config.min_mode_count: reasons.append("low_mode_count")
            if mass_moving < config.min_mode_probability: reasons.append("low_mode_probability")
            if relative_prominence < config.min_relative_prominence: reasons.append("low_peak_prominence")
            if mean_distance < config.min_mean_distance_km: reasons.append("short_mean_distance")
            candidate_records.append({
                "candidate_index": candidate_index, "peak_bin": int(peak_bin),
                "peak_bearing_deg": peak_bearing, "relative_prominence": relative_prominence,
                "peak_prominence": prominence, "mode_probability_moving": mass_moving,
                "mode_probability_all": mass_all, "transition_count": count,
                "modal_dx_km": dx, "modal_dy_km": dy, "modal_bearing_deg": vector_bearing,
                "modal_mean_distance_km": mean_distance, "modal_distance_sd_km": distance_sd,
                "angular_sd_deg": angular_sd, "n_endpoint_cells": int(members.end_cell_id.nunique()),
                "n_transition_records": len(members), "member_ids": members.transition_id.to_numpy(int),
                "rejection_reasons": reasons,
            })

        accepted = sorted(
            [r for r in candidate_records if not r["rejection_reasons"]],
            key=lambda r: r["modal_bearing_deg"],
        )
        accepted_ids = {r["candidate_index"]: f"{int(cell_id)}:m{i:02d}" for i, r in enumerate(accepted)}
        for rec in candidate_records:
            candidate_id = f"{int(cell_id)}:c{rec['candidate_index']:02d}"
            base = {
                "start_cell_id": int(cell_id), "candidate_mode_id": candidate_id,
                "start_lon_bin": int(group.start_lon_bin.iloc[0]),
                "start_lat_bin": int(group.start_lat_bin.iloc[0]),
                "start_lon": float(group.start_lon_center.iloc[0]),
                "start_lat": float(group.start_lat_center.iloc[0]),
                **{k: v for k, v in rec.items() if k not in {"member_ids", "rejection_reasons", "candidate_index"}},
            }
            ids = rec["member_ids"]
            candidate_assignments[ids] = candidate_id
            if rec["candidate_index"] in accepted_ids:
                mode_id = accepted_ids[rec["candidate_index"]]
                base.update(mode_id=mode_id, support_flags="")
                mode_rows.append(base)
                mode_assignments[ids] = mode_id
                assignment_reasons[ids] = "assigned"
                angular_mismatches[ids] = circular_difference_degrees(
                    bearing_by_id[ids], rec["modal_bearing_deg"]
                )
            else:
                reasons = ";".join(rec["rejection_reasons"])
                base.update(rejection_reasons=reasons)
                rejected_rows.append(base)
                assignment_reasons[ids] = reasons

    modes = pd.DataFrame(mode_rows)
    rejected = pd.DataFrame(rejected_rows)
    membership["mode_id"] = mode_assignments[membership.transition_id.to_numpy(int)]
    membership["candidate_mode_id"] = candidate_assignments[membership.transition_id.to_numpy(int)]
    membership["assignment_reason"] = assignment_reasons[membership.transition_id.to_numpy(int)]
    membership["angular_mismatch_deg"] = angular_mismatches[membership.transition_id.to_numpy(int)]
    membership = membership.sort_values("transition_id").reset_index(drop=True)
    if modes.empty:
        summary = pd.DataFrame(columns=["start_cell_id", "number_of_modes", "dominant_mode_probability"])
    else:
        summary = modes.groupby("start_cell_id").agg(
            number_of_modes=("mode_id", "size"),
            dominant_mode_probability=("mode_probability_moving", "max"),
        ).reset_index()
    return ModesResult(modes, membership, rejected, summary)
