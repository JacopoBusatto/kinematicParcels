"""Probable fronts from persistent loss of locally aligned directional signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

from ._edge_kernel import robust_contiguous_median
from .config import CompactConfig
from .directional_corridors import DirectionalCorridorSolution
from .geometry import bilinear_supported_sample, grid_array


@dataclass(frozen=True)
class DirectionalFrontSolution:
    fronts: pd.DataFrame
    candidate_drops: pd.DataFrame
    cross_sections: pd.DataFrame
    section_summaries: pd.DataFrame
    summary: dict[str, Any]


SAMPLED_FIELDS = (
    "D_out_all_east",
    "D_out_all_north",
    "D_out_all_magnitude",
    "P_move",
    "R1_out",
    "N_out_move",
)


def _sample_fields(
    prepared: dict[str, np.ndarray],
    support: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    config: CompactConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    sampled: dict[str, np.ndarray] = {}
    boundary_any = np.zeros(len(target_lon), dtype=bool)
    missing_any = np.zeros(len(target_lon), dtype=bool)
    for field, values in prepared.items():
        field_values, boundary, missing = bilinear_supported_sample(
            values,
            support,
            target_lon,
            target_lat,
            config.grid,
            weight_tolerance=config.branches.interpolation_weight_tolerance,
        )
        sampled[field] = field_values
        if field in {"D_out_all_east", "D_out_all_north"}:
            boundary_any |= boundary
            missing_any |= missing
    return sampled, boundary_any, missing_any


def _section_rows(
    center: pd.Series,
    prepared: dict[str, np.ndarray],
    support: np.ndarray,
    config: CompactConfig,
    geod: Geod,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    half_width = config.edges.half_width_grid_scales
    interval = config.edges.sampling_interval_grid_scales
    offsets = np.arange(-half_width, half_width + interval / 2.0, interval)
    distance_km = offsets * float(center.grid_effective_scale_km)
    bearings = np.where(
        distance_km < 0.0,
        float(center.theta1_out) - 90.0,
        float(center.theta1_out) + 90.0,
    )
    target_lon, target_lat, _ = geod.fwd(
        np.full(len(offsets), float(center.lon)),
        np.full(len(offsets), float(center.lat)),
        bearings,
        np.abs(distance_km) * 1000.0,
    )
    zero = np.isclose(offsets, 0.0)
    target_lon[zero] = float(center.lon)
    target_lat[zero] = float(center.lat)
    sampled, boundary, missing = _sample_fields(
        prepared, support, target_lon, target_lat, config
    )
    tangent_east = np.sin(np.deg2rad(float(center.theta1_out)))
    tangent_north = np.cos(np.deg2rad(float(center.theta1_out)))
    d_parallel = (
        sampled["D_out_all_east"] * tangent_east
        + sampled["D_out_all_north"] * tangent_north
    )

    left_observable = bool(center.left_side_observable)
    right_observable = bool(center.right_side_observable)
    original_side = np.where(
        offsets < 0.0, "left", np.where(offsets > 0.0, "right", "axis")
    )
    original_side_observable = (
        (original_side == "axis")
        | ((original_side == "left") & left_observable)
        | ((original_side == "right") & right_observable)
    )
    valid = np.isfinite(d_parallel)
    refinement = config.edges.core_refinement_grid_scales
    allowed_axis = (
        valid
        & original_side_observable
        & (np.abs(offsets) <= refinement + 1.0e-12)
    )
    if allowed_axis.any():
        indexes = np.flatnonzero(allowed_axis)
        axis_index = indexes[np.nanargmax(d_parallel[allowed_axis])]
        axis_offset = float(offsets[axis_index])
        axis_value = float(d_parallel[axis_index])
        refined_lon = float(target_lon[axis_index])
        refined_lat = float(target_lat[axis_index])
        axis_uncertain = bool(np.isclose(abs(axis_offset), refinement))
    else:
        axis_offset = np.nan
        axis_value = np.nan
        refined_lon = np.nan
        refined_lat = np.nan
        axis_uncertain = True

    refined_offset = (
        offsets - axis_offset
        if np.isfinite(axis_offset)
        else np.full_like(offsets, np.nan)
    )
    refined_distance_km = refined_offset * float(center.grid_effective_scale_km)
    side = np.where(
        refined_offset < 0.0,
        "left",
        np.where(refined_offset > 0.0, "right", "axis"),
    )
    side_observable = (
        (side == "axis")
        | ((side == "left") & left_observable)
        | ((side == "right") & right_observable)
    )
    smooth = robust_contiguous_median(
        d_parallel, window=config.edges.robust_median_window_samples
    )
    relative = (
        d_parallel / axis_value
        if axis_value > 0.0
        else np.full_like(d_parallel, np.nan)
    )
    sample_class = np.full(len(offsets), "interpolated", dtype=object)
    sample_class[zero & valid] = "direct_grid_information"
    sample_class[missing] = "missing_because_of_support"
    sample_class[boundary] = "missing_because_of_boundary"
    sample_class[~valid & ~(missing | boundary)] = "missing_invalid"
    flags: list[str] = []
    if missing.any():
        flags.append("unsupported_transverse_sample")
    if boundary.any():
        flags.append("domain_boundary")
    if not left_observable:
        flags.append("left_side_not_observable")
    if not right_observable:
        flags.append("right_side_not_observable")
    if axis_uncertain:
        flags.append("axis_location_uncertain")
    if bool(center.directional_graph_junction):
        flags.append("directional_graph_junction")
    if bool(center.directional_graph_endpoint):
        flags.append("directional_graph_endpoint")
    if valid.sum() < config.edges.diagnostic_min_full_section_valid_samples:
        flags.append("short_available_cross_section")

    section_id = f"directional_cell_{int(center.cell_id):05d}"
    rows = pd.DataFrame(
        {
            "component_id": center.component_id,
            "section_id": section_id,
            "corridor_cell_id": int(center.cell_id),
            "corridor_lon": float(center.lon),
            "corridor_lat": float(center.lat),
            "refined_axis_lon": refined_lon,
            "refined_axis_lat": refined_lat,
            "offset_index_from_corridor_cell": offsets,
            "offset_index_from_refined_axis": refined_offset,
            "distance_from_corridor_cell_km": distance_km,
            "distance_from_refined_axis_km": refined_distance_km,
            "sample_lon": target_lon,
            "sample_lat": target_lat,
            "side": side,
            "theta1_out_center": float(center.theta1_out),
            "D_parallel_raw": d_parallel,
            "D_parallel_smoothed": smooth,
            "D_parallel_relative": relative,
            "sample_valid": valid,
            "side_observable": side_observable,
            "analysis_sample_eligible": valid & side_observable,
            "sample_class": sample_class,
            "quality_flags": ";".join(flags),
        }
    )
    for field, values in sampled.items():
        rows[field] = values
    summary = {
        "component_id": center.component_id,
        "section_id": section_id,
        "corridor_cell_id": int(center.cell_id),
        "corridor_lon": float(center.lon),
        "corridor_lat": float(center.lat),
        "theta1_out_center": float(center.theta1_out),
        "grid_effective_scale_km": float(center.grid_effective_scale_km),
        "axis_refinement_grid_scales": axis_offset,
        "axis_refinement_distance_km": (
            axis_offset * float(center.grid_effective_scale_km)
            if np.isfinite(axis_offset)
            else np.nan
        ),
        "refined_axis_lon": refined_lon,
        "refined_axis_lat": refined_lat,
        "D_parallel_axis": axis_value,
        "axis_location_uncertain": axis_uncertain,
        "directional_graph_degree": int(center.directional_graph_degree),
        "directional_graph_endpoint": bool(center.directional_graph_endpoint),
        "directional_graph_junction": bool(center.directional_graph_junction),
        "left_side_observable": left_observable,
        "right_side_observable": right_observable,
        "n_observable_sides": int(left_observable) + int(right_observable),
        "n_unobservable_sides": 2 - int(left_observable) - int(right_observable),
        "n_valid_samples": int(valid.sum()),
        "n_missing_support_samples": int(missing.sum()),
        "n_missing_boundary_samples": int(boundary.sum()),
        "short_available_cross_section": bool(
            valid.sum() < config.edges.diagnostic_min_full_section_valid_samples
        ),
        "opposing_outer_directional_signal": bool(
            np.any(d_parallel[np.abs(refined_offset) >= 2.0] < 0.0)
        )
        if np.isfinite(axis_offset)
        else False,
        "quality_flags": ";".join(flags),
    }
    return rows, summary


def _adjacency(corridors: DirectionalCorridorSolution) -> dict[int, set[int]]:
    result = {int(cell_id): set() for cell_id in corridors.corridors.cell_id}
    if corridors.edges.empty:
        return result
    for edge in corridors.edges.itertuples(index=False):
        first = int(edge.first_cell_id)
        second = int(edge.second_cell_id)
        result[first].add(second)
        result[second].add(first)
    return result


def _neighbors_within_hops(
    start: int, adjacency: dict[int, set[int]], maximum_hops: int
) -> set[int]:
    visited = {start}
    frontier = {start}
    for _ in range(maximum_hops):
        frontier = {
            neighbor
            for cell_id in frontier
            for neighbor in adjacency.get(cell_id, set())
            if neighbor not in visited
        }
        if not frontier:
            break
        visited.update(frontier)
    return visited


def _add_graph_composites(
    cross_sections: pd.DataFrame,
    summaries: pd.DataFrame,
    corridors: DirectionalCorridorSolution,
    config: CompactConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = _adjacency(corridors)
    section_by_cell = summaries.set_index("corridor_cell_id").section_id.to_dict()
    section_frames = {
        section_id: frame.copy()
        for section_id, frame in cross_sections.groupby("section_id", sort=False)
    }
    output_frames: list[pd.DataFrame] = []
    neighbor_count: dict[str, int] = {}
    for focal in summaries.itertuples(index=False):
        neighbor_cells = _neighbors_within_hops(
            int(focal.corridor_cell_id),
            graph,
            config.edges.composite_half_window_sections,
        )
        neighbor_sections = {
            section_by_cell[cell_id]
            for cell_id in neighbor_cells
            if cell_id in section_by_cell
        }
        profiles = pd.concat(
            [
                section_frames[section_id][
                    [
                        "section_id",
                        "offset_index_from_refined_axis",
                        "D_parallel_raw",
                    ]
                ]
                for section_id in neighbor_sections
            ],
            ignore_index=True,
        )
        aggregate = profiles.groupby("offset_index_from_refined_axis").D_parallel_raw.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            count="count",
        )
        focal_rows = section_frames[focal.section_id].copy()
        indexes = focal_rows.offset_index_from_refined_axis
        focal_rows["D_parallel_composite_median"] = indexes.map(
            aggregate["median"]
        )
        focal_rows["D_parallel_composite_q25"] = indexes.map(aggregate["q25"])
        focal_rows["D_parallel_composite_q75"] = indexes.map(aggregate["q75"])
        focal_rows["n_composite_sections"] = indexes.map(aggregate["count"])
        focal_rows["n_sections_with_outward_decline"] = np.nan
        focal_rows["fraction_sections_with_outward_decline"] = np.nan
        pivot = profiles.pivot_table(
            index="section_id",
            columns="offset_index_from_refined_axis",
            values="D_parallel_raw",
            aggfunc="first",
        )
        for side in ("left", "right"):
            side_rows = focal_rows.loc[focal_rows.side.isin(("axis", side))].copy()
            side_rows["outward_distance"] = (
                side_rows.offset_index_from_refined_axis.abs()
            )
            side_rows = side_rows.sort_values("outward_distance")
            for pair_index in range(len(side_rows) - 1):
                inner_offset = side_rows.iloc[pair_index].offset_index_from_refined_axis
                outer = side_rows.iloc[pair_index + 1]
                outer_offset = outer.offset_index_from_refined_axis
                if inner_offset not in pivot or outer_offset not in pivot:
                    continue
                valid_pairs = pivot[[inner_offset, outer_offset]].dropna()
                if valid_pairs.empty:
                    continue
                declines = valid_pairs[inner_offset] > valid_pairs[outer_offset]
                focal_rows.loc[
                    int(outer.name), "n_sections_with_outward_decline"
                ] = int(declines.sum())
                focal_rows.loc[
                    int(outer.name), "fraction_sections_with_outward_decline"
                ] = float(declines.mean())
        output_frames.append(focal_rows)
        neighbor_count[focal.section_id] = len(neighbor_sections)
    output = (
        pd.concat(output_frames).sort_index().reset_index(drop=True)
        if output_frames
        else cross_sections.copy()
    )
    summaries = summaries.copy()
    summaries["n_neighbor_sections"] = summaries.section_id.map(neighbor_count).fillna(0)
    summaries["n_neighbor_sections"] = summaries.n_neighbor_sections.astype(int)
    return output, summaries


def _contiguous_following(
    side_rows: pd.DataFrame,
    start_index: int,
    *,
    interval: float,
    maximum: int = 2,
) -> pd.DataFrame:
    selected_indexes: list[int] = []
    previous = side_rows.iloc[start_index]
    for position in range(start_index + 1, min(len(side_rows), start_index + 1 + maximum)):
        current = side_rows.iloc[position]
        gap = float(current.outward_offset - previous.outward_offset)
        if not np.isclose(gap, interval, rtol=0.0, atol=1.0e-10):
            break
        if not bool(current.analysis_sample_eligible):
            break
        selected_indexes.append(position)
        previous = current
    return side_rows.iloc[selected_indexes].copy()


def _first_distance(side_rows: pd.DataFrame, threshold: float) -> float:
    selected = side_rows.loc[
        side_rows.analysis_sample_eligible
        & side_rows.D_parallel_relative.le(threshold)
    ]
    return (
        float(selected.outward_distance_km.iloc[0]) if not selected.empty else np.nan
    )


def _detect_drops(
    cross_sections: pd.DataFrame,
    summaries: pd.DataFrame,
    config: CompactConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_records: list[dict[str, Any]] = []
    summary_updates: list[dict[str, Any]] = []
    interval = config.edges.sampling_interval_grid_scales
    for summary in summaries.itertuples(index=False):
        profile = cross_sections.loc[
            cross_sections.section_id.eq(summary.section_id)
        ].copy()
        side_results: dict[str, Any] = {}
        for side in ("left", "right"):
            observable = bool(getattr(summary, f"{side}_side_observable"))
            if not observable:
                side_results.update(
                    {
                        f"{side}_front_detected": False,
                        f"{side}_candidate_count": 0,
                        f"{side}_front_distance_km": np.nan,
                        f"{side}_directional_drop": np.nan,
                        f"{side}_relative_directional_drop": np.nan,
                        f"{side}_status": "side_not_observable",
                    }
                )
                continue
            selected = profile.loc[profile.side.isin(("axis", side))].copy()
            selected["outward_offset"] = selected.offset_index_from_refined_axis.abs()
            selected["outward_distance_km"] = selected.distance_from_refined_axis_km.abs()
            selected = selected.sort_values("outward_offset").reset_index(drop=True)
            zones: list[dict[str, Any]] = []
            for position in range(len(selected) - 1):
                inner = selected.iloc[position]
                outer = selected.iloc[position + 1]
                if not (
                    bool(inner.analysis_sample_eligible)
                    and bool(outer.analysis_sample_eligible)
                ):
                    continue
                if not np.isclose(
                    float(outer.outward_offset - inner.outward_offset),
                    interval,
                    rtol=0.0,
                    atol=1.0e-10,
                ):
                    continue
                profile_field = (
                    "D_parallel_composite_median"
                    if min(inner.n_composite_sections, outer.n_composite_sections) >= 2
                    else "D_parallel_smoothed"
                )
                inner_value = float(inner[profile_field])
                outer_value = float(outer[profile_field])
                if not np.isfinite(inner_value) or not np.isfinite(outer_value):
                    continue
                following = _contiguous_following(
                    selected, position + 1, interval=interval
                )
                following_values = following[profile_field].dropna()
                remains_lower = bool(
                    len(following_values)
                    and following_values.median() <= (inner_value + outer_value) / 2.0
                )
                absolute_drop = inner_value - outer_value
                relative_drop = (
                    absolute_drop / summary.D_parallel_axis
                    if summary.D_parallel_axis > 0.0
                    else np.nan
                )
                width = float(
                    outer.outward_distance_km - inner.outward_distance_km
                )
                persistence_available = bool(outer.n_composite_sections >= 2)
                persistent = bool(
                    outer.n_sections_with_outward_decline
                    >= config.edges.minimum_persistent_neighbor_sections
                    and outer.fraction_sections_with_outward_decline
                    >= config.edges.minimum_persistent_fraction
                )
                eligible = bool(
                    absolute_drop > 0.0
                    and remains_lower
                    and (not persistence_available or persistent)
                )
                zones.append(
                    {
                        "component_id": summary.component_id,
                        "section_id": summary.section_id,
                        "corridor_cell_id": summary.corridor_cell_id,
                        "side": side,
                        "candidate_distance_km": float(outer.outward_distance_km),
                        "candidate_lon": float(outer.sample_lon),
                        "candidate_lat": float(outer.sample_lat),
                        "D_parallel_axis": summary.D_parallel_axis,
                        "D_parallel_inner": inner_value,
                        "D_parallel_outer": outer_value,
                        "absolute_directional_drop": absolute_drop,
                        "relative_directional_drop": relative_drop,
                        "directional_drop_per_km": absolute_drop / width,
                        "drop_width_km": width,
                        "along_corridor_persistence": persistent,
                        "n_sections_with_outward_decline": int(
                            outer.n_sections_with_outward_decline
                        )
                        if np.isfinite(outer.n_sections_with_outward_decline)
                        else 0,
                        "fraction_sections_with_outward_decline": float(
                            outer.fraction_sections_with_outward_decline
                        )
                        if np.isfinite(outer.fraction_sections_with_outward_decline)
                        else np.nan,
                        "n_neighbor_sections": int(
                            min(inner.n_composite_sections, outer.n_composite_sections)
                        ),
                        "candidate_eligible": eligible,
                        "quality_flags": summary.quality_flags,
                    }
                )
            zones_frame = pd.DataFrame.from_records(zones)
            eligible = (
                zones_frame.loc[zones_frame.candidate_eligible].copy()
                if not zones_frame.empty
                else pd.DataFrame()
            )
            if not eligible.empty:
                eligible = eligible.sort_values(
                    [
                        "along_corridor_persistence",
                        "absolute_directional_drop",
                        "directional_drop_per_km",
                    ],
                    ascending=[False, False, False],
                    kind="stable",
                )
                best_index = eligible.index[0]
                zones_frame["drop_rank"] = np.nan
                zones_frame.loc[best_index, "drop_rank"] = 1.0
                best = zones_frame.loc[best_index]
                detected = True
                candidate_count = len(eligible)
                distance = float(best.candidate_distance_km)
                drop = float(best.absolute_directional_drop)
                relative_drop = float(best.relative_directional_drop)
            else:
                if not zones_frame.empty:
                    zones_frame["drop_rank"] = np.nan
                detected = False
                candidate_count = 0
                distance = drop = relative_drop = np.nan
            if not zones_frame.empty:
                candidate_records.extend(zones_frame.to_dict("records"))
            side_results.update(
                {
                    f"{side}_front_detected": detected,
                    f"{side}_candidate_count": candidate_count,
                    f"{side}_front_distance_km": distance,
                    f"{side}_directional_drop": drop,
                    f"{side}_relative_directional_drop": relative_drop,
                    f"{side}_status": (
                        "probable_directional_front"
                        if detected
                        else "observable_no_retained_directional_front"
                    ),
                    f"{side}_distance_50pct_km": _first_distance(selected, 0.5),
                    f"{side}_distance_1e_km": _first_distance(selected, 1.0 / np.e),
                    f"{side}_zero_crossing_km": _first_distance(selected, 0.0),
                }
            )
        summary_updates.append({"section_id": summary.section_id, **side_results})
    candidates = pd.DataFrame.from_records(candidate_records)
    updates = pd.DataFrame.from_records(summary_updates)
    return candidates, summaries.merge(updates, on="section_id", how="left")


def _canonical_fronts(
    corridors: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    base = corridors[
        ["cell_id", "component_id", "lon", "lat", "corridor_observability"]
    ].rename(
        columns={
            "cell_id": "corridor_cell_id",
            "lon": "corridor_lon",
            "lat": "corridor_lat",
        }
    )
    base = base.merge(pd.DataFrame({"side": ["left", "right"]}), how="cross")
    side_observable = corridors.set_index("cell_id")
    base["observable"] = [
        bool(
            side_observable.loc[row.corridor_cell_id, f"{row.side}_side_observable"]
        )
        for row in base.itertuples(index=False)
    ]
    selected = (
        candidates.loc[candidates.drop_rank.eq(1)].copy()
        if not candidates.empty
        else pd.DataFrame()
    )
    fields = [
        "front_lon",
        "front_lat",
        "distance_from_corridor_axis_km",
        "absolute_directional_drop",
        "relative_directional_drop",
    ]
    if not selected.empty:
        selected = selected.rename(
            columns={
                "candidate_lon": "front_lon",
                "candidate_lat": "front_lat",
                "candidate_distance_km": "distance_from_corridor_axis_km",
            }
        )
        selected = selected[["corridor_cell_id", "side", *fields]]
        fronts = base.merge(selected, on=["corridor_cell_id", "side"], how="left")
    else:
        fronts = base.copy()
        for field in fields:
            fronts[field] = np.nan
    fronts["front_detected"] = fronts.observable & fronts.front_lon.notna()
    fronts["front_status"] = np.select(
        [~fronts.observable, fronts.front_detected],
        ["side_not_observable", "probable_directional_front"],
        default="observable_no_retained_directional_front",
    )
    fronts.loc[~fronts.front_detected, fields] = np.nan
    side_order = pd.Categorical(fronts.side, ["left", "right"], ordered=True)
    return (
        fronts.assign(_side_order=side_order)
        .sort_values(["corridor_cell_id", "_side_order"])
        .drop(columns="_side_order")
        .reset_index(drop=True)
    )


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    finite = values.dropna()
    if finite.empty:
        return {f"{prefix}_{name}": np.nan for name in ("q10", "q50", "q90")}
    return {
        f"{prefix}_q10": float(finite.quantile(0.1)),
        f"{prefix}_q50": float(finite.quantile(0.5)),
        f"{prefix}_q90": float(finite.quantile(0.9)),
    }


def compute_probable_directional_fronts(
    cells: pd.DataFrame,
    corridors: DirectionalCorridorSolution,
    config: CompactConfig,
) -> DirectionalFrontSolution:
    """Calculate directional sections and persistent outward D_parallel losses."""
    required = set(SAMPLED_FIELDS)
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"Directional-front cells missing columns: {missing}")
    if corridors.corridors.empty:
        empty = pd.DataFrame()
        return DirectionalFrontSolution(empty, empty, empty, empty, {
            "sections": 0,
            "sections_with_two_retained_fronts": 0,
            "sections_with_one_retained_front": 0,
            "observable_sides_without_retained_front": 0,
            "unobservable_sides": 0,
        })

    prepared = {field: grid_array(cells, config.grid, field) for field in SAMPLED_FIELDS}
    support = (
        prepared["N_out_move"] >= config.statistics.min_moving_support
    ) & np.isfinite(prepared["D_out_all_east"]) & np.isfinite(
        prepared["D_out_all_north"]
    )
    geod = Geod(ellps=config.ellipsoid)
    cross_outputs: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    for center in corridors.corridors.itertuples(index=False):
        rows, summary = _section_rows(
            pd.Series(center._asdict()), prepared, support, config, geod
        )
        cross_outputs.append(rows)
        summary_records.append(summary)
    cross_sections = pd.concat(cross_outputs, ignore_index=True)
    section_summaries = pd.DataFrame.from_records(summary_records)
    cross_sections, section_summaries = _add_graph_composites(
        cross_sections, section_summaries, corridors, config
    )
    candidates, section_summaries = _detect_drops(
        cross_sections, section_summaries, config
    )
    fronts = _canonical_fronts(corridors.corridors, candidates)
    statuses = fronts.front_status.value_counts()
    two = section_summaries.left_front_detected & section_summaries.right_front_detected
    one = section_summaries.left_front_detected ^ section_summaries.right_front_detected
    detected = fronts.loc[fronts.front_detected]
    summary: dict[str, Any] = {
        "sections": len(section_summaries),
        "sections_with_two_retained_fronts": int(two.sum()),
        "sections_with_one_retained_front": int(one.sum()),
        "sections_with_no_retained_front": int((~two & ~one).sum()),
        "probable_directional_fronts": int(
            statuses.get("probable_directional_front", 0)
        ),
        "observable_sides_without_retained_front": int(
            statuses.get("observable_no_retained_directional_front", 0)
        ),
        "unobservable_sides": int(statuses.get("side_not_observable", 0)),
        "sections_with_missing_support_samples": int(
            section_summaries.n_missing_support_samples.gt(0).sum()
        ),
        "short_available_sections": int(
            section_summaries.short_available_cross_section.sum()
        ),
        "sections_with_opposing_outer_signal": int(
            section_summaries.opposing_outer_directional_signal.sum()
        ),
        "D_parallel_definition": (
            "interpolated D_out_all dot central local theta1_out; tangent fixed "
            "within each section"
        ),
        "missing_support_semantics": (
            "missing samples break contiguous profiles and cannot form drops"
        ),
        **_quantiles(
            detected.distance_from_corridor_axis_km,
            "front_distance_from_corridor_axis_km",
        ),
        **_quantiles(
            detected.absolute_directional_drop, "absolute_directional_drop"
        ),
        **_quantiles(
            detected.relative_directional_drop, "relative_directional_drop"
        ),
    }
    return DirectionalFrontSolution(
        fronts=fronts,
        candidate_drops=candidates,
        cross_sections=cross_sections,
        section_summaries=section_summaries,
        summary=summary,
    )
