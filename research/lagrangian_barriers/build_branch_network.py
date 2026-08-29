from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod

from .common import circular_difference_degrees, local_grid_diagonal_km
from .config import GeometryConfig, GraphConfig, GridConfig


@dataclass(frozen=True)
class GraphResult:
    nodes: pd.DataFrame
    edges_all: pd.DataFrame
    edges_selected: pd.DataFrame
    graph: nx.DiGraph


def build_branch_network(
    modes: pd.DataFrame,
    membership: pd.DataFrame,
    grid: GridConfig,
    geometry: GeometryConfig,
    config: GraphConfig,
) -> GraphResult:
    nodes = modes.copy()
    if nodes.empty:
        return GraphResult(nodes, pd.DataFrame(), pd.DataFrame(), nx.DiGraph())
    target_by_cell = {int(k): g.copy() for k, g in modes.groupby("start_cell_id", sort=True)}
    assigned = membership.loc[membership.mode_id.notna() & ~membership.is_stay].copy()
    geod = Geod(ellps=geometry.ellipsoid)
    rows: list[dict] = []

    for source in modes.sort_values("mode_id").itertuples(index=False):
        members = assigned.loc[assigned.mode_id.eq(source.mode_id)]
        exact: list[dict] = []
        for transition in members.itertuples(index=False):
            targets = target_by_cell.get(int(transition.end_cell_id))
            if targets is None:
                rows.append({
                    "source_node": source.mode_id, "target_node": None,
                    "represented_transition_ids": str(int(transition.transition_id)),
                    "transition_count_support": int(transition.transition_count),
                    "probability_support_all": float(transition.transition_probability),
                    "region_support_fraction": float(transition.transition_probability / source.mode_probability_all),
                    "angular_mismatch_deg": np.nan, "spatial_gap_km": float(transition.distance_km),
                    "distance_mismatch_km": np.nan, "edge_score": 0.0,
                    "selected": False, "rejection_reasons": "no_target_mode",
                })
                continue
            for target in targets.itertuples(index=False):
                mismatch = float(circular_difference_degrees(transition.bearing_deg, target.modal_bearing_deg))
                grid_diag = local_grid_diagonal_km(
                    geod, source.start_lon, source.start_lat, grid.dlon, grid.dlat
                )
                scale = max(float(source.modal_distance_sd_km), grid_diag, 1.0)
                distance_mismatch = abs(float(transition.distance_km) - float(source.modal_mean_distance_km))
                angular_factor = np.exp(-0.5 * (mismatch / config.alignment_scale_degrees) ** 2)
                distance_factor = np.exp(-0.5 * (distance_mismatch / scale) ** 2)
                support = float(transition.transition_probability / source.mode_probability_all)
                exact.append({
                    "source_node": source.mode_id, "target_node": target.mode_id,
                    "target_lon": float(target.start_lon), "target_lat": float(target.start_lat),
                    "target_bearing": float(target.modal_bearing_deg),
                    "transition_id": int(transition.transition_id),
                    "transition_count": int(transition.transition_count),
                    "probability": float(transition.transition_probability),
                    "support_fraction": support, "mismatch": mismatch,
                    "distance": float(transition.distance_km), "distance_mismatch": distance_mismatch,
                    "angular_factor": float(angular_factor), "distance_factor": float(distance_factor),
                })
        if not exact:
            continue

        exact.sort(key=lambda r: (-r["support_fraction"] * r["angular_factor"] * r["distance_factor"], r["target_node"]))
        clusters: list[list[dict]] = []
        for candidate in exact:
            placed = False
            for cluster in clusters:
                representative = cluster[0]
                _, _, distance_m = geod.inv(
                    representative["target_lon"], representative["target_lat"],
                    candidate["target_lon"], candidate["target_lat"],
                )
                radius = config.cluster_radius_grid_diagonals * local_grid_diagonal_km(
                    geod, candidate["target_lon"], candidate["target_lat"], grid.dlon, grid.dlat
                )
                if distance_m / 1000 <= radius and float(circular_difference_degrees(
                    representative["target_bearing"], candidate["target_bearing"]
                )) <= config.cluster_bearing_degrees:
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])

        source_rows: list[dict] = []
        for cluster_index, cluster in enumerate(clusters):
            representative = cluster[0]
            unique = {item["transition_id"]: item for item in cluster}
            probabilities = sum(item["probability"] for item in unique.values())
            counts = sum(item["transition_count"] for item in unique.values())
            support_fraction = probabilities / float(source.mode_probability_all)
            mismatch = float(np.average([x["mismatch"] for x in cluster], weights=[x["probability"] for x in cluster]))
            gap = float(np.average([x["distance"] for x in cluster], weights=[x["probability"] for x in cluster]))
            distance_mismatch = float(np.average([x["distance_mismatch"] for x in cluster], weights=[x["probability"] for x in cluster]))
            angular_factor = float(np.exp(-0.5 * (mismatch / config.alignment_scale_degrees) ** 2))
            scale = max(float(source.modal_distance_sd_km), local_grid_diagonal_km(
                geod, source.start_lon, source.start_lat, grid.dlon, grid.dlat
            ), 1.0)
            distance_factor = float(np.exp(-0.5 * (distance_mismatch / scale) ** 2))
            score = support_fraction * angular_factor * distance_factor
            reasons = []
            if mismatch > config.max_angular_mismatch_degrees: reasons.append("angular_mismatch")
            if gap > config.max_edge_distance_km: reasons.append("edge_too_long")
            source_rows.append({
                "source_node": source.mode_id, "target_node": representative["target_node"],
                "continuation_region": cluster_index,
                "represented_transition_ids": ",".join(str(v) for v in sorted(unique)),
                "represented_target_nodes": ",".join(sorted({x["target_node"] for x in cluster})),
                "represented_endpoint_count": len({x["transition_id"] for x in cluster}),
                "transition_count_support": counts, "probability_support_all": probabilities,
                "region_support_fraction": support_fraction, "angular_mismatch_deg": mismatch,
                "spatial_gap_km": gap, "distance_mismatch_km": distance_mismatch,
                "angular_factor": angular_factor, "distance_factor": distance_factor,
                "edge_score": score, "selected": False, "rejection_reasons": ";".join(reasons),
            })
        eligible = [row for row in source_rows if not row["rejection_reasons"]]
        eligible.sort(key=lambda r: (-r["edge_score"], r["target_node"]))
        best = eligible[0]["edge_score"] if eligible else 0.0
        cumulative = 0.0
        cutoff_score: float | None = None
        for row in eligible:
            row["relative_score"] = row["edge_score"] / best if best > 0 else 0.0
            if row["relative_score"] + 1e-15 < config.min_relative_score:
                row["rejection_reasons"] = "low_relative_score"
                continue
            if cumulative >= config.cumulative_endpoint_mass and (
                cutoff_score is None or not np.isclose(row["edge_score"], cutoff_score)
            ):
                row["rejection_reasons"] = "beyond_cumulative_mass"
                continue
            row["selected"] = True
            cumulative += row["region_support_fraction"]
            if cumulative >= config.cumulative_endpoint_mass and cutoff_score is None:
                cutoff_score = row["edge_score"]
        for row in source_rows:
            row.setdefault("relative_score", row["edge_score"] / best if best > 0 else 0.0)
        rows.extend(source_rows)

    edges_all = pd.DataFrame(rows)
    selected = edges_all.loc[edges_all.selected & edges_all.target_node.notna()].copy()
    graph = nx.DiGraph()
    for node in nodes.itertuples(index=False):
        graph.add_node(node.mode_id, lon=float(node.start_lon), lat=float(node.start_lat),
                       probability=float(node.mode_probability_all), count=int(node.transition_count))
    for edge in selected.itertuples(index=False):
        graph.add_edge(edge.source_node, edge.target_node, score=float(edge.edge_score),
                       probability=float(edge.probability_support_all), count=int(edge.transition_count_support))
    return GraphResult(nodes, edges_all, selected, graph)
