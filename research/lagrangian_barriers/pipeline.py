from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml
import xarray as xr

from .build_branch_geometry import BranchResult, build_branch_geometry
from .build_branch_network import GraphResult, build_branch_network
from .common import configure_logging, environment_record, file_record, json_write, utc_now
from .compute_transition_geometry import GeometryResult, compute_transition_geometry, diagnostics_dataset
from .config import BarrierAnalysisConfig, dump_config
from .connect_barrier_segments import BarrierResult, analyze_barriers
from .detect_transport_modes import ModesResult, detect_transport_modes
from .diagnose_cross_branch_permeability import PermeabilityResult, diagnose_cross_branch_permeability
from .exports import (
    line_geojson, save_cross_sections_netcdf, save_dataset_netcdf, save_graphml, save_table,
)
from .plot_lagrangian_barriers import produce_figures
from .validate_transition_matrix import validate_transition_matrix


STAGES = ("validation", "geometry", "modes", "graph", "branches", "permeability", "barriers", "figures")


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest: dict[str, Any]
    completed_stages: tuple[str, ...]


def _config_digest(config: BarrierAnalysisConfig) -> str:
    raw = yaml.safe_dump(config.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _new_run_dir(config: BarrierAnalysisConfig, overwrite: bool, resume: bool) -> Path:
    root = Path(config.output.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    matches = sorted(root.glob(f"{config.output.run_name}_*"))
    if resume:
        if not matches:
            raise FileNotFoundError(f"No run exists to resume under {root}")
        return matches[-1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (root / f"{config.output.run_name}_{stamp}").resolve()
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        if root not in run_dir.parents:
            raise ValueError("Refusing to overwrite a directory outside output root")
        shutil.rmtree(run_dir)
    return run_dir


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(run_dir)), "size_bytes": path.stat().st_size}
            for path in sorted(run_dir.rglob("*")) if path.is_file()]


def _summary_text(manifest: dict[str, Any]) -> str:
    lines = [f"Lagrangian barrier analysis: {manifest['status']}", f"Run: {manifest['run_dir']}",
             f"Input: {manifest['input']['path']}"]
    for stage, record in manifest["stages"].items():
        lines.append(f"{stage}: {record.get('status', 'pending')}")
        for key, value in record.get("counts", {}).items(): lines.append(f"  {key}: {value}")
        for warning in record.get("warnings", []): lines.append(f"  WARNING: {warning}")
    return "\n".join(lines) + "\n"


def run_analysis(
    config: BarrierAnalysisConfig, *, overwrite: bool = False, resume: bool = False,
    stop_after: str | None = None,
) -> RunResult:
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError(f"stop_after must be one of {STAGES}")
    input_path = Path(config.input.transition_table).resolve()
    if not input_path.is_file(): raise FileNotFoundError(input_path)
    run_dir = _new_run_dir(config, overwrite, resume)
    for child in ("validation", "transition_geometry", "modes", "graph", "branches",
                  "permeability", "barriers", "figures", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    input_info = file_record(input_path, calculate_hash=config.validation.calculate_sha256)
    config_hash = _config_digest(config)
    manifest_path = run_dir / "run_manifest.json"
    previous = None
    if resume and manifest_path.exists():
        previous = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash or previous.get("input", {}).get("sha256") != input_info["sha256"]:
            raise ValueError("Resume requires the same resolved configuration and input SHA256")
    companion_info = None
    if config.input.companion_netcdf is not None:
        companion_path = Path(config.input.companion_netcdf).resolve()
        if not companion_path.is_file(): raise FileNotFoundError(companion_path)
        companion_info = file_record(companion_path, calculate_hash=config.validation.calculate_sha256)
    manifest: dict[str, Any] = {
        "analysis_name": "lagrangian_transport_branches_and_barriers",
        "analysis_version": config.analysis_version, "created_at_utc": utc_now(),
        "run_dir": str(run_dir), "status": "running", "config_sha256": config_hash,
        "input": input_info, "companion_netcdf": companion_info,
        "timestep_days": config.input.timestep_days,
        "grid": config.to_dict()["grid"], "random_seed": config.random_seed,
        "environment": environment_record(Path(__file__).resolve().parents[2]),
        "parameters": config.to_dict(), "stages": {stage: {"status": "pending"} for stage in STAGES},
        "output_inventory": [],
    }
    if previous is not None:
        manifest = previous
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = utc_now()
        manifest.pop("failure", None); manifest.pop("failed_at_utc", None)
    dump_config(config, run_dir / "run_config_resolved.yaml"); json_write(manifest, manifest_path)

    def complete(stage: str, counts: dict[str, Any], warnings: list[str] | None = None):
        manifest["stages"][stage] = {"status": "complete", "completed_at_utc": utc_now(),
                                      "counts": counts, "warnings": warnings or []}
        logger.info("stage=%s status=complete counts=%s", stage, counts)
        manifest["output_inventory"] = _inventory(run_dir); json_write(manifest, manifest_path)
        (run_dir / "run_summary.txt").write_text(_summary_text(manifest), encoding="utf-8")

    try:
        def stage_complete(name: str) -> bool:
            return resume and manifest.get("stages", {}).get(name, {}).get("status") == "complete"

        if stage_complete("geometry"):
            geometry_transitions = pd.read_parquet(run_dir / "transition_geometry" / "transition_geometry.parquet")
            geometry_ds = xr.load_dataset(run_dir / "transition_geometry" / "cell_diagnostics.nc")
            cells_loaded = geometry_ds.to_dataframe().reset_index()
            cells_loaded = cells_loaded.loc[cells_loaded.N_i.notna()].copy()
            cells_loaded["start_lon_bin"] = np.rint((cells_loaded.lon - config.grid.lon_min) / config.grid.dlon - .5).astype(int)
            cells_loaded["start_lat_bin"] = np.rint((cells_loaded.lat - config.grid.lat_min) / config.grid.dlat - .5).astype(int)
            cells_loaded["start_cell_id"] = cells_loaded.start_lat_bin * config.grid.nlon + cells_loaded.start_lon_bin
            geometry = GeometryResult(geometry_transitions, cells_loaded, geometry_ds)
        else:
            table = pd.read_parquet(input_path)
            validation = validate_transition_matrix(table, config.grid, config.validation)
            companion_errors: list[str] = []
            if config.input.companion_netcdf is not None:
                with xr.open_dataset(config.input.companion_netcdf) as companion:
                    expected_attrs = {"lon_min": config.grid.lon_min, "lon_max": config.grid.lon_max,
                        "lat_min": config.grid.lat_min, "lat_max": config.grid.lat_max,
                        "dlon": config.grid.dlon, "dlat": config.grid.dlat}
                    for key, expected in expected_attrs.items():
                        if key not in companion.attrs or not np.isclose(float(companion.attrs[key]), expected):
                            companion_errors.append(f"companion_grid_mismatch:{key}")
                    timestep = companion.attrs.get("timestep_seconds")
                    if not isinstance(timestep, (int, float, np.integer, np.floating)) or not np.isclose(float(timestep), config.input.timestep_days * 86400.0):
                        companion_errors.append("companion_timestep_mismatch")
                    if "n_segments_start" in companion:
                        observed = companion.n_segments_start.values[validation.row_normalization.start_lat_bin.to_numpy(int), validation.row_normalization.start_lon_bin.to_numpy(int)]
                        if not np.array_equal(observed, validation.row_normalization.N_i.to_numpy(int)):
                            companion_errors.append("companion_start_count_mismatch")
            validation.summary["companion_errors"] = companion_errors
            save_table(validation.row_normalization, run_dir / "validation" / "row_normalization.parquet")
            json_write(validation.summary, run_dir / "validation" / "validation_summary.json")
            if len(validation.duplicates): save_table(validation.duplicates, run_dir / "validation" / "duplicate_transitions.parquet")
            all_validation_errors = [*validation.errors, *companion_errors]
            complete("validation", validation.summary, all_validation_errors)
            if all_validation_errors and config.validation.fail_on_error: raise ValueError(f"Transition validation failed: {all_validation_errors}")
            if stop_after == "validation": return _finish_partial(run_dir, manifest, ("validation",))
            geometry = compute_transition_geometry(validation.transitions, config.grid, config.geometry, config.modes)
            save_table(geometry.transitions, run_dir / "transition_geometry" / "transition_geometry.parquet")
            save_dataset_netcdf(
                geometry.dataset, run_dir / "transition_geometry" / "cell_diagnostics.nc"
            )
            complete("geometry", {"transition_records": len(geometry.transitions), "diagnostic_cells": len(geometry.cell_diagnostics)})
        if stop_after == "geometry": return _finish_partial(run_dir, manifest, STAGES[:2])

        if stage_complete("modes"):
            loaded_modes = pd.read_parquet(run_dir / "modes" / "modes.parquet")
            loaded_membership = pd.read_parquet(run_dir / "modes" / "mode_membership.parquet")
            loaded_rejected = pd.read_parquet(run_dir / "modes" / "rejected_modes.parquet")
            loaded_summary = loaded_modes.groupby("start_cell_id").agg(number_of_modes=("mode_id", "size"), dominant_mode_probability=("mode_probability_moving", "max")).reset_index()
            mode_result = ModesResult(loaded_modes, loaded_membership, loaded_rejected, loaded_summary)
            cells = geometry.cell_diagnostics
        else:
            mode_result = detect_transport_modes(geometry.transitions, config.modes)
            save_table(mode_result.modes, run_dir / "modes" / "modes.parquet")
            save_table(mode_result.membership, run_dir / "modes" / "mode_membership.parquet")
            save_table(mode_result.rejected, run_dir / "modes" / "rejected_modes.parquet")
            cells = geometry.cell_diagnostics.merge(mode_result.cell_mode_summary, on="start_cell_id", how="left")
            cells[["number_of_modes", "dominant_mode_probability"]] = cells[["number_of_modes", "dominant_mode_probability"]].fillna(0)
            save_dataset_netcdf(
                diagnostics_dataset(cells, config.grid),
                run_dir / "transition_geometry" / "cell_diagnostics.nc",
            )
            complete("modes", {"accepted_modes": len(mode_result.modes), "rejected_modes": len(mode_result.rejected),
                               "unassigned_moving_links": int(((mode_result.membership.assignment_reason != 'assigned') & ~mode_result.membership.is_stay).sum())})
        if stop_after == "modes": return _finish_partial(run_dir, manifest, STAGES[:3])

        if stage_complete("graph"):
            graph_nodes = pd.read_parquet(run_dir / "graph" / "mode_nodes.parquet")
            graph_all = pd.read_parquet(run_dir / "graph" / "mode_edges_all.parquet")
            graph_selected = pd.read_parquet(run_dir / "graph" / "mode_edges_selected.parquet")
            loaded_graph = nx.DiGraph()
            for node in graph_nodes.itertuples(): loaded_graph.add_node(node.mode_id, lon=float(node.start_lon), lat=float(node.start_lat))
            for edge in graph_selected.itertuples(): loaded_graph.add_edge(edge.source_node, edge.target_node, score=float(edge.edge_score))
            graph_result = GraphResult(graph_nodes, graph_all, graph_selected, loaded_graph)
        else:
            graph_result = build_branch_network(mode_result.modes, mode_result.membership, config.grid, config.geometry, config.graph)
            save_table(graph_result.nodes, run_dir / "graph" / "mode_nodes.parquet")
            save_table(graph_result.edges_all, run_dir / "graph" / "mode_edges_all.parquet")
            save_table(graph_result.edges_selected, run_dir / "graph" / "mode_edges_selected.parquet")
            all_graph = nx.MultiDiGraph()
            for node in graph_result.nodes.itertuples():
                all_graph.add_node(node.mode_id, lon=float(node.start_lon), lat=float(node.start_lat))
            for edge_index, edge in graph_result.edges_all.loc[graph_result.edges_all.target_node.notna()].iterrows():
                all_graph.add_edge(edge.source_node, edge.target_node, key=str(edge_index),
                                   selected=bool(edge.selected), score=float(edge.edge_score),
                                   rejection_reasons=str(edge.rejection_reasons))
            save_graphml(all_graph, run_dir / "graph" / "mode_graph_all.graphml")
            save_graphml(graph_result.graph, run_dir / "graph" / "branch_graph.graphml")
            complete("graph", {"nodes": len(graph_result.nodes), "candidate_edges": len(graph_result.edges_all),
                               "selected_edges": len(graph_result.edges_selected), "components": nx.number_weakly_connected_components(graph_result.graph) if len(graph_result.graph) else 0})
        if stop_after == "graph": return _finish_partial(run_dir, manifest, STAGES[:4])

        if stage_complete("branches"):
            branch_points_loaded = pd.read_parquet(run_dir / "branches" / "branch_points.parquet")
            branch_summary_loaded = pd.read_csv(run_dir / "branches" / "branch_summary.csv")
            branch_result = BranchResult(branch_points_loaded, branch_summary_loaded, ())
        else:
            branch_result = build_branch_geometry(graph_result.graph, graph_result.nodes, config.geometry, config.branches)
            save_table(branch_result.points, run_dir / "branches" / "branch_points.parquet")
            save_table(branch_result.summary, run_dir / "branches" / "branch_summary.csv")
            line_geojson(branch_result.points, ["branch_id"], run_dir / "branches" / "branches.geojson")
            complete("branches", {"branches": len(branch_result.summary), "branch_points": len(branch_result.points),
                                  "scan_eligible": int(branch_result.summary.scan_eligible.sum()) if len(branch_result.summary) else 0})
        if stop_after == "branches": return _finish_partial(run_dir, manifest, STAGES[:5])

        contribution_root = run_dir / "permeability" / "source_contributions.parquet"
        def contribution_sink(branch_id: str, frame: pd.DataFrame) -> None:
            target = contribution_root / f"branch_id={branch_id}"
            target.mkdir(parents=True, exist_ok=True); frame.to_parquet(target / "part-000.parquet", index=False)
        if stage_complete("permeability"):
            perm_cross_loaded = pd.read_parquet(run_dir / "permeability" / "cross_sections.parquet")
            perm_summary_loaded = pd.read_csv(run_dir / "permeability" / "permeability_summary.csv")
            perm_result = PermeabilityResult(perm_cross_loaded, None, perm_summary_loaded)
        else:
            eligible = set(branch_result.summary.loc[branch_result.summary.scan_eligible, "branch_id"])
            perm_result = diagnose_cross_branch_permeability(
                branch_result.points, geometry.transitions, config.grid, config.geometry, config.permeability,
                eligible_branch_ids=eligible, contribution_sink=contribution_sink if config.permeability.save_contributions else None,
            )
            save_table(perm_result.cross_sections, run_dir / "permeability" / "cross_sections.parquet")
            save_cross_sections_netcdf(perm_result.cross_sections, run_dir / "permeability" / "cross_sections.nc")
            save_table(perm_result.summary, run_dir / "permeability" / "permeability_summary.csv")
            complete("permeability", {"cross_section_records": len(perm_result.cross_sections),
                                      "supported_records": int(perm_result.cross_sections.support_valid.sum()) if len(perm_result.cross_sections) else 0,
                                      "contributions_saved": bool(config.permeability.save_contributions)})
        if stop_after == "permeability": return _finish_partial(run_dir, manifest, STAGES[:6])

        if stage_complete("barriers"):
            barrier_result = BarrierResult(
                pd.read_parquet(run_dir / "barriers" / "barrier_candidates_all.parquet"),
                pd.read_parquet(run_dir / "barriers" / "barrier_candidates_selected.parquet"),
                pd.read_parquet(run_dir / "barriers" / "barrier_points.parquet"),
                pd.read_csv(run_dir / "barriers" / "barrier_summary.csv"),
            )
        else:
            barrier_result = analyze_barriers(perm_result.cross_sections, config.geometry, config.permeability, config.barriers)
            save_table(barrier_result.candidates_all, run_dir / "barriers" / "barrier_candidates_all.parquet")
            save_table(barrier_result.candidates_selected, run_dir / "barriers" / "barrier_candidates_selected.parquet")
            save_table(barrier_result.points, run_dir / "barriers" / "barrier_points.parquet")
            save_table(barrier_result.summary, run_dir / "barriers" / "barrier_summary.csv")
            line_geojson(barrier_result.points, ["barrier_id", "geometry_part"],
                         run_dir / "barriers" / "barrier_segments.geojson", robust_only=True)
            complete("barriers", {"mathematical_minima": len(barrier_result.candidates_all),
                                  "selected_minima": len(barrier_result.candidates_selected),
                                  "barrier_tracks": len(barrier_result.summary),
                                  "robust_barriers": int(barrier_result.summary.robust_segment.sum()) if len(barrier_result.summary) else 0})
        if stop_after == "barriers": return _finish_partial(run_dir, manifest, STAGES[:7])

        produce_figures(run_dir / "figures", cells, geometry.transitions, mode_result.modes,
                        graph_result.edges_all, graph_result.edges_selected, branch_result.points,
                        branch_result.summary, perm_result.cross_sections, barrier_result.candidates_all,
                        barrier_result.points, config.grid, config.plotting)
        complete("figures", {"figure_files": len(list((run_dir / 'figures').glob('*.png')))})
        manifest["status"] = "complete"; manifest["completed_at_utc"] = utc_now()
        manifest["output_inventory"] = _inventory(run_dir); json_write(manifest, manifest_path)
        (run_dir / "run_summary.txt").write_text(_summary_text(manifest), encoding="utf-8")
        return RunResult(run_dir, manifest, STAGES)
    except Exception as exc:
        manifest["status"] = "failed"; manifest["failed_at_utc"] = utc_now(); manifest["failure"] = repr(exc)
        manifest["output_inventory"] = _inventory(run_dir); json_write(manifest, manifest_path)
        (run_dir / "run_summary.txt").write_text(_summary_text(manifest) + f"FAILURE: {exc!r}\n", encoding="utf-8")
        logger.exception("analysis failed")
        raise


def _finish_partial(run_dir: Path, manifest: dict[str, Any], completed: tuple[str, ...]) -> RunResult:
    manifest["status"] = "stopped"; manifest["output_inventory"] = _inventory(run_dir)
    json_write(manifest, run_dir / "run_manifest.json")
    (run_dir / "run_summary.txt").write_text(_summary_text(manifest), encoding="utf-8")
    return RunResult(run_dir, manifest, tuple(completed))
