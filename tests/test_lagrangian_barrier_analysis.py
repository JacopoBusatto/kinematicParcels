from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from research.lagrangian_barriers.annotations import (
    annotate_external_front_distances, annotate_seed_family,
)
from research.lagrangian_barriers.build_branch_geometry import (
    build_branch_geometry, decompose_directed_graph,
)
from research.lagrangian_barriers.common import circular_difference_degrees
from research.lagrangian_barriers.compute_transition_geometry import compute_transition_geometry
from research.lagrangian_barriers.config import (
    BarrierAnalysisConfig, BarriersConfig, BranchesConfig, GeometryConfig, GridConfig,
    InputConfig, ModesConfig, OutputConfig, PermeabilityConfig, PlottingConfig, load_config,
)
from research.lagrangian_barriers.connect_barrier_segments import (
    connect_barrier_segments, detect_barrier_candidates,
)
from research.lagrangian_barriers.detect_transport_modes import circular_pdf_peaks, detect_transport_modes
from research.lagrangian_barriers.diagnose_cross_branch_permeability import (
    diagnose_cross_branch_permeability, wilson_interval,
)
from research.lagrangian_barriers.exports import _split_antimeridian, save_dataset_netcdf
from research.lagrangian_barriers.pipeline import run_analysis
from research.lagrangian_barriers.plot_lagrangian_barriers import _grid_map, map_projection
from research.lagrangian_barriers.synthetic import sparse_transition_table, zonal_corridor
from research.lagrangian_barriers.validate_transition_matrix import validate_transition_matrix


def grid() -> GridConfig:
    return GridConfig(lon_min=-5, lon_max=5, lat_min=-5, lat_max=5, dlon=1, dlat=1,
                      periodic_longitude=False)


def prepared(table: pd.DataFrame, g: GridConfig | None = None):
    g = g or grid()
    validation = validate_transition_matrix(table, g, BarrierAnalysisConfig(
        input=InputConfig("unused"), output=OutputConfig("unused"), grid=g,
    ).validation)
    assert not validation.errors
    return compute_transition_geometry(validation.transitions, g, GeometryConfig(), ModesConfig())


def profile(values: list[float], *, point_order=0, branch="b", s=0) -> pd.DataFrame:
    offsets = np.arange(len(values), dtype=float) * 25 - 25 * (len(values) // 2)
    frame = pd.DataFrame({
        "branch_id": branch, "branch_point_id": f"{branch}:p{point_order}",
        "point_order": point_order, "s_km": s, "offset_km": offsets,
        "candidate_lon": offsets / 100 + s / 111.0, "candidate_lat": 0.0,
        "P_cross": values, "P_minus_to_plus": values, "P_plus_to_minus": values,
        "P_cross_moving": values, "directional_asymmetry": 0.0,
        "counts_minus": 200, "counts_plus": 200,
        "P_cross_ci_low": np.maximum(0, np.asarray(values)-.02),
        "P_cross_ci_high": np.minimum(1, np.asarray(values)+.02),
        "support_valid": True,
    })
    return frame


def save_profile_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3)); ax.plot(frame.offset_km, frame.P_cross, marker="o")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def test_strict_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("input: {transition_table: x}\noutput: {root: y}\nunknown: 1\n")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)


def test_plot_projection_is_strict_and_supports_south_polar_stereo(tmp_path):
    plotting = PlottingConfig(
        projection="SouthPolarStereo", central_longitude=15,
        circular_boundary=True, draw_coastlines=False,
    )
    assert isinstance(map_projection(plotting), ccrs.SouthPolarStereo)

    with pytest.raises(ValueError, match="plotting.projection"):
        BarrierAnalysisConfig(
            input=InputConfig("unused"), output=OutputConfig("unused"),
            plotting=PlottingConfig(projection="NotAProjection"),
        )


def test_regular_grid_map_uses_configured_polar_projection(tmp_path):
    polar_grid = GridConfig(
        lon_min=-180, lon_max=180, lat_min=-90, lat_max=-30,
        dlon=30, dlat=10,
    )
    cells = pd.DataFrame({
        "start_lon_bin": [0, 3, 7, 11],
        "start_lat_bin": [1, 2, 3, 4],
        "N_i": [10, 20, 30, 40],
    })
    output = tmp_path / "polar_grid.png"
    _grid_map(
        cells, "N_i", "Polar gridded field", output, polar_grid,
        PlottingConfig(
            projection="SouthPolarStereo", circular_boundary=True,
            draw_coastlines=False, dpi=60,
        ),
    )
    assert output.exists() and output.stat().st_size > 0


def test_eager_netcdf_write_does_not_probe_distributed_scheduler(tmp_path, monkeypatch):
    import xarray as xr
    from xarray.backends import locks, writers

    def unexpected_scheduler_probe(*args, **kwargs):
        raise AssertionError("eager NetCDF export must not probe Distributed")

    monkeypatch.setattr(writers, "get_dask_scheduler", unexpected_scheduler_probe)
    monkeypatch.setattr(locks, "get_dask_scheduler", unexpected_scheduler_probe)
    output = tmp_path / "eager.nc"
    save_dataset_netcdf(xr.Dataset({"value": (("x",), np.array([1.0, 2.0]))}), output)
    with xr.open_dataset(output) as reopened:
        np.testing.assert_array_equal(reopened.value.values, [1.0, 2.0])


def test_validation_fails_without_renormalizing():
    table = sparse_transition_table(grid(), [(2, 5, 3, 5, 10), (2, 5, 2, 5, 10)])
    table.loc[0, "transition_probability"] = .2
    result = validate_transition_matrix(table, grid(), BarrierAnalysisConfig(
        input=InputConfig("x"), output=OutputConfig("y"), grid=grid()).validation)
    assert "row_normalization_failure" in result.errors
    assert .2 in result.transitions.transition_probability.to_list()


def test_dateline_transition_geometry_is_short_and_eastward():
    g = GridConfig(lon_min=-180, lon_max=180, lat_min=-1, lat_max=1,
                   dlon=1, dlat=1, periodic_longitude=True)
    table = sparse_transition_table(g, [(359, 1, 0, 1, 20)])
    result = prepared(table, g)
    row = result.transitions.iloc[0]
    assert row.distance_km < 120
    assert float(circular_difference_degrees(row.bearing_deg, 90)) < 1


def test_periodic_mode_detector_and_multimodal_mean():
    bearings = np.asarray([355, 0, 5, 115, 120, 125], float)
    weights = np.ones(6) / 6
    _, _, peaks, _ = circular_pdf_peaks(bearings, weights, replace(
        ModesConfig(), smoothing_bandwidth_degrees=8, min_peak_separation_degrees=30,
    ))
    peak_bearings = (peaks + .5) * 5
    assert np.min(circular_difference_degrees(peak_bearings, 0)) < 10
    assert np.min(circular_difference_degrees(peak_bearings, 120)) < 10


def test_mode_detector_preserves_two_pathways():
    table = sparse_transition_table(grid(), [
        (4, 5, 5, 5, 15), (4, 5, 4, 6, 15), (4, 5, 4, 5, 5),
    ])
    geo = prepared(table)
    config = replace(ModesConfig(), smoothing_bandwidth_degrees=8,
                     min_relative_prominence=.01, min_mode_probability=.1)
    result = detect_transport_modes(geo.transitions, config)
    assert len(result.modes) == 2
    assert result.membership.loc[~result.membership.is_stay, "mode_id"].notna().all()


def test_branch_decomposition_preserves_split_merge_and_gap():
    graph = nx.DiGraph([("a","b"),("b","c"),("b","d"),("c","e"),("d","e")])
    paths = decompose_directed_graph(graph)
    assert {p for p in paths} == {("a","b"),("b","c","e"),("b","d","e")}
    detached = nx.DiGraph([("a","b"),("x","y")])
    assert len(decompose_directed_graph(detached)) == 2


def test_cycle_and_curved_branch_have_deterministic_left_normals():
    graph = nx.DiGraph([("a","b"),("b","c"),("c","a")])
    nodes = pd.DataFrame({
        "mode_id":["a","b","c"], "start_lon":[0,1,1], "start_lat":[0,0,1],
    })
    result = build_branch_geometry(graph, nodes, GeometryConfig(),
                                   replace(BranchesConfig(), sample_spacing_km=50,
                                           smoothing_window=3, smoothing_order=1))
    assert len(result.paths) == 1
    points = result.points
    np.testing.assert_allclose(points.tangent_x*points.normal_x + points.tangent_y*points.normal_y, 0, atol=1e-8)
    np.testing.assert_allclose(points.normal_x, -points.tangent_y)
    assert {"node_id", "local_support_count", "local_support_probability"} <= set(points)


def test_permeability_counts_direction_and_stay_separately():
    g = grid()
    table = sparse_transition_table(g, [
        (4,4,4,6,30), (4,4,4,4,20),
        (5,4,5,6,30), (5,4,5,4,20),
        (4,6,4,4,10), (4,6,4,6,40),
        (5,6,5,4,10), (5,6,5,6,40),
    ])
    geo = prepared(table, g)
    branch = pd.DataFrame({
        "branch_id":["b"], "branch_point_id":["b:p0"], "point_order":[0], "s_km":[0.],
        "lon":[0.], "lat":[0.], "bearing_deg":[90.], "tangent_x":[1.], "tangent_y":[0.],
        "normal_x":[0.], "normal_y":[1.], "radius_curvature_km":[np.inf],
        "nearest_other_branch_km":[np.inf], "self_proximity_km":[np.inf],
    })
    config = replace(PermeabilityConfig(), min_offset_km=0, max_offset_km=1,
                     offset_spacing_km=1, min_counts_per_side=1,
                     min_moving_counts_per_side=1, min_source_cells_per_side=1,
                     source_along_halfwidth_km=300, source_normal_halfwidth_km=300,
                     save_contributions=False)
    result = diagnose_cross_branch_permeability(branch, geo.transitions, g, GeometryConfig(), config)
    row = result.cross_sections.iloc[0]
    assert row.P_minus_to_plus > row.P_plus_to_minus
    assert row.P_cross_moving > row.P_cross
    assert row.noncross_count_minus + row.cross_count_minus_to_plus == row.counts_minus


@pytest.mark.parametrize("name,values,expected", [
    ("no_barrier", [.2,.2,.2,.2,.2], 0),
    ("core_barrier", [.3,.2,.05,.2,.3], 1),
    ("flank_barriers", [.3,.15,.05,.15,.3,.15,.05,.15,.3], 2),
])
def test_barrier_profiles(name, values, expected, tmp_path):
    frame = profile(values)
    save_profile_figure(frame, tmp_path / f"{name}.png")
    all_candidates, selected = detect_barrier_candidates(
        frame, replace(BarriersConfig(), min_prominence=.02), PermeabilityConfig())
    assert len(selected) == expected
    assert (tmp_path / f"{name}.png").exists()
    assert len(all_candidates) >= expected


def test_directional_barrier_and_confidence_interval():
    low, high = wilson_interval(10, 100)
    assert low < .1 < high
    frame = profile([.3,.15,.05,.15,.3])
    frame["P_minus_to_plus"] = frame.P_cross * 1.5
    frame["P_plus_to_minus"] = frame.P_cross * .5
    _, selected = detect_barrier_candidates(frame, BarriersConfig(), PermeabilityConfig())
    assert len(selected) == 1
    assert selected.iloc[0].P_minus_to_plus != selected.iloc[0].P_plus_to_minus


def test_barrier_connection_keeps_a_recorded_sampling_gap():
    frames=[]
    for order in (0,1,3,4):
        frame=profile([.3,.1,.3],point_order=order,s=order*50)
        _, selected=detect_barrier_candidates(frame,BarriersConfig(),PermeabilityConfig())
        frames.append(selected)
    points, summary=connect_barrier_segments(pd.concat(frames,ignore_index=True),GeometryConfig(),BarriersConfig())
    robust=summary.loc[summary.robust_segment]
    assert len(robust)==1 and robust.iloc[0].gap_count==1
    assert points.geometry_part.max()==1


def test_antimeridian_geojson_parts_do_not_bridge():
    parts = _split_antimeridian([(179, -50), (-179, -50), (-178, -50)])
    assert len(parts) == 1  # singleton pre-split part is intentionally omitted
    assert parts[0][0][0] == -179


def test_post_detection_annotation_hooks_do_not_change_geometry(tmp_path):
    points=pd.DataFrame({"branch_id":["b1","b2"],"component_id":["c1","c2"],
                         "branch_point_id":["p1","p2"],"lon":[0.,10.],"lat":[-50.,-40.]})
    selected=annotate_seed_family(points,lon_min=-1,lon_max=1,lat_min=-51,lat_max=-49)
    assert selected.set_index("branch_id").seed_family_selected.to_dict()=={"b1":True,"b2":False}
    front=tmp_path/"front.geojson"
    front.write_text('{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":[[-1,-49],[1,-49]]}}]}')
    distances=annotate_external_front_distances(points.iloc[[0]],front)
    assert distances.iloc[0].front_distance_km > 100
    assert points.lon.to_list()==[0.,10.]


def test_pipeline_manifest_and_no_overwrite(tmp_path):
    g=grid(); table=zonal_corridor(g)
    path=tmp_path/"table.parquet"; table.to_parquet(path,index=False)
    cfg=BarrierAnalysisConfig(input=InputConfig(str(path)),output=OutputConfig(str(tmp_path/"runs"),"test"),grid=g,
                              modes=replace(ModesConfig(),min_relative_prominence=0,min_mode_probability=.05))
    result=run_analysis(cfg,stop_after="modes")
    assert result.manifest["status"]=="stopped"
    assert (result.run_dir/"modes"/"mode_membership.parquet").exists()
    with pytest.raises(FileNotFoundError):
        run_analysis(replace(cfg,output=OutputConfig(str(tmp_path/"empty"),"test")),resume=True)


def test_end_to_end_synthetic_corridor_writes_all_stage_products(tmp_path):
    g=grid(); path=tmp_path/"table.parquet"; zonal_corridor(g).to_parquet(path,index=False)
    cfg=BarrierAnalysisConfig(
        input=InputConfig(str(path)),output=OutputConfig(str(tmp_path/"runs"),"full"),grid=g,
        modes=replace(ModesConfig(),min_relative_prominence=0,min_mode_probability=.05),
        permeability=replace(PermeabilityConfig(),min_offset_km=-50,max_offset_km=50,
                             offset_spacing_km=50,save_contributions=True),
        plotting=PlottingConfig(enabled=False),
    )
    result=run_analysis(cfg)
    assert result.manifest["status"]=="complete"
    assert all(item["status"]=="complete" for item in result.manifest["stages"].values())
    assert (result.run_dir/"graph"/"branch_graph.graphml").exists()
    assert (result.run_dir/"graph"/"mode_graph_all.graphml").exists()
    assert (result.run_dir/"barriers"/"barrier_segments.geojson").exists()
