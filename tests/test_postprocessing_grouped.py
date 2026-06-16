from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import kinematicparcels.postprocessing.analyses.start_end_regions as start_end_regions_analysis

from kinematicparcels.postprocessing.analyses.density import compute_time_density
from kinematicparcels.postprocessing.analyses.start_end_regions import (
    compute_mode_region_map,
    compute_mode_region_summary,
    compute_start_end_region_maps,
)
from kinematicparcels.postprocessing.animations.trajectories import _resolve_trail_color, animate_trajectories
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import DatasetConfig, DatasetCoordinatesConfig, ExportsConfig, GridConfig, OutputConfig, PostprocessConfig
from kinematicparcels.postprocessing.core.gridding import RegularGrid, build_grid_from_config
from kinematicparcels.postprocessing.core.summaries import build_particle_summary
from kinematicparcels.postprocessing.io.parcels import sanitize_trajectories
from kinematicparcels.postprocessing.plotting.trajectories import (
    _split_longitude_wrapped_path,
    plot_trajectories_map,
)
from kinematicparcels.postprocessing.plotting.maps import plot_discrete_grid_map
from kinematicparcels.postprocessing.workflows.run_start_end_regions import _prepare_region_trajectory_inputs
from kinematicparcels.postprocessing.workflows.run_summary import run_summary


def test_build_particle_summary_separates_group_members() -> None:
    df = pd.DataFrame(
        {
            "trajectory": [10, 10, 10, 10],
            "group_member": [1, 1, 2, 2],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [12.00, 12.10, 12.20, 12.30],
            "lat": [37.00, 37.05, 37.20, 37.25],
        }
    )

    summary = build_particle_summary(df)

    assert len(summary) == 2
    assert set(summary["group_member"]) == {1, 2}
    assert set(summary["lat0"]) == {37.0, 37.2}


def test_build_particle_summary_preserves_circle_id() -> None:
    df = pd.DataFrame(
        {
            "trajectory": [10, 10, 11, 11],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [12.00, 12.10, 12.20, 12.30],
            "lat": [37.00, 37.05, 37.20, 37.25],
            "circle_id": [1, 1, 2, 2],
        }
    )

    summary = build_particle_summary(df)

    assert "circle_id" in summary.columns
    assert summary["circle_id"].tolist() == [1, 2]


def test_resolve_trail_color_depends_on_tracer_visibility() -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    grey = _resolve_trail_color(show_tracer=True, color_code=0.0, cmap=cmap, norm=norm)
    colored = _resolve_trail_color(show_tracer=False, color_code=0.0, cmap=cmap, norm=norm)

    assert grey == "0.4"
    assert colored != "0.4"


def test_split_longitude_wrapped_path_breaks_antimeridian_jump() -> None:
    lon = np.array([178.0, 179.5, -179.8, -178.9])
    lat = np.array([-60.0, -60.2, -60.4, -60.6])

    plot_segments = _split_longitude_wrapped_path(lon, lat)

    assert len(plot_segments) == 2
    np.testing.assert_allclose(plot_segments[0][0], np.array([178.0, 179.5]))
    np.testing.assert_allclose(plot_segments[0][1], np.array([-60.0, -60.2]))
    np.testing.assert_allclose(plot_segments[1][0], np.array([-179.8, -178.9]))
    np.testing.assert_allclose(plot_segments[1][1], np.array([-60.4, -60.6]))


def test_split_longitude_wrapped_path_keeps_regular_path_contiguous() -> None:
    lon = np.array([10.0, 10.5, 11.0, 11.5])
    lat = np.array([35.0, 35.2, 35.4, 35.6])

    plot_segments = _split_longitude_wrapped_path(lon, lat)

    assert len(plot_segments) == 1
    np.testing.assert_allclose(plot_segments[0][0], lon)
    np.testing.assert_allclose(plot_segments[0][1], lat)



def test_compute_time_density_accepts_pandas_timestamps() -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["1_m1", "1_m1", "1_m2", "1_m2"],
            "group_member": [1, 1, 2, 2],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T12:00:00",
                    "2026-04-15T18:00:00",
                    "2026-04-15T12:00:00",
                    "2026-04-15T18:00:00",
                ]
            ),
            "lon": [12.0, 12.1, 12.2, 12.3],
            "lat": [37.0, 37.0, 37.1, 37.1],
        }
    )

    grid = RegularGrid(
        lon_min=11.9,
        lon_max=12.4,
        lat_min=36.9,
        lat_max=37.2,
        dlon=0.1,
        dlat=0.1,
    )

    table, ds = compute_time_density(df, grid=grid)

    assert len(table) == 4
    assert ds.sizes["time"] == 2
    assert float(table["particle_count"].sum()) == 4.0


def test_run_density_group_member_filter() -> None:
    """DensityConfig.group_member filters rows before density is computed."""
    from kinematicparcels.postprocessing.config.models import DensityConfig

    df = pd.DataFrame(
        {
            "trajectory": ["1_m1", "1_m1", "1_m2", "1_m2"],
            "group_member": [1, 1, 2, 2],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T12:00:00",
                    "2026-04-15T18:00:00",
                    "2026-04-15T12:00:00",
                    "2026-04-15T18:00:00",
                ]
            ),
            "lon": [12.0, 12.1, 12.2, 12.3],
            "lat": [37.0, 37.0, 37.1, 37.1],
        }
    )

    grid = RegularGrid(
        lon_min=11.9,
        lon_max=12.4,
        lat_min=36.9,
        lat_max=37.2,
        dlon=0.1,
        dlat=0.1,
    )

    # Filter to member 1 only: expect 2 rows (one per timestep), count == 2
    member1_only = df.loc[df["group_member"] == 1].copy()
    table, ds = compute_time_density(member1_only, grid=grid)

    assert float(table["particle_count"].sum()) == 2.0
    assert ds.sizes["time"] == 2

    # DensityConfig.group_member=None (default) includes all members
    cfg_default = DensityConfig()
    assert cfg_default.group_member is None

    # DensityConfig.group_member=1 restricts to member 1
    cfg_member1 = DensityConfig(group_member=1)
    assert cfg_member1.group_member == 1


def test_plot_trajectories_map_accepts_categorical_summary_coloring(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["1", "1", "2", "2"],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [14.30, 14.45, 15.10, 15.22],
            "lat": [36.90, 36.82, 37.20, 37.26],
        }
    )
    summary = pd.DataFrame(
        {
            "trajectory": ["1", "2"],
            "start_region": ["sesc-mod", "sesc-sir"],
        }
    )

    outpath = tmp_path / "trajectories_by_start_region.png"
    plot_trajectories_map(
        df,
        outpath=outpath,
        title="Trajectories by start region",
        summary_df=summary,
        color_by="start_region",
        show_end=False,
    )

    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_plot_trajectories_map_accepts_array_like_summary_trajectory_ids(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trajectory": [1, 1, 2, 2],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [14.30, 14.45, 15.10, 15.22],
            "lat": [36.90, 36.82, 37.20, 37.26],
        }
    )
    summary = pd.DataFrame(
        {
            "trajectory": [np.array([1]), np.array([2])],
            "start_region": ["sesc-mod", "sesc-sir"],
        }
    )

    outpath = tmp_path / "trajectories_by_start_region_array_ids.png"
    plot_trajectories_map(
        df,
        outpath=outpath,
        title="Trajectories by start region",
        summary_df=summary,
        color_by="start_region",
        show_end=False,
    )

    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_animate_trajectories_accepts_categorical_summary_coloring(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["1", "1", "2", "2"],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [14.30, 14.45, 15.10, 15.22],
            "lat": [36.90, 36.82, 37.20, 37.26],
        }
    )
    summary = pd.DataFrame(
        {
            "trajectory": ["1", "2"],
            "start_region": ["sesc-mod", "sesc-sir"],
        }
    )

    outpath = tmp_path / "trajectories_by_start_region.gif"
    animate_trajectories(
        df,
        outpath=outpath,
        title="Trajectories by start region",
        summary_df=summary,
        color_by="start_region",
        colorbar_label="Start region",
        fps=2,
        show_time_bar=False,
        trail=False,
        show_tracer=False,
    )

    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_prepare_region_trajectory_inputs_filters_member1_and_labels() -> None:
    traj = pd.DataFrame(
        {
            "trajectory": ["1_m1", "1_m1", "1_m2", "1_m2", "2_m1", "2_m1"],
            "group_member": [1, 1, 2, 2, 1, 1],
            "obs": [0, 1, 0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [14.1, 14.2, 14.1, 14.2, 15.1, 15.2],
            "lat": [36.8, 36.9, 36.8, 36.9, 37.0, 37.1],
        }
    )
    summary = pd.DataFrame(
        {
            "trajectory": ["1_m1", "1_m2", "2_m1"],
            "group_member": [1, 2, 1],
            "start_region": ["sesc-mod", "sesc-mod", "sesc-sir"],
        }
    )

    traj_plot, summary_plot, labels = _prepare_region_trajectory_inputs(
        traj,
        summary,
        color_by="start_region",
        max_group_member=1,
    )

    assert sorted(traj_plot["group_member"].unique().tolist()) == [1]
    assert sorted(summary_plot["group_member"].unique().tolist()) == [1]
    assert labels == ["sesc-mod", "sesc-sir"]



def test_run_summary_preserves_group_member_for_grouped_datasets(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {
            "time": (("trajectory", "obs"), np.array([
                ["2026-04-15T00:00:00", "2026-04-15T06:00:00"],
                ["2026-04-15T00:00:00", "2026-04-15T06:00:00"],
            ], dtype="datetime64[ns]")),
            "lon": (("trajectory", "obs"), np.array([[14.0, 14.1], [14.2, 14.3]])),
            "lat": (("trajectory", "obs"), np.array([[36.8, 36.9], [37.0, 37.1]])),
            "z": (("trajectory", "obs"), np.zeros((2, 2))),
            "group_size": (("trajectory", "obs"), np.array([[2, 2], [2, 2]])),
            "lon_1": (("trajectory", "obs"), np.array([[14.0, 14.1], [14.2, 14.3]])),
            "lat_1": (("trajectory", "obs"), np.array([[36.8, 36.9], [37.0, 37.1]])),
            "lon_2": (("trajectory", "obs"), np.array([[14.05, 14.15], [14.25, 14.35]])),
            "lat_2": (("trajectory", "obs"), np.array([[36.85, 36.95], [37.05, 37.15]])),
        },
        coords={"trajectory": [0, 1], "obs": [0, 1]},
    )
    dataset_path = tmp_path / "grouped.nc"
    ds.to_netcdf(dataset_path)

    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path=str(dataset_path)),
        output=OutputConfig(output_dir=str(tmp_path / "out")),
        exports=ExportsConfig(save_trajectory_table=True, save_particle_summary=True),
    )

    context: dict = {}
    run_summary(cfg, context)

    traj = context["trajectory_table"]
    summary = context["particle_summary"]

    assert "group_member" in traj.columns
    assert sorted(traj["group_member"].unique().tolist()) == [1, 2]
    assert "group_member" in summary.columns
    assert sorted(summary["group_member"].unique().tolist()) == [1, 2]


def test_run_summary_expands_member_five_when_present(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {
            "time": (("trajectory", "obs"), np.array([["2026-04-15T00:00:00"]], dtype="datetime64[ns]")),
            "lon": (("trajectory", "obs"), np.array([[14.0]])),
            "lat": (("trajectory", "obs"), np.array([[36.8]])),
            "z": (("trajectory", "obs"), np.zeros((1, 1))),
            "group_size": (("trajectory", "obs"), np.array([[5]])),
            "lon_1": (("trajectory", "obs"), np.array([[14.00]])),
            "lat_1": (("trajectory", "obs"), np.array([[36.80]])),
            "lon_2": (("trajectory", "obs"), np.array([[14.01]])),
            "lat_2": (("trajectory", "obs"), np.array([[36.81]])),
            "lon_3": (("trajectory", "obs"), np.array([[14.02]])),
            "lat_3": (("trajectory", "obs"), np.array([[36.82]])),
            "lon_4": (("trajectory", "obs"), np.array([[14.03]])),
            "lat_4": (("trajectory", "obs"), np.array([[36.83]])),
            "lon_5": (("trajectory", "obs"), np.array([[14.04]])),
            "lat_5": (("trajectory", "obs"), np.array([[36.84]])),
        },
        coords={"trajectory": [0], "obs": [0]},
    )
    dataset_path = tmp_path / "grouped5.nc"
    ds.to_netcdf(dataset_path)

    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path=str(dataset_path)),
        output=OutputConfig(output_dir=str(tmp_path / "out")),
        exports=ExportsConfig(save_trajectory_table=True, save_particle_summary=True),
    )

    context: dict = {}
    run_summary(cfg, context)

    traj = context["trajectory_table"]
    summary = context["particle_summary"]

    assert sorted(traj["group_member"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert sorted(summary["group_member"].unique().tolist()) == [1, 2, 3, 4, 5]


def test_sanitize_trajectories_drops_trajectory_if_first_obs_is_invalid() -> None:
    df = pd.DataFrame(
        {
            "trajectory": [10, 10, 10, 10, 11, 11],
            "obs": [0, 1, 2, 3, 0, 1],
            "time": pd.to_datetime(
                [
                    "2020-01-01T00:00:00",
                    "2020-01-11T00:00:00",
                    "2020-01-21T00:00:00",
                    "2020-01-31T00:00:00",
                    "2020-01-01T00:00:00",
                    "2020-01-11T00:00:00",
                ]
            ),
            "lon": [np.nan, 10.0, 11.0, np.nan, 20.0, 21.0],
            "lat": [np.nan, -55.0, -54.0, np.nan, -50.0, -49.0],
        }
    )

    cleaned = sanitize_trajectories(df)

    assert cleaned["trajectory"].unique().tolist() == [11]
    assert cleaned.loc[cleaned["trajectory"] == 11, "obs"].tolist() == [0, 1]


def test_build_grid_from_config_uses_member1_release_centers() -> None:
    df = pd.DataFrame(
        {
            "time0": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon0": [14.30, 14.3008, 14.35, 14.3508],
            "lat0": [36.90, 36.9008, 36.95, 36.9508],
            "group_member": [1, 2, 1, 2],
        }
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="dummy.zarr"),
        grid=GridConfig(
            mode="from_initial_centers",
            lon_min=14.0,
            lon_max=15.0,
            lat_min=36.5,
            lat_max=37.5,
            dlon=0.05,
            dlat=0.05,
        ),
    )

    grid = build_grid_from_config(
        cfg,
        df,
        lon_col="lon0",
        lat_col="lat0",
        time_col="time0",
    )

    assert grid.dlon == 0.05
    assert grid.dlat == 0.05
    assert grid.nlon < 100
    assert grid.nlat < 100


def test_load_postprocess_config_parses_connectivity_alpha(tmp_path: Path) -> None:
    cfg_path = tmp_path / "post.yml"
    cfg_path.write_text(
                """
                dataset:
                    input_path: ./dummy.zarr
                trajectories:
                    alpha: 0.35
                density:
                    group_member: 2
                start_end_regions:
                    connectivity_alpha: 0.25
                    connectivity_animation_show_tracer: false
                """,
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.trajectories.alpha == 0.35
    assert cfg.density.group_member == 2
    assert cfg.start_end_regions.connectivity_alpha == 0.25
    assert cfg.start_end_regions.connectivity_animation_show_tracer is False


def test_load_postprocess_config_parses_start_end_region_map_styling(tmp_path: Path) -> None:
    cfg_path = tmp_path / "post.yml"
    cfg_path.write_text(
                """
                dataset:
                    input_path: ./dummy.zarr
                start_end_regions:
                    discrete_cmap: Set3
                    colorbar_label_mode: region_label
                    show_region_labels: true
                """,
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.start_end_regions.discrete_cmap == "Set3"
    assert cfg.start_end_regions.colorbar_label_mode == "region_label"
    assert cfg.start_end_regions.show_region_labels is True


def test_plot_discrete_grid_map_accepts_label_modes_and_annotations(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {
            "start_numericLabel": (("lat", "lon"), np.array([[1.0, 2.0], [2.0, np.nan]]))
        },
        coords={
            "lat": np.array([36.9, 37.0]),
            "lon": np.array([14.3, 14.4]),
        },
    )

    outpath = tmp_path / "discrete_start_regions.png"
    plot_discrete_grid_map(
        ds,
        var_name="start_numericLabel",
        outpath=outpath,
        cmap_name="Set3",
        colorbar_label_mode="region_name",
        category_label_map={
            1: {"label": "sic", "name": "Sicily Channel"},
            2: {"label": "sesc", "name": "South East Sicily"},
        },
        show_labels=True,
    )

    assert outpath.exists()
    assert outpath.stat().st_size > 0



def test_compute_start_end_region_maps_prefers_highest_priority() -> None:
    classified = pd.DataFrame(
        {
            "lon0": [14.3, 14.3],
            "lat0": [36.9, 36.9],
            "start_numericLabel": [30.0, 1.0],
            "start_priority": [7.0, 6.0],
            "end_numericLabel": [11.0, 99.0],
            "end_priority": [4.0, 8.0],
        }
    )
    grid = RegularGrid(
        lon_min=14.25,
        lon_max=14.35,
        lat_min=36.85,
        lat_max=36.95,
        dlon=0.1,
        dlat=0.1,
    )

    start_table, _, end_table, _ = compute_start_end_region_maps(
        classified,
        grid=grid,
        lon_col="lon0",
        lat_col="lat0",
    )

    assert float(start_table["start_numericLabel"].iloc[0]) == 30.0
    assert float(end_table["end_numericLabel"].iloc[0]) == 99.0


def test_compute_start_end_region_maps_mode_for_continuous() -> None:
    classified = pd.DataFrame(
        {
            "lon0": [14.3, 14.3, 14.3],
            "lat0": [36.9, 36.9, 36.9],
            "start_numericLabel": [10.0, 10.0, 20.0],
            "start_priority": [1.0, 1.0, 9.0],
            "end_numericLabel": [90.0, 80.0, 90.0],
            "end_priority": [2.0, 2.0, 2.0],
        }
    )
    grid = RegularGrid(
        lon_min=14.25,
        lon_max=14.35,
        lat_min=36.85,
        lat_max=36.95,
        dlon=0.1,
        dlat=0.1,
    )

    start_table, _, end_table, _ = compute_start_end_region_maps(
        classified,
        grid=grid,
        lon_col="lon0",
        lat_col="lat0",
        use_mode=True,
    )

    assert float(start_table["start_numericLabel"].iloc[0]) == 10.0
    assert float(end_table["end_numericLabel"].iloc[0]) == 90.0


def test_compute_start_end_region_maps_mode_tie_returns_tied_value() -> None:
    classified = pd.DataFrame(
        {
            "lon0": [14.3, 14.3, 14.3, 14.3],
            "lat0": [36.9, 36.9, 36.9, 36.9],
            "start_numericLabel": [10.0, 10.0, 20.0, 20.0],
            "start_priority": [1.0, 1.0, 9.0, 9.0],
            "end_numericLabel": [80.0, 90.0, 80.0, 90.0],
            "end_priority": [2.0, 2.0, 2.0, 2.0],
        }
    )
    grid = RegularGrid(
        lon_min=14.25,
        lon_max=14.35,
        lat_min=36.85,
        lat_max=36.95,
        dlon=0.1,
        dlat=0.1,
    )

    start_table, _, end_table, _ = compute_start_end_region_maps(
        classified,
        grid=grid,
        lon_col="lon0",
        lat_col="lat0",
        use_mode=True,
    )

    assert float(start_table["start_numericLabel"].iloc[0]) in {10.0, 20.0}
    assert float(end_table["end_numericLabel"].iloc[0]) in {80.0, 90.0}


def test_compute_mode_region_summary_uses_label_mode_and_infers_metadata(monkeypatch) -> None:
    traj = pd.DataFrame(
        {
            "trajectory": [1, 1, 1, 2, 2],
            "obs": [0, 1, 2, 0, 1],
            "lon": [0.0, 0.1, 0.2, 1.0, 1.1],
            "lat": [0.0, 0.1, 0.2, 1.0, 1.1],
        }
    )

    def _fake_classify_region_points(df: pd.DataFrame, **kwargs):
        out = df.copy()
        out["mode_region_point"] = ["A", "A", "B", "B", "C"]
        out["_mode_numericLabel_point"] = [10.0, 10.0, 20.0, 20.0, 30.0]
        out["_mode_priority_point"] = [1.0, 1.0, 2.0, 2.0, 3.0]
        return out

    monkeypatch.setattr(start_end_regions_analysis, "classify_region_points", _fake_classify_region_points)

    class _FakeRegion:
        def __init__(self, label: str, numeric: int, priority: int):
            self.label = label
            self.NumericLabel = numeric
            self.priority = priority

    class _FakeManager:
        def get_regions(self):
            return [
                _FakeRegion("A", 10, 1),
                _FakeRegion("B", 20, 2),
                _FakeRegion("C", 30, 3),
            ]

    out = compute_mode_region_summary(
        traj,
        region_manager=_FakeManager(),
    )

    row_t1 = out.loc[out["trajectory"] == 1].iloc[0]
    row_t2 = out.loc[out["trajectory"] == 2].iloc[0]

    assert row_t1["mode_region"] == "A"
    assert float(row_t1["mode_numericLabel"]) == 10.0
    assert float(row_t1["mode_priority"]) == 1.0
    assert row_t2["mode_region"] in {"B", "C"}
    assert float(row_t2["mode_numericLabel"]) in {20.0, 30.0}
    assert float(row_t2["mode_priority"]) in {2.0, 3.0}


def test_compute_mode_region_map_builds_dataset() -> None:
    classified = pd.DataFrame(
        {
            "lon0": [14.3, 14.3, 14.31],
            "lat0": [36.9, 36.9, 36.91],
            "mode_numericLabel": [50.0, 50.0, 70.0],
            "mode_priority": [3.0, 3.0, 4.0],
        }
    )
    grid = RegularGrid(
        lon_min=14.25,
        lon_max=14.35,
        lat_min=36.85,
        lat_max=36.95,
        dlon=0.1,
        dlat=0.1,
    )

    table, ds = compute_mode_region_map(
        classified,
        grid=grid,
        lon_col="lon0",
        lat_col="lat0",
    )

    assert not table.empty
    assert "mode_numericLabel" in table.columns
    assert "mode_numericLabel" in ds.data_vars
