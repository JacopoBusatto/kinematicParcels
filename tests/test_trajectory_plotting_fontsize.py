from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest
from cartopy.mpl.geoaxes import GeoAxes

matplotlib.use("Agg", force=True)

import kinematicparcels.postprocessing.animations.trajectories as animation_plotting
import kinematicparcels.postprocessing.plotting.trajectories as trajectory_plotting
import kinematicparcels.postprocessing.workflows.quicklook as quicklook_workflow
import kinematicparcels.postprocessing.workflows.run_start_end_regions as start_end_workflow
import kinematicparcels.postprocessing.workflows.run_trajectories as trajectories_workflow
from kinematicparcels.postprocessing.config.models import (
    DatasetConfig,
    OutputConfig,
    PlottingConfig,
    PostprocessConfig,
    TrajectoriesConfig,
)


def _trajectory_table() -> pd.DataFrame:
    return pd.DataFrame(
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
            "value": [1.0, 1.0, 2.0, 2.0],
        }
    )


def _particle_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trajectory": [1, 2],
            "lon0": [14.30, 15.10],
            "lat0": [36.90, 37.20],
            "lonf": [14.45, 15.22],
            "latf": [36.82, 37.26],
            "value": [1.0, 2.0],
        }
    )


def _assert_font_kwargs(kwargs: dict[str, object]) -> None:
    assert kwargs["title_fontsize"] == 17
    assert kwargs["colorbar_fontsize"] == 13
    assert kwargs["colorbar_tick_fontsize"] == 11
    assert kwargs["axis_tick_fontsize"] == 14


def test_plot_trajectories_map_applies_font_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_figures = []
    captured_gridliners = []
    original_close = trajectory_plotting.plt.close
    original_gridlines = GeoAxes.gridlines
    monkeypatch.setattr(
        trajectory_plotting.plt,
        "close",
        lambda figure: (
            captured_figures.append(figure)
            if hasattr(figure, "axes")
            else original_close(figure)
        ),
    )
    monkeypatch.setattr(
        GeoAxes,
        "gridlines",
        lambda axis, *args, **kwargs: captured_gridliners.append(
            original_gridlines(axis, *args, **kwargs)
        )
        or captured_gridliners[-1],
    )

    trajectory_plotting.plot_trajectories_map(
        _trajectory_table(),
        outpath=tmp_path / "trajectories.png",
        title="Sized trajectories",
        summary_df=_particle_summary(),
        color_by="value",
        title_fontsize=17,
        colorbar_fontsize=13,
        colorbar_tick_fontsize=11,
        axis_tick_fontsize=14,
    )

    assert len(captured_figures) == 1
    figure = captured_figures[0]
    map_axis, colorbar_axis = figure.axes
    assert map_axis.get_title() == "Sized trajectories"
    assert map_axis.title.get_fontsize() == pytest.approx(17)
    assert captured_gridliners[0].xlabel_style == {"size": 14}
    assert captured_gridliners[0].ylabel_style == {"size": 14}
    assert colorbar_axis.yaxis.label.get_fontsize() == pytest.approx(13)
    assert all(
        tick.get_fontsize() == pytest.approx(11)
        for tick in colorbar_axis.get_yticklabels()
    )

    original_close(figure)


def test_plot_trajectories_map_omits_title_at_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_figures = []
    original_close = trajectory_plotting.plt.close
    monkeypatch.setattr(
        trajectory_plotting.plt,
        "close",
        lambda figure: (
            captured_figures.append(figure)
            if hasattr(figure, "axes")
            else original_close(figure)
        ),
    )

    trajectory_plotting.plot_trajectories_map(
        _trajectory_table(),
        outpath=tmp_path / "trajectories_without_title.png",
        title="This title must be omitted",
        title_fontsize=0,
    )

    figure = captured_figures[0]
    assert figure.axes[0].get_title() == ""
    original_close(figure)


def test_plot_connectivity_map_applies_axis_font_and_omits_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_figures = []
    captured_gridliners = []
    original_close = trajectory_plotting.plt.close
    original_gridlines = GeoAxes.gridlines
    monkeypatch.setattr(
        trajectory_plotting.plt,
        "close",
        lambda figure: (
            captured_figures.append(figure)
            if hasattr(figure, "axes")
            else original_close(figure)
        ),
    )
    monkeypatch.setattr(
        GeoAxes,
        "gridlines",
        lambda axis, *args, **kwargs: captured_gridliners.append(
            original_gridlines(axis, *args, **kwargs)
        )
        or captured_gridliners[-1],
    )
    summary = _particle_summary().assign(
        start_region=["west", "east"],
        end_region=["east", "west"],
    )

    trajectory_plotting.plot_connectivity_map(
        _trajectory_table(),
        summary,
        outpath=tmp_path / "connectivity.png",
        title="This title must be omitted",
        title_fontsize=0,
        axis_tick_fontsize=14,
    )

    assert len(captured_figures) == 1
    assert captured_figures[0].axes[0].get_title() == ""
    assert captured_gridliners[0].xlabel_style == {"size": 14}
    assert captured_gridliners[0].ylabel_style == {"size": 14}
    original_close(captured_figures[0])


def test_animate_trajectories_applies_fonts_and_omits_zero_sized_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_figures = []
    captured_gridliners = []
    original_close = animation_plotting.plt.close
    original_gridlines = GeoAxes.gridlines
    monkeypatch.setattr(
        animation_plotting.plt,
        "close",
        lambda figure: (
            captured_figures.append(figure)
            if hasattr(figure, "axes")
            else original_close(figure)
        ),
    )
    monkeypatch.setattr(
        GeoAxes,
        "gridlines",
        lambda axis, *args, **kwargs: captured_gridliners.append(
            original_gridlines(axis, *args, **kwargs)
        )
        or captured_gridliners[-1],
    )

    animation_plotting.animate_trajectories(
        _trajectory_table(),
        outpath=tmp_path / "trajectories.gif",
        title="This title must be omitted",
        color_by="value",
        show_time_bar=False,
        trail=False,
        title_fontsize=0,
        colorbar_fontsize=13,
        colorbar_tick_fontsize=11,
        axis_tick_fontsize=14,
    )

    assert captured_figures
    for figure, gridliner in zip(captured_figures, captured_gridliners, strict=True):
        map_axis, colorbar_axis = figure.axes
        assert map_axis.get_title() == ""
        assert gridliner.xlabel_style == {"size": 14}
        assert gridliner.ylabel_style == {"size": 14}
        assert colorbar_axis.yaxis.label.get_fontsize() == pytest.approx(13)
        assert all(
            tick.get_fontsize() == pytest.approx(11)
            for tick in colorbar_axis.get_yticklabels()
        )
        original_close(figure)


def test_run_trajectories_forwards_font_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_calls: list[dict[str, object]] = []
    animation_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        trajectories_workflow,
        "get_trajectory_table",
        lambda cfg, context: _trajectory_table(),
    )
    monkeypatch.setattr(
        trajectories_workflow,
        "get_particle_summary",
        lambda cfg, context: _particle_summary(),
    )
    monkeypatch.setattr(
        trajectories_workflow,
        "plot_trajectories_map",
        lambda *args, **kwargs: static_calls.append(kwargs),
    )
    monkeypatch.setattr(
        trajectories_workflow,
        "animate_trajectories",
        lambda *args, **kwargs: animation_calls.append(kwargs),
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="unused.zarr"),
        output=OutputConfig(output_dir=str(tmp_path)),
        trajectories=TrajectoriesConfig(
            plot=True,
            animate=True,
            plot_color_by="value",
            animation_color_by="value",
        ),
        plotting=PlottingConfig(
            title_fontsize=17,
            colorbar_fontsize=13,
            colorbar_tick_fontsize=11,
            axis_tick_fontsize=14,
        ),
    )

    trajectories_workflow.run_trajectories(cfg, {})

    assert len(static_calls) == 1
    assert len(animation_calls) == 1
    _assert_font_kwargs(static_calls[0])
    _assert_font_kwargs(animation_calls[0])


def test_start_end_connectivity_forwards_font_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _particle_summary()
    classified = summary.assign(
        start_region=["west", "east"],
        end_region=["east", "west"],
    )
    trajectory_calls: list[dict[str, object]] = []
    connectivity_calls: list[dict[str, object]] = []
    animation_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        start_end_workflow,
        "get_particle_summary",
        lambda cfg, context: summary,
    )
    monkeypatch.setattr(
        start_end_workflow,
        "get_trajectory_table",
        lambda cfg, context: _trajectory_table(),
    )
    monkeypatch.setattr(start_end_workflow, "build_region_manager", lambda **kwargs: object())
    monkeypatch.setattr(
        start_end_workflow,
        "classify_start_end_regions",
        lambda *args, **kwargs: classified,
    )
    monkeypatch.setattr(
        start_end_workflow,
        "compute_mode_region_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"trajectory": [1, 2], "mode_region": ["west", "east"]}
        ),
    )
    monkeypatch.setattr(start_end_workflow, "save_grid_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        start_end_workflow,
        "plot_trajectories_map",
        lambda *args, **kwargs: trajectory_calls.append(kwargs),
    )
    monkeypatch.setattr(
        start_end_workflow,
        "plot_connectivity_map",
        lambda *args, **kwargs: connectivity_calls.append(kwargs),
    )
    monkeypatch.setattr(
        start_end_workflow,
        "animate_trajectories",
        lambda *args, **kwargs: animation_calls.append(kwargs),
    )

    cfg = SimpleNamespace(
        output=SimpleNamespace(output_dir=str(tmp_path)),
        exports=SimpleNamespace(table_format="csv"),
        release=SimpleNamespace(mode="point_list"),
        plotting=SimpleNamespace(
            projection="PlateCarree",
            title_fontsize=17,
            colorbar_fontsize=13,
            colorbar_tick_fontsize=11,
            axis_tick_fontsize=14,
        ),
        trajectories=SimpleNamespace(
            alpha=0.7,
            max_group_member=None,
            animation_fps=2,
            animation_vmin=None,
            animation_vmax=None,
            animation_cmap=None,
            animation_cmap_mode="auto",
            show_time_bar=False,
            trail=False,
            trail_steps=None,
        ),
        start_end_regions=SimpleNamespace(
            region_labels=None,
            infer_grid_from_start=True,
            how_many="priority_max",
            priority_level=None,
            priority_mode="exact",
            input_lon_mode="-180_180",
            plot=False,
            plot_connectivity=True,
            animate_connectivity=True,
            connectivity_segments=True,
            connectivity_color_by="start_region",
            connectivity_label="region",
            connectivity_title="Trajectories by region",
            connectivity_show_start=True,
            connectivity_show_end=True,
            connectivity_alpha=None,
            connectivity_max_group_member=None,
            connectivity_animation_fps=None,
            connectivity_animation_show_tracer=None,
            connectivity_trail=None,
            connectivity_trail_steps=None,
        ),
    )

    start_end_workflow.run_start_end_regions(cfg, {})

    assert len(trajectory_calls) == 1
    assert len(connectivity_calls) == 1
    assert len(animation_calls) == 1
    _assert_font_kwargs(trajectory_calls[0])
    _assert_font_kwargs(connectivity_calls[0])
    _assert_font_kwargs(animation_calls[0])


def test_quicklook_forwards_optional_font_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        quicklook_workflow,
        "load_trajectory_table",
        lambda *args, **kwargs: _trajectory_table(),
    )
    monkeypatch.setattr(
        quicklook_workflow,
        "build_particle_summary",
        lambda dataframe: _particle_summary(),
    )
    monkeypatch.setattr(
        quicklook_workflow,
        "plot_trajectories_map",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    quicklook_workflow.quicklook_trajectories(
        "unused.zarr",
        tmp_path / "quicklook.png",
        title_fontsize=17,
        colorbar_fontsize=13,
        colorbar_tick_fontsize=11,
        axis_tick_fontsize=14,
    )

    assert len(calls) == 1
    _assert_font_kwargs(calls[0])
