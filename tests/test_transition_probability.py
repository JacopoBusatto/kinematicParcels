from __future__ import annotations

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.transition_probability import compute_transition_probability
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import TransitionProbabilityConfig
from kinematicparcels.postprocessing.plotting import transition_probability as transition_probability_plotting
from kinematicparcels.postprocessing.plotting.transition_probability import (
    plot_transition_probability_by_source,
    plot_transition_probability_overview,
)
from kinematicparcels.regions import Region, RegionManager


def test_load_postprocess_config_parses_transition_probability_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_transition_probability.yml"
    cfg_path.write_text(
                dedent(
                        """
                        dataset:
                            input_path: ./dummy.zarr
                        analysis:
                            types:
                                - transition_probability
                        transition_probability:
                            region_labels:
                                - sesc-mod
                                - sesc-sir
                            time_step_stride: 3
                            how_many: priority_max
                            priority_level: 7
                            priority_mode: exact
                            input_lon_mode: "-180_180"
                            min_life_days: 4
                            trimming_age_days: 10
                            max_group_member: 2
                            filter_isolated: true
                            plotting:
                                enabled: true
                                x_log_scale: true
                                y_log_scale: true
                                colormap: Paired
                        """
                ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("transition_probability",)
    assert cfg.transition_probability.region_labels == ("sesc-mod", "sesc-sir")
    assert cfg.transition_probability.time_step_stride == 3
    assert cfg.transition_probability.priority_level == 7
    assert cfg.transition_probability.min_life_days == 4
    assert cfg.transition_probability.trimming_age_days == 10
    assert cfg.transition_probability.max_group_member == 2
    assert cfg.transition_probability.filter_isolated is True
    assert cfg.transition_probability.plotting.enabled is True
    assert cfg.transition_probability.plotting.x_log_scale is True
    assert cfg.transition_probability.plotting.y_log_scale is True
    assert cfg.transition_probability.plotting.colormap == "Paired"


def test_transition_probability_plotting_writes_overview_and_source_plots(tmp_path) -> None:
    transition_table = pd.DataFrame(
        {
            "age_days": [0.0, 1.0, 2.0],
            "p_r1__r1": [1.0, 0.5, 0.0],
            "p_r1__r2": [0.0, 0.5, 1.0],
            "p_r2__r1": [0.0, 0.0, 1.0],
            "p_r2__r2": [1.0, 1.0, 0.0],
        }
    )

    overview_path = plot_transition_probability_overview(
        transition_table,
        region_labels=["r1", "r2"],
        outpath=tmp_path / "transition_probability_plot.png",
        x_log_scale=True,
        y_log_scale=True,
        colormap="Paired",
    )
    source_paths = plot_transition_probability_by_source(
        transition_table,
        region_labels=["r1", "r2"],
        outdir=tmp_path,
        x_log_scale=True,
        y_log_scale=True,
        colormap="Paired",
    )

    assert overview_path.exists()
    assert [path.name for path in source_paths] == [
        "transition_probability_r1_plot.png",
        "transition_probability_r2_plot.png",
    ]
    assert all(path.exists() for path in source_paths)


def test_transition_probability_plotting_uses_shared_log_y_limits(tmp_path, monkeypatch) -> None:
    transition_table = pd.DataFrame(
        {
            "age_days": [0.0, 1.0, 2.0, 3.0],
            "p_r1__r1": [1.0, 0.25, 0.05, 0.0],
            "p_r1__r2": [0.0, 0.75, 0.95, 1.0],
            "p_r2__r1": [0.0, 0.02, 0.2, 0.5],
            "p_r2__r2": [1.0, 0.98, 0.8, 0.5],
        }
    )
    captured_limits: dict[str, tuple[float, float]] = {}
    original_save = transition_probability_plotting._save_figure

    def _capture_limits(fig, outpath) -> None:
        captured_limits[outpath.name] = fig.axes[0].get_ylim()
        original_save(fig, outpath)

    monkeypatch.setattr(transition_probability_plotting, "_save_figure", _capture_limits)

    overview_path = plot_transition_probability_overview(
        transition_table,
        region_labels=["r1", "r2"],
        outpath=tmp_path / "transition_probability_plot.png",
        x_log_scale=True,
        y_log_scale=True,
        colormap="Paired",
    )
    source_paths = plot_transition_probability_by_source(
        transition_table,
        region_labels=["r1", "r2"],
        outdir=tmp_path,
        x_log_scale=True,
        y_log_scale=True,
        colormap="Paired",
    )

    expected_limits = pytest.approx((1.0e-2, 1.0))
    assert overview_path.exists()
    assert all(path.exists() for path in source_paths)
    assert captured_limits["transition_probability_plot.png"] == expected_limits
    assert captured_limits["transition_probability_r1_plot.png"] == expected_limits
    assert captured_limits["transition_probability_r2_plot.png"] == expected_limits


def test_transition_probability_plotting_adds_reference_fraction_lines(tmp_path, monkeypatch) -> None:
    transition_table = pd.DataFrame(
        {
            "age_days": [0.0, 1.0, 2.0],
            "represented_fraction_total": [1.0, 2.0 / 3.0, 2.0 / 3.0],
            "n_r1": [2, 2, 2],
            "n_r2": [1, 1, 1],
            "p_r1__r1": [1.0, 0.5, 0.0],
            "p_r1__r2": [0.0, 0.0, 0.5],
            "p_r2__r1": [0.0, 0.0, 0.0],
            "p_r2__r2": [1.0, 1.0, 1.0],
        }
    )
    captured_lines: dict[str, list[np.ndarray]] = {}
    original_save = transition_probability_plotting._save_figure

    def _capture_lines(fig, outpath) -> None:
        black_lines = []
        for line in fig.axes[0].lines:
            if line.get_color() == "black" and line.get_linestyle() == "-":
                black_lines.append(np.asarray(line.get_ydata(), dtype=float))
        captured_lines[outpath.name] = black_lines
        original_save(fig, outpath)

    monkeypatch.setattr(transition_probability_plotting, "_save_figure", _capture_lines)

    plot_transition_probability_overview(
        transition_table,
        region_labels=["r1", "r2"],
        outpath=tmp_path / "transition_probability_plot.png",
    )
    plot_transition_probability_by_source(
        transition_table,
        region_labels=["r1", "r2"],
        outdir=tmp_path,
    )

    assert len(captured_lines["transition_probability_plot.png"]) == 1
    assert captured_lines["transition_probability_plot.png"][0] == pytest.approx([1.0, 2.0 / 3.0, 2.0 / 3.0])
    assert captured_lines["transition_probability_r1_plot.png"][0] == pytest.approx([1.0, 0.5, 0.5])
    assert captured_lines["transition_probability_r2_plot.png"][0] == pytest.approx([1.0, 1.0, 1.0])


def _region_manager(priority_r2: int = 1) -> RegionManager:
    return RegionManager(
        [
            Region(
                name="Region 1",
                label="r1",
                numericLabel=1,
                polygons=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
                priority=1,
            ),
            Region(
                name="Region 2",
                label="r2",
                numericLabel=2,
                polygons=[[(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]],
                priority=priority_r2,
            ),
        ]
    )


def _base_cfg(**kwargs) -> TransitionProbabilityConfig:
    return TransitionProbabilityConfig(
        region_labels=("r1", "r2"),
        **kwargs,
    )


def _trajectory_rows(
    trajectory: str,
    coords: list[tuple[float, float]],
    *,
    group_member: int | None = None,
    start_time: str = "2026-01-01",
 ) -> list[dict]:
    times = pd.date_range(start_time, periods=len(coords), freq="1D")
    rows: list[dict] = []
    for obs, ((lon, lat), time) in enumerate(zip(coords, times, strict=True)):
        row = {
            "trajectory": trajectory,
            "obs": obs,
            "time": time,
            "lon": lon,
            "lat": lat,
        }
        if group_member is not None:
            row["group_member"] = group_member
        rows.append(row)
    return rows


def test_compute_transition_probability_counts_and_excludes_outside_starts() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (0.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (2.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("c", [(5.0, 5.0), (2.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("d", [(2.5, 0.5), (2.5, 0.5), (0.5, 0.5)])
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(),
        region_manager=_region_manager(),
    )

    assert result["age_days"].tolist() == [0.0, 1.0, 2.0]
    assert result["represented_fraction_total"].tolist() == [1.0, 1.0, 1.0]
    assert result["n_r1"].tolist() == [2, 2, 2]
    assert result["n_r2"].tolist() == [1, 1, 1]
    assert result["p_r1__r1"].tolist() == [1.0, 0.5, 0.0]
    assert result["p_r1__r2"].tolist() == [0.0, 0.5, 1.0]
    assert result["p_r2__r1"].tolist() == [0.0, 0.0, 1.0]
    assert result["p_r2__r2"].tolist() == [1.0, 1.0, 0.0]


def test_compute_transition_probability_exports_weighted_total_represented_fraction() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (0.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (5.0, 5.0), (5.0, 5.0)])
        + _trajectory_rows("c", [(2.5, 0.5), (2.5, 0.5), (2.5, 0.5)])
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(),
        region_manager=_region_manager(),
    )

    assert result["age_days"].tolist() == [0.0, 1.0, 2.0]
    assert result["n_r1"].tolist() == [2, 2, 2]
    assert result["n_r2"].tolist() == [1, 1, 1]
    assert result["represented_fraction_total"].tolist() == pytest.approx([1.0, 2.0 / 3.0, 2.0 / 3.0])
    assert result["p_r1__r1"].tolist() == [1.0, 0.5, 0.0]
    assert result["p_r1__r2"].tolist() == [0.0, 0.0, 0.5]
    assert result["p_r2__r2"].tolist() == [1.0, 1.0, 1.0]


def test_compute_transition_probability_aligns_unsynchronized_starts_on_age_axis() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (2.5, 0.5)], start_time="2026-01-01")
        + _trajectory_rows("b", [(0.6, 0.5), (2.5, 0.5)], start_time="2026-01-05")
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(),
        region_manager=_region_manager(),
    )

    assert result["age_days"].tolist() == [0.0, 1.0]
    assert result["p_r1__r1"].tolist() == [1.0, 0.0]
    assert result["p_r1__r2"].tolist() == [0.0, 1.0]


def test_compute_transition_probability_filter_isolated_reclassifies_single_symbol() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (2.5, 0.5), (0.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (0.6, 0.5), (0.6, 0.5)])
    )

    no_filter = compute_transition_probability(
        df,
        cfg=_base_cfg(filter_isolated=False),
        region_manager=_region_manager(),
    )
    filtered = compute_transition_probability(
        df,
        cfg=_base_cfg(filter_isolated=True),
        region_manager=_region_manager(),
    )

    assert no_filter.loc[1, "p_r1__r2"] == 0.5
    assert filtered.loc[1, "p_r1__r2"] == 0.0
    assert filtered.loc[1, "p_r1__r1"] == 1.0


def test_compute_transition_probability_supports_group_member_filter_and_stride() -> None:
    df = pd.DataFrame(
        _trajectory_rows("g1", [(0.5, 0.5), (0.5, 0.5), (2.5, 0.5)], group_member=1)
        + _trajectory_rows("g1", [(2.5, 0.5), (2.5, 0.5), (0.5, 0.5)], group_member=2)
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(time_step_stride=2, max_group_member=1),
        region_manager=_region_manager(),
    )

    assert result["age_days"].tolist() == [0.0, 2.0]
    assert result["p_r1__r1"].tolist() == [1.0, 0.0]
    assert result["p_r1__r2"].tolist() == [0.0, 1.0]
    assert result["p_r2__r1"].isna().all()
    assert result["p_r2__r2"].isna().all()


def test_compute_transition_probability_filters_by_min_life_and_trims_without_interpolation() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (0.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (2.5, 0.5)])
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(min_life_days=2.0, trimming_age_days=1.5),
        region_manager=_region_manager(),
    )

    assert result["age_days"].tolist() == [0.0, 1.0]
    assert result["p_r1__r1"].tolist() == [1.0, 1.0]
    assert result["p_r1__r2"].tolist() == [0.0, 0.0]


def test_compute_transition_probability_trimming_does_not_imply_constant_denominator() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (0.6, 0.5), (2.5, 0.5)])
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(trimming_age_days=2.0),
        region_manager=_region_manager(),
    )

    assert result["age_days"].tolist() == [0.0, 1.0, 2.0]
    assert result["p_r1__r2"].tolist() == [0.0, 0.5, 0.5]


def test_compute_transition_probability_warns_on_mixed_priorities() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (0.5, 0.5)])
        + _trajectory_rows("b", [(2.5, 0.5), (2.5, 0.5)])
    )

    with pytest.warns(UserWarning, match="priority"):
        compute_transition_probability(
            df,
            cfg=_base_cfg(),
            region_manager=_region_manager(priority_r2=2),
        )