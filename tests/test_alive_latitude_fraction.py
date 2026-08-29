from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import matplotlib
import numpy as np
import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses import (
    compute_alive_latitude_fraction,
)
from kinematicparcels.postprocessing.config import (
    AliveLatitudeFractionConfig,
    AliveLatitudeFractionPlottingConfig,
    DatasetConfig,
    OutputConfig,
    PostprocessConfig,
    load_postprocess_config,
)
from kinematicparcels.postprocessing.plotting import plot_alive_latitude_fraction
from kinematicparcels.postprocessing.plotting.alive_latitude_fraction import (
    _mask_values_at_or_below,
)
from kinematicparcels.postprocessing.workflows.run_alive_latitude_fraction import (
    run_alive_latitude_fraction,
)


def _trajectory_table(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["trajectory", "obs", "time", "lat"])


def test_load_config_parses_alive_latitude_fraction(tmp_path) -> None:
    config_path = tmp_path / "postprocess.yml"
    config_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types: [alive_latitude_fraction]
            alive_latitude_fraction:
              lat_min: -80
              lat_max: -30
              bin_width_deg: 2.5
              minimum_alive_tracers: 12
              time_axis: time
              resample_days: 0.5
              max_time_days: 30
              max_group_member: 3
              output:
                save_csv: true
                save_figure: false
              plotting:
                cmap: plasma
                vmin: 0.1
                vmax: 0.8
                min_mask_value: 0.05
                as_percent: false
                masked_color: silver
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(config_path).alive_latitude_fraction

    assert cfg.lat_min == -80.0
    assert cfg.lat_max == -30.0
    assert cfg.bin_width_deg == 2.5
    assert cfg.minimum_alive_tracers == 12
    assert cfg.time_axis == "time"
    assert cfg.resample_days == 0.5
    assert cfg.max_time_days == 30.0
    assert cfg.max_group_member == 3
    assert cfg.output.save_csv is True
    assert cfg.output.save_figure is False
    assert cfg.plotting.cmap == "plasma"
    assert cfg.plotting.vmin == 0.1
    assert cfg.plotting.vmax == 0.8
    assert cfg.plotting.min_mask_value == 0.05
    assert cfg.plotting.as_percent is False
    assert cfg.plotting.masked_color == "silver"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"lat_min": -91.0}, "latitude bounds"),
        ({"lat_max": 91.0}, "latitude bounds"),
        ({"bin_width_deg": 0.0}, "bin_width_deg"),
        ({"minimum_alive_tracers": 0}, "minimum_alive_tracers"),
        ({"time_axis": "release"}, "time_axis"),
        ({"resample_days": 0.0}, "resample_days"),
        ({"max_time_days": 0.0}, "max_time_days"),
        ({"max_time_days": -1.0}, "max_time_days"),
        ({"max_group_member": 0}, "max_group_member"),
    ],
)
def test_config_rejects_invalid_values(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        AliveLatitudeFractionConfig(**changes)


def test_plotting_config_rejects_invalid_limits_and_names() -> None:
    with pytest.raises(ValueError, match="cmap"):
        AliveLatitudeFractionPlottingConfig(cmap="")
    with pytest.raises(ValueError, match="masked_color"):
        AliveLatitudeFractionPlottingConfig(masked_color="")
    with pytest.raises(ValueError, match="between 0 and 1"):
        AliveLatitudeFractionPlottingConfig(vmax=1.1)
    with pytest.raises(ValueError, match="min_mask_value"):
        AliveLatitudeFractionPlottingConfig(min_mask_value=-0.1)
    with pytest.raises(ValueError, match="min_mask_value"):
        AliveLatitudeFractionPlottingConfig(min_mask_value=1.1)
    with pytest.raises(ValueError, match="vmin"):
        AliveLatitudeFractionPlottingConfig(vmin=0.8, vmax=0.2)


def test_native_age_fraction_uses_all_alive_tracers_as_denominator() -> None:
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", -75.0),
            ("a", 1, "2020-01-02", -65.0),
            ("a", 2, "2020-01-03", -55.0),
            ("b", 0, "2020-01-02", -75.0),
            ("b", 1, "2020-01-03", -45.0),
            ("c", 0, "2020-01-01", -20.0),
            ("c", 1, "2020-01-02", -20.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-80.0,
        lat_max=-40.0,
        bin_width_deg=20.0,
        minimum_alive_tracers=2,
        time_axis="age",
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    np.testing.assert_allclose(result.age_days.values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(result.alive_tracer_count.values, [3, 3, 1])
    np.testing.assert_array_equal(
        result.latitude_bin_count.values,
        [[2, 0], [1, 1], [0, 1]],
    )
    np.testing.assert_allclose(
        result.alive_tracer_fraction.values[:2],
        [[2.0 / 3.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]],
    )
    assert np.isnan(result.alive_tracer_fraction.values[2]).all()
    np.testing.assert_array_equal(
        result.meets_minimum_alive.values, [True, True, False]
    )


def test_max_time_days_crops_native_absolute_time_inclusively() -> None:
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-02", 1.0),
            ("a", 2, "2020-01-03", 2.0),
            ("a", 3, "2020-01-04", 3.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=5.0,
        time_axis="time",
        max_time_days=2.0,
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    assert pd.DatetimeIndex(result.time.values).tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]


def test_max_time_days_crops_signed_ages_symmetrically() -> None:
    table = _trajectory_table(
        [
            ("forward", 0, "2020-01-01", 0.0),
            ("forward", 1, "2020-01-02", 1.0),
            ("forward", 2, "2020-01-04", 3.0),
            ("backward", 0, "2020-01-04", 0.0),
            ("backward", 1, "2020-01-03", -1.0),
            ("backward", 2, "2020-01-01", -3.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=5.0,
        time_axis="age",
        max_time_days=2.0,
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    np.testing.assert_allclose(result.age_days.values, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(result.alive_tracer_count.values, [1, 2, 1])


def test_time_resampling_interpolates_without_extrapolation() -> None:
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-03", 20.0),
            ("b", 0, "2020-01-02", 0.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=25.0,
        bin_width_deg=10.0,
        time_axis="time",
        resample_days=1.0,
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    np.testing.assert_array_equal(result.alive_tracer_count.values, [1, 2, 1])
    np.testing.assert_array_equal(
        result.latitude_bin_count.values,
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
    )


def test_age_resampling_uses_out_of_crop_observation_to_interpolate() -> None:
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-04", 30.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=35.0,
        bin_width_deg=10.0,
        time_axis="age",
        resample_days=1.0,
        max_time_days=2.0,
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    np.testing.assert_allclose(result.age_days.values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        result.latitude_bin_count.values,
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    )


def test_backward_age_resampling_uses_signed_regular_axis() -> None:
    table = _trajectory_table(
        [
            ("backward", 0, "2020-01-04", 30.0),
            ("backward", 1, "2020-01-01", 0.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=35.0,
        bin_width_deg=10.0,
        time_axis="age",
        resample_days=1.0,
        max_time_days=2.0,
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    np.testing.assert_allclose(result.age_days.values, [-2.0, -1.0, 0.0])
    np.testing.assert_array_equal(
        result.latitude_bin_count.values,
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )


def test_partial_final_bin_includes_lat_max_and_outside_stays_in_denominator() -> None:
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("b", 0, "2020-01-01", 1.0),
            ("c", 0, "2020-01-01", 2.0),
            ("d", 0, "2020-01-01", 2.5),
            ("e", 0, "2020-01-01", -1.0),
            ("f", 0, "2020-01-01", 3.0),
        ]
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=0.0,
        lat_max=2.5,
        bin_width_deg=1.0,
        time_axis="age",
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    np.testing.assert_allclose(result.lat_upper.values, [1.0, 2.0, 2.5])
    np.testing.assert_array_equal(result.latitude_bin_count.values[0], [1, 1, 2])
    assert result.alive_tracer_count.values[0] == 6
    np.testing.assert_allclose(
        result.alive_tracer_fraction.values[0], [1 / 6, 1 / 6, 2 / 6]
    )


def test_group_member_cutoff_is_applied_before_counting() -> None:
    table = pd.DataFrame(
        {
            "trajectory": [1, 1],
            "group_member": [1, 2],
            "obs": [0, 0],
            "time": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "lat": [0.0, 10.0],
        }
    )
    cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=15.0,
        bin_width_deg=10.0,
        max_group_member=1,
    )

    result = compute_alive_latitude_fraction(table, cfg=cfg)

    assert result.alive_tracer_count.values[0] == 1
    np.testing.assert_array_equal(result.latitude_bin_count.values[0], [1, 0])


def test_conflicting_duplicate_and_nonmonotonic_particle_time_raise() -> None:
    duplicate = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-01", 1.0),
        ]
    )
    with pytest.raises(ValueError, match="conflicting latitudes"):
        compute_alive_latitude_fraction(
            duplicate, cfg=AliveLatitudeFractionConfig()
        )

    nonmonotonic = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-03", 1.0),
            ("a", 2, "2020-01-02", 2.0),
        ]
    )
    with pytest.raises(ValueError, match="monotonic time ordering"):
        compute_alive_latitude_fraction(
            nonmonotonic, cfg=AliveLatitudeFractionConfig()
        )


def test_workflow_writes_long_csv_and_heatmap(tmp_path, monkeypatch) -> None:
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-02", 1.0),
        ]
    )
    analysis_cfg = AliveLatitudeFractionConfig(
        lat_min=-5.0,
        lat_max=5.0,
        time_axis="age",
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="unused.zarr"),
        output=OutputConfig(output_dir=str(tmp_path)),
        alive_latitude_fraction=analysis_cfg,
    )
    monkeypatch.setattr(
        "kinematicparcels.postprocessing.workflows.run_alive_latitude_fraction.get_trajectory_table",
        lambda _cfg, _context: table,
    )
    plot_call = {}

    def _capture_plot(_dataset, *, outpath, **kwargs):
        plot_call.update(kwargs)
        Path(outpath).touch()

    monkeypatch.setattr(
        "kinematicparcels.postprocessing.workflows.run_alive_latitude_fraction.plot_alive_latitude_fraction",
        _capture_plot,
    )

    run_alive_latitude_fraction(cfg, {})

    csv_path = tmp_path / "alive_latitude_fraction.csv"
    png_path = tmp_path / "alive_latitude_fraction.png"
    assert csv_path.exists()
    assert png_path.exists()
    assert plot_call["as_percent"] is True
    assert plot_call["vmin"] == 0.0
    assert plot_call["cmap"] == "viridis"
    assert plot_call["min_mask_value"] is None
    exported = pd.read_csv(csv_path)
    assert list(exported.columns) == [
        "age_days",
        "latitude_bin",
        "lat_lower",
        "lat_center",
        "lat_upper",
        "latitude_bin_count",
        "alive_tracer_count",
        "alive_tracer_fraction",
        "meets_minimum_alive",
    ]
    assert exported["alive_tracer_fraction"].max() == pytest.approx(1.0)


def test_workflow_writes_empty_csv_and_skips_unsupported_heatmap(
    tmp_path, monkeypatch
) -> None:
    table = _trajectory_table([("a", 0, "2020-01-01", 0.0)])
    analysis_cfg = AliveLatitudeFractionConfig(
        minimum_alive_tracers=2,
        output=replace(AliveLatitudeFractionConfig().output, save_figure=True),
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="unused.zarr"),
        output=OutputConfig(output_dir=str(tmp_path)),
        alive_latitude_fraction=analysis_cfg,
    )
    monkeypatch.setattr(
        "kinematicparcels.postprocessing.workflows.run_alive_latitude_fraction.get_trajectory_table",
        lambda _cfg, _context: table,
    )

    with pytest.warns(RuntimeWarning, match="minimum tracer support"):
        run_alive_latitude_fraction(cfg, {})

    assert (tmp_path / "alive_latitude_fraction.csv").exists()
    assert not (tmp_path / "alive_latitude_fraction.png").exists()


def test_plotter_writes_time_heatmap_with_headless_backend(tmp_path) -> None:
    matplotlib.use("Agg", force=True)
    table = _trajectory_table(
        [
            ("a", 0, "2020-01-01", 0.0),
            ("a", 1, "2020-01-02", 1.0),
        ]
    )
    result = compute_alive_latitude_fraction(
        table,
        cfg=AliveLatitudeFractionConfig(
            lat_min=-5.0,
            lat_max=5.0,
            time_axis="time",
        ),
    )
    output = tmp_path / "heatmap.png"

    plot_alive_latitude_fraction(
        result,
        outpath=output,
        as_percent=True,
        masked_color="silver",
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_mask_threshold_is_inclusive_and_preserves_existing_nan() -> None:
    values = np.array([[0.0, 0.1, 0.10001, np.nan]])

    masked = _mask_values_at_or_below(values, min_mask_value=0.1)

    assert np.isnan(masked[0, 0])
    assert np.isnan(masked[0, 1])
    assert masked[0, 2] == pytest.approx(0.10001)
    assert np.isnan(masked[0, 3])
