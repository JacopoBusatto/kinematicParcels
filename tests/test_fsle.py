from __future__ import annotations

import math
from textwrap import dedent

import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.fsle import (
    EARTH_RADIUS_KM,
    build_fsle_pair_trajectories,
    compute_fsle,
)
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.plotting.fsle import _build_reference_lines, plot_fsle_spectrum


def _lon_offset_for_distance_km(distance_km: float) -> float:
    return math.degrees(distance_km / EARTH_RADIUS_KM)


def _build_group_rows(
    *,
    group_id: int,
    distances_km: list[float],
    times: list[str],
    group_size: int = 2,
) -> list[dict]:
    rows: list[dict] = []
    for obs, (distance_km, time_value) in enumerate(zip(distances_km, times)):
        rows.append(
            {
                "trajectory": f"{group_id}_m1",
                "group_id": group_id,
                "group_member": 1,
                "group_size": group_size,
                "obs": obs,
                "time": pd.Timestamp(time_value),
                "lon": 0.0,
                "lat": 0.0,
            }
        )
        rows.append(
            {
                "trajectory": f"{group_id}_m2",
                "group_id": group_id,
                "group_member": 2,
                "group_size": group_size,
                "obs": obs,
                "time": pd.Timestamp(time_value),
                "lon": _lon_offset_for_distance_km(distance_km),
                "lat": 0.0,
            }
        )
    return rows


def test_build_fsle_pair_trajectories_respects_pair_mode() -> None:
    times = ["2026-01-01", "2026-01-02"]
    rows = []
    for obs, time_value in enumerate(times):
        for member, distance_km in ((1, 0.0), (2, 1.0), (3, 2.0)):
            rows.append(
                {
                    "trajectory": f"1_m{member}",
                    "group_id": 1,
                    "group_member": member,
                    "group_size": 3,
                    "obs": obs,
                    "time": pd.Timestamp(time_value),
                    "lon": _lon_offset_for_distance_km(distance_km),
                    "lat": 0.0,
                }
            )
    df = pd.DataFrame(rows)

    center_pairs = build_fsle_pair_trajectories(df, pair_mode="center_pairs")
    all_pairs = build_fsle_pair_trajectories(df, pair_mode="all_pairs")

    assert set(center_pairs["pair_id"].unique()) == {"1_m1_m2", "1_m1_m3"}
    assert set(all_pairs["pair_id"].unique()) == {"1_m1_m2", "1_m1_m3", "1_m2_m3"}


def test_build_fsle_pair_trajectories_can_ignore_zonal_separation() -> None:
    df = pd.DataFrame(
        [
            {
                "trajectory": "1_m1",
                "group_id": 1,
                "group_member": 1,
                "group_size": 2,
                "obs": 0,
                "time": pd.Timestamp("2026-01-01"),
                "lon": 0.0,
                "lat": 10.0,
            },
            {
                "trajectory": "1_m2",
                "group_id": 1,
                "group_member": 2,
                "group_size": 2,
                "obs": 0,
                "time": pd.Timestamp("2026-01-01"),
                "lon": 1.0,
                "lat": 11.0,
            },
        ]
    )

    default_pairs = build_fsle_pair_trajectories(df)
    meridional_pairs = build_fsle_pair_trajectories(df, meridional_only=True)

    expected_meridional = EARTH_RADIUS_KM * math.radians(1.0)

    assert default_pairs.loc[0, "distance_km"] > meridional_pairs.loc[0, "distance_km"]
    assert meridional_pairs.loc[0, "distance_km"] == pytest.approx(expected_meridional)


def test_compute_fsle_rejects_single_trajectory_output() -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["1", "1"],
            "group_id": [1, 1],
            "group_member": [1, 1],
            "group_size": [1, 1],
            "obs": [0, 1],
            "time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "lon": [0.0, 0.0],
            "lat": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="group_size > 1"):
        compute_fsle(df)


def test_compute_fsle_can_use_meridional_distance_only() -> None:
    df = pd.DataFrame(
        [
            {
                "trajectory": "1_m1",
                "group_id": 1,
                "group_member": 1,
                "group_size": 2,
                "obs": 0,
                "time": pd.Timestamp("2026-01-01"),
                "lon": 0.0,
                "lat": 0.0,
            },
            {
                "trajectory": "1_m2",
                "group_id": 1,
                "group_member": 2,
                "group_size": 2,
                "obs": 0,
                "time": pd.Timestamp("2026-01-01"),
                "lon": _lon_offset_for_distance_km(3.0),
                "lat": 0.0,
            },
            {
                "trajectory": "1_m1",
                "group_id": 1,
                "group_member": 1,
                "group_size": 2,
                "obs": 1,
                "time": pd.Timestamp("2026-01-02"),
                "lon": 0.0,
                "lat": 0.0,
            },
            {
                "trajectory": "1_m2",
                "group_id": 1,
                "group_member": 2,
                "group_size": 2,
                "obs": 1,
                "time": pd.Timestamp("2026-01-02"),
                "lon": _lon_offset_for_distance_km(3.0),
                "lat": math.degrees(2.0 / EARTH_RADIUS_KM),
            },
            {
                "trajectory": "1_m1",
                "group_id": 1,
                "group_member": 1,
                "group_size": 2,
                "obs": 2,
                "time": pd.Timestamp("2026-01-03"),
                "lon": 0.0,
                "lat": 0.0,
            },
            {
                "trajectory": "1_m2",
                "group_id": 1,
                "group_member": 2,
                "group_size": 2,
                "obs": 2,
                "time": pd.Timestamp("2026-01-03"),
                "lon": _lon_offset_for_distance_km(3.0),
                "lat": math.degrees(4.0 / EARTH_RADIUS_KM),
            },
        ]
    )

    default_result = compute_fsle(
        df,
        min_scale=1.0,
        max_scale=3.0,
        rho_increment=2.0,
    )
    meridional_result = compute_fsle(
        df,
        meridional_only=True,
        min_scale=1.0,
        max_scale=3.0,
        rho_increment=2.0,
    )

    assert default_result.crossing_events.empty
    assert len(meridional_result.crossing_events) == 1
    assert meridional_result.spectrum.iloc[0]["fsle"] == pytest.approx(math.log(2.0))


def test_load_postprocess_config_parses_fsle_meridional_only_flag(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_fsle.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - fsle
            fsle:
              pair_mode: all_pairs
              meridional_only: true
              min_scale: 0.01
              max_scale: 5.0
              rho_increment: 1.4142135623730951
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("fsle",)
    assert cfg.fsle.pair_mode == "all_pairs"
    assert cfg.fsle.meridional_only is True


def test_compute_fsle_uses_overshoot_lambda_and_configurable_error_factor() -> None:
    rows = []
    rows.extend(
        _build_group_rows(
            group_id=1,
            distances_km=[0.8, 1.2, 2.4],
            times=["2026-01-01", "2026-01-02", "2026-01-03"],
        )
    )
    rows.extend(
        _build_group_rows(
            group_id=2,
            distances_km=[0.9, 1.1, 2.5],
            times=["2026-01-01", "2026-01-03", "2026-01-06"],
        )
    )
    df = pd.DataFrame(rows)

    result = compute_fsle(
        df,
        min_scale=1.0,
        max_scale=2.1,
        rho_increment=2.0,
    )

    assert len(result.crossing_events) == 2
    assert len(result.spectrum) == 1

    spectrum_row = result.spectrum.iloc[0]
    expected_mean_log = (math.log(2.4 / 1.2) + math.log(2.5 / 1.1)) / 2.0
    expected_mean_tau = (1.0 + 3.0) / 2.0
    expected_fsle = expected_mean_log / expected_mean_tau

    expected_mean_inv_tau = (1.0 / 1.0 + 1.0 / 3.0) / 2.0
    variance_factor = (expected_mean_inv_tau * expected_mean_tau - 1.0) / (expected_mean_tau ** 2)
    expected_sigma = math.log(2.0) * math.sqrt(variance_factor)
    expected_std = expected_sigma / math.sqrt(2.0)

    assert spectrum_row["scale"] == pytest.approx(1.0)
    assert spectrum_row["mean_log_ratio"] == pytest.approx(expected_mean_log)
    assert spectrum_row["mean_time_delta_days"] == pytest.approx(expected_mean_tau)
    assert spectrum_row["fsle"] == pytest.approx(expected_fsle)
    assert spectrum_row["sigma"] == pytest.approx(expected_sigma)
    assert spectrum_row["std"] == pytest.approx(expected_std)


def test_build_reference_lines_supports_slope_selection_and_anchor_scales() -> None:
    spectrum = pd.DataFrame(
        {
            "scale": [1.0, 2.0, 4.0],
            "fsle": [3.0, 1.5, 0.75],
        }
    )

    lines = _build_reference_lines(
        spectrum,
        reference_slopes=("delta^-1",),
        reference_slope_anchor_scales={"delta^-1": 3.1},
    )

    assert len(lines) == 1
    assert lines[0]["slope_id"] == "delta^-1"
    assert lines[0]["x_anchor"] == pytest.approx(4.0)
    assert lines[0]["y_anchor"] == pytest.approx(0.75)
    assert lines[0]["y"] == pytest.approx([3.0, 1.5, 0.75])


def test_plot_fsle_spectrum_accepts_manual_reference_slope_controls(tmp_path) -> None:
    spectrum = pd.DataFrame(
        {
            "scale": [1.0, 2.0, 4.0],
            "fsle": [3.0, 1.5, 0.75],
            "std": [0.1, 0.1, 0.1],
        }
    )

    outpath = tmp_path / "fsle.png"
    plot_fsle_spectrum(
        spectrum,
        outpath=outpath,
        reference_slopes=("delta^-1",),
        reference_slope_anchor_scales={"delta^-1": 2.1},
    )

    assert outpath.exists()
    assert outpath.stat().st_size > 0