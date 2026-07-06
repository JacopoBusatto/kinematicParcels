from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels.tools.rtraj_to_zarr_v2 import (
    DepthBin,
    DepthBinConfig,
    IsolatedDepthBinConfig,
    JumpQcConfig,
    MergeConfig,
    MissingDepthConfig,
    OutputConfig,
    ParkingDepthConfig,
    RtrajV2Config,
    TrajectoryFixConfig,
    RegionSelectionConfig,
    ResampleConfig,
    _adjusted_time_with_raw_fallback,
    _count_nested_reasons,
    _filter_trajectory_fixes,
    _map_representative_pressure_to_observations,
    _resolve_parking_depth_to_observations,
    _select_cycle_representative_fixes,
    apply_jump_qc_segments,
    apply_depth_bin_segmentation,
    apply_region_selection_segments,
    apply_resampling_segments,
    prepare_output_trajectories,
    process_qc_stage,
)


def _config() -> RtrajV2Config:
    return RtrajV2Config(
        path=Path("RTRAJ_to_zarr_v2.yml"),
        raw={},
        mode="diagnostics",
        max_files=50,
        input_files=[],
        source_variables={
            "time": "JULD",
            "lon": "LONGITUDE",
            "lat": "LATITUDE",
            "depth": "PRES",
            "position_qc": "POSITION_QC",
            "time_qc": "JULD_QC",
        },
        normalized_variables={
            "time": "time",
            "lon": "lon",
            "lat": "lat",
            "depth": "depth",
        },
        trajectory_fixes=TrajectoryFixConfig(
            one_per_cycle=False,
            require_finite_cycle=True,
            prefer_valid_position_qc=True,
            prefer_valid_time_qc=True,
            prefer_repeated_position=True,
            position_round_decimals=5,
            tie_breaker="last",
        ),
        parking_depth=ParkingDepthConfig(
            mode="representative_park_pressure",
            fallback_value=1000.0,
            fill_missing=True,
            infer_from_park_window=True,
            pressure_variable="PRES_ADJUSTED",
            fallback_pressure_variable="PRES",
            percentile=95.0,
            min_pressure=50.0,
        ),
        qc={
            "enabled": True,
            "missing_qc": "fail",
            "variables": {
                "position_qc": {
                    "source": "POSITION_QC",
                    "valid_values": ["1", "2"],
                    "applies_to": ["lon", "lat"],
                },
                "time_qc": {
                    "source": "JULD_QC",
                    "valid_values": ["1", "2"],
                    "applies_to": ["time"],
                },
            },
        },
        merge=MergeConfig(
            enabled=True,
            max_gap_points=1,
            max_gap_duration_days=5.0,
            max_bridge_speed_m_per_s=100.0,
            max_bridge_vertical_rate_m_per_day=None,
        ),
        jump_qc=JumpQcConfig(
            enabled=False,
            max_speed_m_per_s=2.0,
            auto_drop_enabled=True,
            max_block_points=3,
            max_block_duration_days=10.0,
            split_remaining_jumps=True,
        ),
        depth_bins=DepthBinConfig(
            enabled=False,
            output_mode="per_bin",
            bins=(),
            missing_depth=MissingDepthConfig(
                strategy="bounded_neighbor",
                max_fill_points=2,
                fill_between_same_bin_only=True,
            ),
            isolated_outlier=IsolatedDepthBinConfig(
                enabled=False,
                max_run_points=1,
                require_same_neighbor_bin=True,
            ),
        ),
        region_selection=RegionSelectionConfig(
            names_or_labels=(),
            selection_mode="from_first_entry",
            input_lon_mode="-180_180",
        ),
        resample=ResampleConfig(
            frequency=None,
            interpolate="time",
            reference_time=None,
            shared_time=False,
            shift_start_to_reference=False,
            min_duration_days=None,
        ),
        output=OutputConfig(
            zarr_path=Path("output.zarr"),
            write_zarr=False,
            overwrite=False,
        ),
        min_segment_points=1,
        diagnostics_dir=Path("diagnostics"),
        diagnostics_formats=("png",),
    )


def _depth_config() -> RtrajV2Config:
    base = _config()
    return RtrajV2Config(
        path=base.path,
        raw=base.raw,
        mode=base.mode,
        max_files=base.max_files,
        input_files=base.input_files,
        source_variables=base.source_variables,
        normalized_variables=base.normalized_variables,
        trajectory_fixes=base.trajectory_fixes,
        parking_depth=base.parking_depth,
        qc=base.qc,
        merge=base.merge,
        jump_qc=base.jump_qc,
        depth_bins=DepthBinConfig(
            enabled=True,
            output_mode="per_bin",
            bins=(
                DepthBin(label="z0900_1100", min_value=900.0, max_value=1100.0),
                DepthBin(label="z2100_inf", min_value=2100.0, max_value=None),
            ),
            missing_depth=MissingDepthConfig(
                strategy="bounded_neighbor",
                max_fill_points=1,
                fill_between_same_bin_only=True,
            ),
            isolated_outlier=IsolatedDepthBinConfig(
                enabled=False,
                max_run_points=1,
                require_same_neighbor_bin=True,
            ),
        ),
        region_selection=base.region_selection,
        resample=base.resample,
        output=base.output,
        min_segment_points=base.min_segment_points,
        diagnostics_dir=base.diagnostics_dir,
        diagnostics_formats=base.diagnostics_formats,
    )


def _jump_config(*, min_segment_points: int = 1) -> RtrajV2Config:
    base = _config()
    return replace(
        base,
        min_segment_points=min_segment_points,
        jump_qc=JumpQcConfig(
            enabled=True,
            max_speed_m_per_s=1.0,
            auto_drop_enabled=True,
            max_block_points=3,
            max_block_duration_days=10.0,
            split_remaining_jumps=True,
        ),
    )


def _jump_segment(lon: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["example_Rtraj.nc"] * len(lon),
            "platform_code": [1902267] * len(lon),
            "source_index": list(range(len(lon))),
            "_qc_order": list(range(len(lon))),
            "time": pd.date_range("2000-01-01", periods=len(lon), freq="D"),
            "lon": lon,
            "lat": [0.0] * len(lon),
            "depth": [1000.0] * len(lon),
            "position_qc": ["1"] * len(lon),
            "time_qc": ["1"] * len(lon),
            "qc_keep": [True] * len(lon),
            "qc_drop_reasons": [""] * len(lon),
        }
    )


def test_qc_stage_merges_small_qc_gap_without_readding_bad_row() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["example_Rtraj.nc"] * 4,
            "platform_code": [1902267] * 4,
            "source_index": [0, 1, 2, 3],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]),
            "lon": [-70.0, -69.9, -69.8, -69.7],
            "lat": [-60.0, -60.1, -60.2, -60.3],
            "depth": [1000.0, 9999.0, 3000.0, 3001.0],
            "position_qc": ["1", "4", "1", "1"],
            "time_qc": ["1", "1", "1", "1"],
        }
    )

    result = process_qc_stage(raw, _config())

    assert result.summary["dropped_points"] == 1
    assert result.summary["initial_segments"] == 2
    assert result.summary["merged_segments"] == 1
    assert result.summary["merge_count"] == 1
    assert result.merged_segments[0]["source_index"].tolist() == [0, 2, 3]
    assert result.dropped["source_index"].tolist() == [1]


def test_adjusted_time_uses_raw_fallback_for_missing_values() -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.asarray([10.0, 11.0, 12.0])),
            "JULD_ADJUSTED": (("N_MEASUREMENT",), np.asarray([10.5, np.nan, 12.5])),
        }
    )

    time = _adjusted_time_with_raw_fallback(ds, "JULD")

    expected = pd.to_datetime(
        [
            "1950-01-11 12:00:00",
            "1950-01-12 00:00:00",
            "1950-01-13 12:00:00",
        ]
    ).astype("datetime64[ns]")
    pd.testing.assert_series_equal(time, pd.Series(expected), check_names=False)


def test_filter_trajectory_fixes_skips_rows_without_time_or_position() -> None:
    frame = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3, 4],
            "time": pd.to_datetime(["2000-01-02", "2000-01-03", None, "2000-01-01", "2000-01-04"]),
            "lon": [-70.0, np.nan, -69.8, -69.7, -69.6],
            "lat": [-60.0, -60.1, -60.2, -60.3, np.nan],
        }
    )

    fixes = _filter_trajectory_fixes(frame, {"time": "time", "lon": "lon", "lat": "lat"})

    assert fixes["source_index"].tolist() == [3, 0]
    assert fixes.attrs["raw_measurement_rows"] == 5
    assert fixes.attrs["trajectory_fix_rows"] == 2
    assert fixes.attrs["non_fix_rows"] == 3


def test_cycle_representative_prefers_repeated_position_before_last_tie_breaker() -> None:
    config = replace(
        _config(),
        trajectory_fixes=TrajectoryFixConfig(
            one_per_cycle=True,
            require_finite_cycle=True,
            prefer_valid_position_qc=True,
            prefer_valid_time_qc=True,
            prefer_repeated_position=True,
            position_round_decimals=5,
            tie_breaker="last",
        ),
    )
    frame = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3],
            "platform_code": [1902267] * 4,
            "cycle_number": [1.0, 1.0, 1.0, 2.0],
            "time": pd.to_datetime(["2000-01-01", "2000-01-01", "2000-01-01", "2000-01-11"]),
            "lon": [-37.0, -37.0, -35.0, -34.0],
            "lat": [-56.0, -56.0, -56.0, -55.0],
            "depth": [1000.0] * 4,
            "position_qc": ["1"] * 4,
            "time_qc": ["1"] * 4,
        }
    )
    frame.attrs["raw_measurement_rows"] = 4
    frame.attrs["finite_trajectory_fix_rows"] = 4
    frame.attrs["trajectory_fix_rows"] = 4
    frame.attrs["non_fix_rows"] = 0

    selected = _select_cycle_representative_fixes(frame, config)

    assert selected["source_index"].tolist() == [1, 3]
    assert selected.attrs["trajectory_fix_rows"] == 2
    assert selected.attrs["cycle_representative_dropped_points"] == 2


def test_jump_qc_drops_isolated_spike_when_bridge_is_plausible() -> None:
    segments, dropped, events, summary = apply_jump_qc_segments(
        [_jump_segment([0.0, 100.0, 0.1])],
        _jump_config(),
    )

    assert dropped["source_index"].tolist() == [1]
    assert [segment["source_index"].tolist() for segment in segments] == [[0, 2]]
    assert summary["jump_qc_dropped_points"] == 1
    assert summary["jump_qc_dropped_blocks"] == 1
    assert events[0]["reason"] == "isolated_spike"


def test_jump_qc_drops_short_bad_location_block() -> None:
    segments, dropped, events, summary = apply_jump_qc_segments(
        [_jump_segment([0.0, 100.0, 101.0, 0.1])],
        _jump_config(),
    )

    assert dropped["source_index"].tolist() == [1, 2]
    assert [segment["source_index"].tolist() for segment in segments] == [[0, 3]]
    assert summary["jump_qc_dropped_points"] == 2
    assert events[0]["reason"] == "short_bad_location_block"


def test_jump_qc_splits_remaining_jumps() -> None:
    segments, dropped, events, summary = apply_jump_qc_segments(
        [_jump_segment([0.0, 100.0])],
        _jump_config(),
    )

    assert dropped.empty
    assert [segment["source_index"].tolist() for segment in segments] == [[0], [1]]
    assert summary["jump_qc_remaining_jumps"] == 1
    assert events[0]["reason"] == "remaining_jump_split"


def test_jump_qc_post_merge_uses_physics_not_gap_points() -> None:
    first = _jump_segment([0.0, 0.1])
    second = _jump_segment([0.2, 0.3])
    second["source_index"] = [100, 101]
    second["_qc_order"] = [100, 101]
    second["time"] = pd.date_range("2000-01-03", periods=2, freq="D")

    segments, dropped, events, summary = apply_jump_qc_segments(
        [first, second],
        _jump_config(),
    )

    assert dropped.empty
    assert [segment["source_index"].tolist() for segment in segments] == [[0, 1, 100, 101]]
    assert summary["jump_qc_post_merge_count"] == 1
    assert summary["jump_qc_post_merge_rejection_count"] == 0
    assert any(event.get("event_type") == "post_jump_merge" and event.get("merged") for event in events)


def test_region_selection_trims_from_first_entry() -> None:
    config = replace(
        _config(),
        min_segment_points=2,
        region_selection=RegionSelectionConfig(
            names_or_labels=("SO",),
            selection_mode="from_first_entry",
            input_lon_mode="-180_180",
        ),
    )
    segment = pd.DataFrame(
        {
            "source_index": [0, 1, 2],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
            "lon": [10.0, 10.0, 11.0],
            "lat": [-20.0, -45.0, -46.0],
            "depth": [1000.0, 1000.0, 1000.0],
        }
    )

    selected, summary = apply_region_selection_segments([segment], config)

    assert [item["source_index"].tolist() for item in selected] == [[1, 2]]
    assert summary["region_selection_enabled"] is True
    assert summary["region_input_segments"] == 1
    assert summary["region_output_segments"] == 1
    assert summary["region_trimmed_segments"] == 1
    assert summary["region_input_points"] == 3
    assert summary["region_output_points"] == 2


def test_resampling_filters_duration_and_uses_shared_time_grid() -> None:
    config = replace(
        _config(),
        resample=ResampleConfig(
            frequency="10d",
            interpolate="time",
            reference_time=pd.Timestamp("2000-01-01T00:00:00"),
            shared_time=True,
            shift_start_to_reference=False,
            min_duration_days=21.0,
        ),
    )
    kept = pd.DataFrame(
        {
            "source_index": [0, 1],
            "time": pd.to_datetime(["2000-01-05", "2000-02-05"]),
            "lon": [0.0, 10.0],
            "lat": [-45.0, -46.0],
            "depth": [1000.0, 1000.0],
            "platform_code": [1902267, 1902267],
        }
    )
    short = pd.DataFrame(
        {
            "source_index": [2, 3],
            "time": pd.to_datetime(["2000-01-01", "2000-01-05"]),
            "lon": [0.0, 1.0],
            "lat": [-45.0, -45.5],
            "depth": [1000.0, 1000.0],
            "platform_code": [1902267, 1902267],
        }
    )

    output, summary = apply_resampling_segments([kept, short], config)

    assert len(output) == 1
    assert summary["resample_duration_dropped_segments"] == 1
    assert summary["resample_output_segments"] == 1
    assert output[0]["time"].tolist() == pd.to_datetime(["2000-01-11", "2000-01-21", "2000-01-31"]).tolist()
    assert output[0]["lon"].between(0.0, 10.0).all()


def test_representative_pressure_maps_to_measurement_rows() -> None:
    ds = xr.Dataset(
        data_vars={
            "CYCLE_NUMBER": (("N_MEASUREMENT",), np.asarray([1.0, 2.0, 2.0, 3.0])),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), np.asarray([1.0, 2.0, 3.0])),
            "REPRESENTATIVE_PARK_PRESSURE": (("N_CYCLE",), np.asarray([900.0, 1000.0, 1100.0])),
        }
    )

    depth = _map_representative_pressure_to_observations(ds, "REPRESENTATIVE_PARK_PRESSURE")

    np.testing.assert_allclose(depth, [900.0, 1000.0, 1000.0, 1100.0])


def test_missing_representative_pressure_returns_nan_depth() -> None:
    ds = xr.Dataset(
        data_vars={
            "CYCLE_NUMBER": (("N_MEASUREMENT",), np.asarray([1.0, 2.0, 3.0])),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), np.asarray([1.0, 2.0, 3.0])),
        }
    )

    depth = _map_representative_pressure_to_observations(ds, "REPRESENTATIVE_PARK_PRESSURE")

    assert len(depth) == 3
    assert np.isnan(depth).all()


def test_missing_representative_pressure_uses_configured_fallback() -> None:
    ds = xr.Dataset(
        data_vars={
            "CYCLE_NUMBER": (("N_MEASUREMENT",), np.asarray([1.0, 2.0, 3.0])),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), np.asarray([1.0, 2.0, 3.0])),
        }
    )

    depth = _map_representative_pressure_to_observations(
        ds,
        "REPRESENTATIVE_PARK_PRESSURE",
        fallback_value=1000.0,
    )

    np.testing.assert_allclose(depth, [1000.0, 1000.0, 1000.0])


def test_parking_depth_infers_from_park_window_pres_adjusted_p95() -> None:
    ds = xr.Dataset(
        data_vars={
            "CYCLE_NUMBER": (("N_MEASUREMENT",), np.asarray([1.0, 1.0, 1.0, 2.0, 2.0])),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), np.asarray([1.0, 2.0])),
            "REPRESENTATIVE_PARK_PRESSURE": (("N_CYCLE",), np.asarray([np.nan, np.nan])),
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0, 1.0, 2.0, 10.0, 11.0])),
            "JULD_PARK_START": (("N_CYCLE",), np.asarray([0.5, 9.5])),
            "JULD_PARK_END": (("N_CYCLE",), np.asarray([2.5, 11.5])),
            "PRES_ADJUSTED": (("N_MEASUREMENT",), np.asarray([10.0, 1000.0, 1100.0, 4000.0, 4200.0])),
        }
    )
    for name in ["JULD", "JULD_PARK_START", "JULD_PARK_END"]:
        ds[name].attrs["units"] = "days since 2000-01-01 00:00:00"
        ds[name].attrs["calendar"] = "gregorian"

    depth, source = _resolve_parking_depth_to_observations(
        ds,
        ParkingDepthConfig(
            mode="representative_park_pressure",
            fallback_value=1000.0,
            fill_missing=True,
            infer_from_park_window=True,
            pressure_variable="PRES_ADJUSTED",
            fallback_pressure_variable=None,
            percentile=95.0,
            min_pressure=50.0,
        ),
        pressure_name="REPRESENTATIVE_PARK_PRESSURE",
    )

    np.testing.assert_allclose(depth[:3], [1095.0, 1095.0, 1095.0])
    np.testing.assert_allclose(depth[3:], [4190.0, 4190.0])
    assert set(source) == {"park_window_pres_adjusted_p95"}


def test_parking_depth_backfills_initial_missing_cycle_before_fallback() -> None:
    ds = xr.Dataset(
        data_vars={
            "CYCLE_NUMBER": (("N_MEASUREMENT",), np.asarray([0.0, 1.0, 1.0])),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), np.asarray([0.0, 1.0])),
            "REPRESENTATIVE_PARK_PRESSURE": (("N_CYCLE",), np.asarray([np.nan, np.nan])),
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0, 1.0, 2.0])),
            "JULD_PARK_START": (("N_CYCLE",), np.asarray([np.nan, 0.5])),
            "JULD_PARK_END": (("N_CYCLE",), np.asarray([np.nan, 2.5])),
            "PRES_ADJUSTED": (("N_MEASUREMENT",), np.asarray([np.nan, 980.0, 1000.0])),
        }
    )

    depth, source = _resolve_parking_depth_to_observations(
        ds,
        ParkingDepthConfig(
            mode="representative_park_pressure",
            fallback_value=1.0,
            fill_missing=True,
            infer_from_park_window=True,
            pressure_variable="PRES_ADJUSTED",
            fallback_pressure_variable=None,
            percentile=50.0,
            min_pressure=50.0,
        ),
        pressure_name="REPRESENTATIVE_PARK_PRESSURE",
    )

    np.testing.assert_allclose(depth, [990.0, 990.0, 990.0])
    assert source.tolist() == [
        "depth_bfill",
        "park_window_pres_adjusted_p50",
        "park_window_pres_adjusted_p50",
    ]


def test_nested_reason_counts_are_aggregated() -> None:
    counts = _count_nested_reasons(
        [
            {"merge_rejection_counts": {"gap_points_too_large": 2}},
            {"merge_rejection_counts": {"gap_points_too_large": 1, "bridge_speed_too_large": 3}},
        ],
        "merge_rejection_counts",
    )

    assert counts == {
        "bridge_speed_too_large": 3,
        "gap_points_too_large": 3,
    }


def test_depth_bins_split_segments_and_fill_short_missing_runs() -> None:
    segment = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3, 4],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"]),
            "lon": [-70.0, -69.9, -69.8, -69.7, -69.6],
            "lat": [-60.0, -60.1, -60.2, -60.3, -60.4],
            "depth": [1000.0, np.nan, 1005.0, 2200.0, 2210.0],
        }
    )

    controlled, summary = apply_depth_bin_segmentation([segment], _depth_config())

    assert [item["depth_bin"].iloc[0] for item in controlled] == ["z0900_1100", "z2100_inf"]
    assert [len(item) for item in controlled] == [3, 2]
    assert summary["depth_bin_fill_count"] == 1
    assert summary["depth_bin_transition_count"] == 1
    assert summary["depth_bin_counts"] == {"z0900_1100": 3, "z2100_inf": 2}


def test_depth_bins_repair_isolated_finite_outlier_between_same_bin_neighbors() -> None:
    base = _config()
    config = replace(
        base,
        depth_bins=DepthBinConfig(
            enabled=True,
            output_mode="per_bin",
            bins=(
                DepthBin(label="z0002_0900", min_value=2.0, max_value=900.0),
                DepthBin(label="z0900_1100", min_value=900.0, max_value=1100.0),
            ),
            missing_depth=MissingDepthConfig(
                strategy="bounded_neighbor",
                max_fill_points=1,
                fill_between_same_bin_only=True,
            ),
            isolated_outlier=IsolatedDepthBinConfig(
                enabled=True,
                max_run_points=1,
                require_same_neighbor_bin=True,
            ),
        ),
    )
    segment = pd.DataFrame(
        {
            "source_index": [0, 1, 2],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
            "lon": [-70.0, -69.9, -69.8],
            "lat": [-60.0, -60.1, -60.2],
            "depth": [1000.0, 5.0, 1005.0],
        }
    )

    controlled, summary = apply_depth_bin_segmentation([segment], config)

    assert len(controlled) == 1
    assert controlled[0]["source_index"].tolist() == [0, 1, 2]
    assert controlled[0]["depth_bin"].tolist() == ["z0900_1100", "z0900_1100", "z0900_1100"]
    assert controlled[0]["depth_bin_repaired"].tolist() == [False, True, False]
    assert summary["depth_bin_repair_count"] == 1
    assert summary["depth_bin_transition_count"] == 0
    assert summary["depth_bin_counts"] == {"z0900_1100": 3}


def test_prepare_output_trajectories_renames_depth_to_z_and_drops_internal_columns() -> None:
    segment = pd.DataFrame(
        {
            "source_file": ["example_Rtraj.nc", "example_Rtraj.nc"],
            "platform_code": [1902267, 1902267],
            "source_index": [4, 5],
            "_qc_order": [0, 1],
            "time": pd.to_datetime(["2000-01-01", "2000-01-11"]),
            "lon": [-70.0, -69.0],
            "lat": [-60.0, -59.0],
            "depth": [1000.0, 1005.0],
            "depth_bin": ["z0900_1100", "z0900_1100"],
            "depth_bin_interval": ["[900, 1100)", "[900, 1100)"],
            "position_qc": ["1", "1"],
            "time_qc": ["1", "1"],
            "qc_keep": [True, True],
            "depth_source": ["park_window_pres_adjusted_p95", "park_window_pres_adjusted_p95"],
        }
    )

    prepared = prepare_output_trajectories([segment], _depth_config())

    assert len(prepared) == 1
    assert "z" in prepared[0].columns
    assert "depth" not in prepared[0].columns
    assert "source_index" not in prepared[0].columns
    assert "position_qc" not in prepared[0].columns
    assert prepared[0]["z"].tolist() == [1000.0, 1005.0]
    assert prepared[0]["depth_bin"].tolist() == ["z0900_1100", "z0900_1100"]
