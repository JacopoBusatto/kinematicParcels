from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kinematicparcels.tools.rtraj_to_zarr import (
    DepthBin,
    DepthBinConfig,
    IsolatedDepthBinConfig,
    JumpQcConfig,
    MergeConfig,
    MissingDepthConfig,
    ObservationConfig,
    ObservationSourceConfig,
    OutputConfig,
    ParkingDepthConfig,
    RegionSelectionConfig,
    ResampleConfig,
    RtrajConfig,
    TrajectoryFixConfig,
    _adjusted_time_with_raw_fallback,
    _build_observation_table,
    _count_nested_reasons,
    _filter_trajectory_fixes,
    _map_representative_pressure_to_observations,
    _resolve_parking_depth_to_observations,
    _select_cycle_representative_fixes,
    apply_depth_bin_segmentation,
    apply_jump_qc_segments,
    apply_region_selection_segments,
    apply_resampling_segments,
    prepare_output_trajectories,
    process_qc_stage,
    resolve_config,
    sample_observations_onto_segments,
    write_output_zarr,
)


def _config() -> RtrajConfig:
    return RtrajConfig(
        path=Path("rtraj_to_zarr.yml"),
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
        observations=ObservationConfig(
            enabled=False,
            time=ObservationSourceConfig(adjusted="JULD_ADJUSTED", fallback="JULD"),
            pressure=ObservationSourceConfig(adjusted="PRES_ADJUSTED", fallback="PRES"),
            variables={},
        ),
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


def _depth_config() -> RtrajConfig:
    base = _config()
    return RtrajConfig(
        path=base.path,
        raw=base.raw,
        mode=base.mode,
        max_files=base.max_files,
        input_files=base.input_files,
        source_variables=base.source_variables,
        normalized_variables=base.normalized_variables,
        observations=base.observations,
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


def _jump_config(*, min_segment_points: int = 1) -> RtrajConfig:
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


def _sampling_config(
    *,
    variable_names: tuple[str, ...] = ("temp", "psal"),
    sample_at_fallback_depth: bool = False,
) -> RtrajConfig:
    base = _config()
    return replace(
        base,
        observations=ObservationConfig(
            enabled=True,
            time=ObservationSourceConfig(adjusted="JULD_ADJUSTED", fallback="JULD"),
            pressure=ObservationSourceConfig(adjusted="PRES_ADJUSTED", fallback="PRES"),
            variables={
                name: ObservationSourceConfig(adjusted=None, fallback=name.upper())
                for name in variable_names
            },
            sample_at_fallback_depth=sample_at_fallback_depth,
        ),
        depth_bins=DepthBinConfig(
            enabled=True,
            output_mode="all",
            bins=(
                DepthBin(label="z0000_0001", min_value=0.0, max_value=1.0),
                DepthBin(label="z0900_1100", min_value=900.0, max_value=1100.0),
                DepthBin(label="z1150_1900", min_value=1150.0, max_value=1900.0),
                DepthBin(label="z1900_inf", min_value=1900.0, max_value=None),
            ),
            missing_depth=base.depth_bins.missing_depth,
            isolated_outlier=base.depth_bins.isolated_outlier,
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


def test_observation_table_resolves_adjusted_raw_fallback_and_metadata() -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0, 1.0, 2.0])),
            "JULD_ADJUSTED": (("N_MEASUREMENT",), np.asarray([0.25, np.nan, 2.25])),
            "PRES": (("N_MEASUREMENT",), np.asarray([100.0, 200.0, 300.0])),
            "PRES_ADJUSTED": (("N_MEASUREMENT",), np.asarray([101.0, np.nan, np.inf])),
            "TEMP": (("N_MEASUREMENT",), np.asarray([10.0, 11.0, 12.0])),
            "TEMP_ADJUSTED": (("N_MEASUREMENT",), np.asarray([10.5, np.nan, 12.5])),
            "PSAL": (("N_MEASUREMENT",), np.asarray([34.0, 34.1, 34.2])),
            "DOXY": (("N_MEASUREMENT",), np.asarray([210.0, 211.0, 212.0])),
        }
    )
    ds["TEMP_ADJUSTED"].attrs.update(
        units="degree_Celsius",
        long_name="Sea temperature in-situ ITS-90 scale",
        standard_name="sea_water_temperature",
    )
    observation_config = ObservationConfig(
        enabled=True,
        time=ObservationSourceConfig(adjusted="JULD_ADJUSTED", fallback="JULD"),
        pressure=ObservationSourceConfig(adjusted="PRES_ADJUSTED", fallback="PRES"),
        variables={
            "temp": ObservationSourceConfig(adjusted="TEMP_ADJUSTED", fallback="TEMP"),
            "psal": ObservationSourceConfig(adjusted="PSAL_ADJUSTED", fallback="PSAL"),
            "doxy": ObservationSourceConfig(adjusted=None, fallback="DOXY"),
            "missing": ObservationSourceConfig(adjusted="MISSING_ADJUSTED", fallback="MISSING"),
        },
    )

    observations, attrs, filter_counts = _build_observation_table(
        ds,
        observation_config,
        length=3,
    )

    expected_time = pd.to_datetime(
        ["1950-01-01 06:00:00", "1950-01-02 00:00:00", "1950-01-03 06:00:00"]
    )
    assert observations["_observation_time"].tolist() == expected_time.tolist()
    np.testing.assert_allclose(observations["_observation_pressure"], [101.0, 200.0, 300.0])
    np.testing.assert_allclose(observations["temp"], [10.5, 11.0, 12.5])
    np.testing.assert_allclose(observations["psal"], [34.0, 34.1, 34.2])
    np.testing.assert_allclose(observations["doxy"], [210.0, 211.0, 212.0])
    assert observations["missing"].isna().all()
    assert attrs["temp"] == {
        "units": "degree_Celsius",
        "long_name": "Sea temperature in-situ ITS-90 scale",
        "standard_name": "sea_water_temperature",
    }
    assert filter_counts == {
        "pressure.adjusted_accepted": 1,
        "pressure.raw_fallback_accepted": 2,
        "temp.adjusted_accepted": 2,
        "temp.raw_fallback_accepted": 1,
        "psal.raw_fallback_accepted": 3,
        "doxy.raw_fallback_accepted": 3,
        "missing.unavailable": 3,
    }


def test_observation_table_filters_chosen_source_qc_and_inclusive_bounds() -> None:
    length = 8
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.arange(length, dtype=float)),
            "PRES": (("N_MEASUREMENT",), np.full(length, 1000.0)),
            "TEMP_ADJUSTED": (
                ("N_MEASUREMENT",),
                np.asarray([5.0, np.nan, -3.0, 15.0, 16.0, np.nan, np.nan, np.nan]),
            ),
            "TEMP_ADJUSTED_QC": (
                ("N_MEASUREMENT",),
                np.asarray(["4", "", "0", "2", "1", "", "", ""]),
            ),
            "TEMP": (
                ("N_MEASUREMENT",),
                np.asarray([10.0, 6.0, 7.0, 8.0, 9.0, -4.0, 5.0, 6.0]),
            ),
            "TEMP_QC": (
                ("N_MEASUREMENT",),
                np.asarray(["1", "1", "1", "1", "1", "1", "", "0"]),
            ),
            "PSAL": (("N_MEASUREMENT",), np.full(length, 34.5)),
            "PSAL_QC": (("N_MEASUREMENT",), np.full(length, "1")),
        }
    )
    filtered = ObservationSourceConfig(
        adjusted="TEMP_ADJUSTED",
        fallback="TEMP",
        adjusted_qc="TEMP_ADJUSTED_QC",
        fallback_qc="TEMP_QC",
        valid_qc=("0", "1", "2"),
        missing_qc="reject",
        valid_min=-3.0,
        valid_max=15.0,
    )
    config = ObservationConfig(
        enabled=True,
        time=ObservationSourceConfig(adjusted=None, fallback="JULD"),
        pressure=ObservationSourceConfig(adjusted=None, fallback="PRES"),
        variables={
            "temp": filtered,
            "psal": ObservationSourceConfig(
                adjusted=None,
                fallback="PSAL",
                fallback_qc="PSAL_QC",
                valid_qc=("0", "1", "2"),
                missing_qc="reject",
                valid_min=30.0,
                valid_max=40.0,
            ),
        },
    )

    observations, _, counts = _build_observation_table(ds, config, length=length)

    expected_temp = [np.nan, 6.0, -3.0, 15.0, np.nan, np.nan, np.nan, 6.0]
    np.testing.assert_allclose(observations["temp"], expected_temp, equal_nan=True)
    np.testing.assert_allclose(observations["psal"], np.full(length, 34.5))
    assert counts["temp.adjusted_qc_4_rejected"] == 1
    assert counts["temp.adjusted_accepted"] == 2
    assert counts["temp.raw_fallback_accepted"] == 2
    assert counts["temp.adjusted_above_valid_max_rejected"] == 1
    assert counts["temp.raw_fallback_below_valid_min_rejected"] == 1
    assert counts["temp.raw_fallback_missing_qc_rejected"] == 1
    assert counts["psal.raw_fallback_accepted"] == length


def test_observation_pressure_qc_excludes_row_from_all_sampling_candidates() -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0])),
            "PRES": (("N_MEASUREMENT",), np.asarray([1000.0])),
            "PRES_QC": (("N_MEASUREMENT",), np.asarray(["4"])),
            "TEMP": (("N_MEASUREMENT",), np.asarray([5.0])),
            "TEMP_QC": (("N_MEASUREMENT",), np.asarray(["1"])),
            "PSAL": (("N_MEASUREMENT",), np.asarray([34.5])),
            "PSAL_QC": (("N_MEASUREMENT",), np.asarray(["1"])),
        }
    )
    source = lambda fallback, fallback_qc: ObservationSourceConfig(
        adjusted=None,
        fallback=fallback,
        fallback_qc=fallback_qc,
        valid_qc=("0", "1", "2"),
        missing_qc="reject",
    )
    observation_config = ObservationConfig(
        enabled=True,
        time=ObservationSourceConfig(adjusted=None, fallback="JULD"),
        pressure=source("PRES", "PRES_QC"),
        variables={
            "temp": source("TEMP", "TEMP_QC"),
            "psal": source("PSAL", "PSAL_QC"),
        },
    )
    observations, _, counts = _build_observation_table(
        ds,
        observation_config,
        length=1,
    )
    config = replace(_sampling_config(), observations=observation_config)
    segment = pd.DataFrame(
        {
            "time": pd.to_datetime(["1950-01-01"]),
            "lon": [0.0],
            "lat": [-50.0],
            "depth": [1000.0],
            "depth_source": ["representative_park_pressure"],
            "depth_bin": ["z0900_1100"],
            "depth_bin_interval": ["[900, 1100)"],
        }
    )

    sampled = sample_observations_onto_segments([segment], observations, config)

    assert np.isnan(observations["_observation_pressure"].iloc[0])
    assert observations[["temp", "psal"]].notna().all().all()
    assert sampled.segments[0][["temp", "psal"]].isna().all().all()
    assert sampled.summary["observation_unmatched_counts"] == {"temp": 1, "psal": 1}
    assert counts["pressure.raw_fallback_qc_4_rejected"] == 1


def test_observation_missing_qc_accepts_absent_qc_source_when_configured() -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0])),
            "PRES": (("N_MEASUREMENT",), np.asarray([1000.0])),
            "TEMP": (("N_MEASUREMENT",), np.asarray([5.0])),
        }
    )
    config = ObservationConfig(
        enabled=True,
        time=ObservationSourceConfig(adjusted=None, fallback="JULD"),
        pressure=ObservationSourceConfig(adjusted=None, fallback="PRES"),
        variables={
            "temp": ObservationSourceConfig(
                adjusted=None,
                fallback="TEMP",
                fallback_qc="MISSING_TEMP_QC",
                valid_qc=("0", "1", "2"),
                missing_qc="accept",
                valid_min=-3.0,
                valid_max=15.0,
            )
        },
    )

    observations, _, counts = _build_observation_table(ds, config, length=1)

    assert observations["temp"].iloc[0] == 5.0
    assert counts["temp.raw_fallback_accepted"] == 1
    assert not any("missing_qc_rejected" in reason for reason in counts)


def test_observation_qc_accepts_only_configured_flags() -> None:
    flags = np.asarray(["0", "1", "2", "3", "4", "5", "8", "9"])
    length = len(flags)
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.arange(length, dtype=float)),
            "PRES": (("N_MEASUREMENT",), np.full(length, 1000.0)),
            "TEMP": (("N_MEASUREMENT",), np.arange(length, dtype=float)),
            "TEMP_QC": (("N_MEASUREMENT",), flags),
        }
    )
    config = ObservationConfig(
        enabled=True,
        time=ObservationSourceConfig(adjusted=None, fallback="JULD"),
        pressure=ObservationSourceConfig(adjusted=None, fallback="PRES"),
        variables={
            "temp": ObservationSourceConfig(
                adjusted=None,
                fallback="TEMP",
                fallback_qc="TEMP_QC",
                valid_qc=("0", "1", "2"),
                missing_qc="reject",
            )
        },
    )

    observations, _, counts = _build_observation_table(ds, config, length=length)

    np.testing.assert_allclose(
        observations["temp"],
        [0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    assert counts["temp.raw_fallback_accepted"] == 3
    for flag in ("3", "4", "5", "8", "9"):
        assert counts[f"temp.raw_fallback_qc_{flag}_rejected"] == 1


def test_filtered_observation_roundtrip_keeps_variables_independent(
    tmp_path: Path,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0])),
            "PRES": (("N_MEASUREMENT",), np.asarray([1000.0])),
            "PRES_QC": (("N_MEASUREMENT",), np.asarray(["1"])),
            "TEMP": (("N_MEASUREMENT",), np.asarray([342.0])),
            "TEMP_QC": (("N_MEASUREMENT",), np.asarray(["0"])),
            "PSAL": (("N_MEASUREMENT",), np.asarray([34.5])),
            "PSAL_QC": (("N_MEASUREMENT",), np.asarray(["1"])),
        }
    )
    pressure = ObservationSourceConfig(
        adjusted=None,
        fallback="PRES",
        fallback_qc="PRES_QC",
        valid_qc=("0", "1", "2"),
        missing_qc="reject",
    )
    variables = {
        "temp": ObservationSourceConfig(
            adjusted=None,
            fallback="TEMP",
            fallback_qc="TEMP_QC",
            valid_qc=("0", "1", "2"),
            missing_qc="reject",
            valid_min=-3.0,
            valid_max=15.0,
        ),
        "psal": ObservationSourceConfig(
            adjusted=None,
            fallback="PSAL",
            fallback_qc="PSAL_QC",
            valid_qc=("0", "1", "2"),
            missing_qc="reject",
            valid_min=30.0,
            valid_max=40.0,
        ),
    }
    observation_config = ObservationConfig(
        enabled=True,
        time=ObservationSourceConfig(adjusted=None, fallback="JULD"),
        pressure=pressure,
        variables=variables,
    )
    observations, _, counts = _build_observation_table(
        ds,
        observation_config,
        length=1,
    )
    config = replace(
        _sampling_config(),
        observations=observation_config,
        output=OutputConfig(
            zarr_path=tmp_path / "filtered.zarr",
            write_zarr=True,
            overwrite=False,
        ),
    )
    segment = pd.DataFrame(
        {
            "platform_code": [1900000],
            "time": pd.to_datetime(["1950-01-01"]),
            "lon": [0.0],
            "lat": [-50.0],
            "depth": [1000.0],
            "depth_source": ["representative_park_pressure"],
            "depth_bin": ["z0900_1100"],
            "depth_bin_interval": ["[900, 1100)"],
        }
    )

    sampled = sample_observations_onto_segments([segment], observations, config)
    write_output_zarr(sampled.segments, config)

    assert counts["temp.raw_fallback_above_valid_max_rejected"] == 1
    assert counts["psal.raw_fallback_accepted"] == 1
    with xr.open_zarr(config.output.zarr_path) as reopened:
        assert np.isnan(reopened["temp"].values).all()
        assert float(reopened["psal"].values[0, 0]) == 34.5
        assert float(reopened["z"].values[0, 0]) == 1000.0


def test_observation_sampling_uses_depth_time_pressure_index_and_current_depth() -> None:
    config = _sampling_config(variable_names=("temp", "psal"))
    segment = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2000-01-10", "2000-01-20", "2000-01-30", "2000-02-10", "2000-02-10", "2000-03-01"]
            ),
            "lon": [0.0] * 6,
            "lat": [-50.0] * 6,
            "depth": [1000.0, 1000.0, 1000.0, 1000.0, 2000.0, 1500.0],
            "depth_source": ["representative_park_pressure"] * 6,
            "depth_bin": [
                "z0900_1100",
                "z0900_1100",
                "z0900_1100",
                "z0900_1100",
                "z1900_inf",
                "z1150_1900",
            ],
            "depth_bin_interval": [
                "[900, 1100)",
                "[900, 1100)",
                "[900, 1100)",
                "[900, 1100)",
                "[1900, +inf)",
                "[1150, 1900)",
            ],
        }
    )
    unknown = pd.DataFrame(
        {
            "time": pd.to_datetime(["2000-03-10"]),
            "lon": [0.0],
            "lat": [-50.0],
            "depth": [0.0],
            "depth_source": ["fallback"],
            "depth_bin": ["z0000_0001"],
            "depth_bin_interval": ["[0, 1)"],
        }
    )
    observations = pd.DataFrame(
        {
            "_observation_index": [10, 11, 12, 13, 20, 19, 15, 16, 17],
            "_observation_time": pd.to_datetime(
                [
                    "2000-01-09",
                    "2000-01-10",
                    "2000-01-19",
                    "2000-01-21",
                    "2000-01-29",
                    "2000-01-31",
                    "2000-02-10",
                    "2000-02-11",
                    "2000-03-10",
                ]
            ),
            "_observation_pressure": [1000.0, 900.0, 950.0, 990.0, 1000.0, 1000.0, 2000.0, 1000.0, 0.0],
            "temp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 7.0, 88.0],
            "psal": [31.0, np.nan, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0],
        }
    )

    sampled = sample_observations_onto_segments([segment, unknown], observations, config)

    assert sampled.segments[0]["temp"].tolist()[:5] == [2.0, 4.0, 6.0, 7.0, 99.0]
    assert np.isnan(sampled.segments[0]["temp"].iloc[5])
    assert sampled.segments[0]["psal"].iloc[0] == 31.0
    assert sampled.segments[0]["psal"].iloc[4] == 36.0
    assert sampled.segments[1][["temp", "psal"]].isna().all().all()
    assert sampled.summary["observation_eligible_points"] == 6
    assert sampled.summary["observation_unknown_depth_skipped_points"] == 1
    assert sampled.summary["observation_matched_counts"] == {"temp": 5, "psal": 5}
    assert sampled.summary["observation_unmatched_counts"] == {"temp": 1, "psal": 1}


def test_observation_sampling_can_match_fallback_depth_points() -> None:
    disabled = _sampling_config(variable_names=("temp",))
    enabled = _sampling_config(
        variable_names=("temp",),
        sample_at_fallback_depth=True,
    )
    fallback = pd.DataFrame(
        {
            "time": pd.to_datetime(["2000-01-10"]),
            "lon": [0.0],
            "lat": [-50.0],
            "depth": [1000.0],
            "depth_source": ["fallback"],
            "depth_bin": ["z0900_1100"],
            "depth_bin_interval": ["[900, 1100)"],
        }
    )
    known = fallback.assign(depth_source="representative_park_pressure")
    observations = pd.DataFrame(
        {
            "_observation_index": [11, 10],
            "_observation_time": pd.to_datetime(["2000-01-09", "2000-01-11"]),
            "_observation_pressure": [950.0, 1000.0],
            "temp": [4.0, 5.0],
        }
    )

    skipped = sample_observations_onto_segments([fallback], observations, disabled)
    sampled = sample_observations_onto_segments([fallback], observations, enabled)
    known_disabled = sample_observations_onto_segments([known], observations, disabled)
    known_enabled = sample_observations_onto_segments([known], observations, enabled)

    assert np.isnan(skipped.segments[0]["temp"].iloc[0])
    assert skipped.summary["observation_eligible_points"] == 0
    assert skipped.summary["observation_unknown_depth_skipped_points"] == 1
    assert sampled.segments[0]["temp"].iloc[0] == 5.0
    assert sampled.summary["observation_eligible_points"] == 1
    assert sampled.summary["observation_unknown_depth_skipped_points"] == 0
    assert sampled.summary["observation_matched_counts"] == {"temp": 1}
    assert sampled.summary["observation_unmatched_counts"] == {"temp": 0}
    assert sampled.summary["observation_median_abs_time_mismatch_days"] == {"temp": 1.0}
    assert sampled.summary["observation_median_abs_pressure_mismatch_dbar"] == {"temp": 0.0}
    pd.testing.assert_frame_equal(known_enabled.segments[0], known_disabled.segments[0])


def test_observation_sampling_fallback_depth_without_candidate_is_unmatched() -> None:
    config = _sampling_config(
        variable_names=("temp",),
        sample_at_fallback_depth=True,
    )
    fallback = pd.DataFrame(
        {
            "time": pd.to_datetime(["2000-01-10"]),
            "lon": [0.0],
            "lat": [-50.0],
            "depth": [1000.0],
            "depth_source": ["fallback"],
            "depth_bin": ["z0900_1100"],
            "depth_bin_interval": ["[900, 1100)"],
        }
    )
    observations = pd.DataFrame(
        {
            "_observation_index": [10],
            "_observation_time": pd.to_datetime(["2000-01-10"]),
            "_observation_pressure": [2000.0],
            "temp": [4.0],
        }
    )

    sampled = sample_observations_onto_segments([fallback], observations, config)

    assert np.isnan(sampled.segments[0]["temp"].iloc[0])
    assert sampled.summary["observation_eligible_points"] == 1
    assert sampled.summary["observation_unknown_depth_skipped_points"] == 0
    assert sampled.summary["observation_matched_counts"] == {"temp": 0}
    assert sampled.summary["observation_unmatched_counts"] == {"temp": 1}


def test_yaml_adds_third_observation_variable_without_sampler_changes(tmp_path: Path) -> None:
    input_path = tmp_path / "1900000_Rtraj.nc"
    input_path.touch()
    config_path = tmp_path / "rtraj.yml"
    config_path.write_text(
        f"""
input:
  rtraj_files: ['{input_path.as_posix()}']
output:
  zarr_path: '{(tmp_path / 'output.zarr').as_posix()}'
diagnostics:
  output_dir: '{(tmp_path / 'diagnostics').as_posix()}'
depth_bins:
  enabled: true
  output_mode: all
  bins:
    - label: z0000_inf
      min: 0.0
      max: null
observations:
  enabled: true
  time:
    adjusted: JULD_ADJUSTED
    fallback: JULD
  pressure:
    adjusted: PRES_ADJUSTED
    adjusted_qc: PRES_ADJUSTED_QC
    fallback: PRES
    fallback_qc: PRES_QC
    valid_qc: ["0", "1", "2"]
    missing_qc: reject
  variables:
    temp:
      adjusted: TEMP_ADJUSTED
      adjusted_qc: TEMP_ADJUSTED_QC
      fallback: TEMP
      fallback_qc: TEMP_QC
      valid_qc: ["0", "1", "2"]
      missing_qc: reject
      valid_min: -3
      valid_max: 15
    psal:
      adjusted: PSAL_ADJUSTED
      fallback: PSAL
    doxy:
      adjusted: null
      fallback: DOXY
""",
        encoding="utf-8",
    )

    config = resolve_config(config_path)

    assert tuple(config.observations.variables) == ("temp", "psal", "doxy")
    assert config.observations.variables["doxy"].adjusted is None
    assert config.observations.variables["doxy"].fallback == "DOXY"
    assert config.observations.sample_at_fallback_depth is False
    assert config.observations.pressure.valid_qc == ("0", "1", "2")
    assert config.observations.pressure.missing_qc == "reject"
    assert config.observations.variables["temp"] == ObservationSourceConfig(
        adjusted="TEMP_ADJUSTED",
        fallback="TEMP",
        adjusted_qc="TEMP_ADJUSTED_QC",
        fallback_qc="TEMP_QC",
        valid_qc=("0", "1", "2"),
        missing_qc="reject",
        valid_min=-3.0,
        valid_max=15.0,
    )


@pytest.mark.parametrize(
    ("extra_yaml", "message"),
    [
        ("missing_qc: maybe", "missing_qc"),
        ("valid_qc: []", "valid_qc"),
        ("valid_min: 5\n      valid_max: 4", "valid_min"),
        ("valid_min: .inf", "valid_min"),
    ],
)
def test_yaml_rejects_invalid_observation_filters(
    tmp_path: Path,
    extra_yaml: str,
    message: str,
) -> None:
    input_path = tmp_path / "1900000_Rtraj.nc"
    input_path.touch()
    config_path = tmp_path / "rtraj.yml"
    config_path.write_text(
        f"""
input:
  rtraj_files: ['{input_path.as_posix()}']
output:
  zarr_path: '{(tmp_path / 'output.zarr').as_posix()}'
diagnostics:
  output_dir: '{(tmp_path / 'diagnostics').as_posix()}'
depth_bins:
  enabled: true
  output_mode: all
  bins:
    - label: z0000_inf
      min: 0.0
      max: null
observations:
  enabled: true
  variables:
    temp:
      adjusted: TEMP_ADJUSTED
      fallback: TEMP
      {extra_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        resolve_config(config_path)


def test_yaml_enables_sampling_at_fallback_depth(tmp_path: Path) -> None:
    input_path = tmp_path / "1900000_Rtraj.nc"
    input_path.touch()
    config_path = tmp_path / "rtraj.yml"
    config_path.write_text(
        f"""
input:
  rtraj_files: ['{input_path.as_posix()}']
output:
  zarr_path: '{(tmp_path / 'output.zarr').as_posix()}'
diagnostics:
  output_dir: '{(tmp_path / 'diagnostics').as_posix()}'
depth_bins:
  enabled: true
  output_mode: all
  bins:
    - label: z0000_inf
      min: 0.0
      max: null
observations:
  enabled: true
  sample_at_fallback_depth: true
  variables:
    temp:
      adjusted: TEMP_ADJUSTED
      fallback: TEMP
""",
        encoding="utf-8",
    )

    config = resolve_config(config_path)

    assert config.observations.sample_at_fallback_depth is True


def test_final_zarr_adds_only_configured_observations_and_preserves_existing_variables(tmp_path: Path) -> None:
    base_config = replace(
        _sampling_config(variable_names=("temp", "psal")),
        output=OutputConfig(zarr_path=tmp_path / "with_observations.zarr", write_zarr=True, overwrite=False),
    )
    disabled_config = replace(
        base_config,
        observations=replace(base_config.observations, enabled=False, variables={}),
        output=OutputConfig(zarr_path=tmp_path / "without_observations.zarr", write_zarr=True, overwrite=False),
    )
    existing = pd.DataFrame(
        {
            "platform_code": [1902267, 1902267],
            "time": pd.to_datetime(["2000-01-01", "2000-01-11"]),
            "lon": [-40.0, -39.0],
            "lat": [-55.0, -54.0],
            "depth": [1000.0, 1000.0],
            "depth_source": ["representative_park_pressure"] * 2,
            "depth_bin": ["z0900_1100"] * 2,
            "depth_bin_interval": ["[900, 1100)"] * 2,
        }
    )
    sampled = existing.assign(temp=[5.0, 6.0], psal=[34.0, 34.1])

    write_output_zarr([existing], disabled_config)
    write_output_zarr(
        [sampled],
        base_config,
        variable_attrs={
            "temp": {
                "units": "degree_Celsius",
                "long_name": "Sea temperature in-situ ITS-90 scale",
                "standard_name": "sea_water_temperature",
            },
            "psal": {"units": "psu", "long_name": "Practical salinity"},
        },
    )

    with xr.open_zarr(disabled_config.output.zarr_path) as baseline, xr.open_zarr(
        base_config.output.zarr_path
    ) as observed:
        for name in baseline.data_vars:
            xr.testing.assert_equal(observed[name], baseline[name])
        assert {"temp", "psal"}.issubset(observed.data_vars)
        assert "doxy" not in observed
        assert "_observation_time" not in observed
        assert "_observation_pressure" not in observed
        assert observed["temp"].attrs["units"] == "degree_Celsius"
        assert observed["temp"].attrs["standard_name"] == "sea_water_temperature"
