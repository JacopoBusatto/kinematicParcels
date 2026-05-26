from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kinematicparcels.postprocessing.config.models import DatasetCoordinatesConfig
from kinematicparcels.postprocessing.io.parcels import build_trajectory_table, open_parcels_dataset
from kinematicparcels.postprocessing.io.parcels import resolve_parcels_schema
from kinematicparcels.tools.argo_to_zarr import convert_argo_to_dataframe, convert_argo_to_zarr


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_convert_argo_to_zarr_creates_parcels_compatible_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-11-30T04:23:36Z",
                "LATITUDE (degree_north)": -45.966,
                "LONGITUDE (degree_east)": 51.947,
                "PRES_ADJUSTED (decibar)": 19.0,
                "TEMP_ADJUSTED (degree_Celsius)": 5.920,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-11-30T04:23:36Z",
                "LATITUDE (degree_north)": -45.966,
                "LONGITUDE (degree_east)": 51.947,
                "PRES_ADJUSTED (decibar)": 2.2,
                "TEMP_ADJUSTED (degree_Celsius)": 5.955,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-05T04:23:36Z",
                "LATITUDE (degree_north)": -45.500,
                "LONGITUDE (degree_east)": 52.100,
                "PRES_ADJUSTED (decibar)": 8.0,
                "TEMP_ADJUSTED (degree_Celsius)": 6.100,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-05T04:23:36Z",
                "LATITUDE (degree_north)": -45.500,
                "LONGITUDE (degree_east)": 52.100,
                "PRES_ADJUSTED (decibar)": 5.0,
                "TEMP_ADJUSTED (degree_Celsius)": 6.200,
            },
        ],
    )

    output_path = tmp_path / "argo_output.zarr"
    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(output_path)},
        "variables": {"optional": ["TEMP_ADJUSTED (degree_Celsius)"]},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
        },
    }

    convert_argo_to_zarr(config)

    ds = open_parcels_dataset(output_path)
    schema = resolve_parcels_schema(ds, coordinates=DatasetCoordinatesConfig())
    table = build_trajectory_table(
        ds,
        schema=schema,
        extra_vars=["platform_code", "TEMP_ADJUSTED (degree_Celsius)"],
    )

    assert ds.dims["trajectory"] == 1
    assert ds.dims["obs"] == 2
    assert list(ds.data_vars)[:4] == ["platform_code", "time", "lat", "lon"] or set(["time", "lat", "lon", "z"]).issubset(ds.data_vars)
    assert np.allclose(ds["z"].values[0, :2], [1000.0, 1000.0])
    assert table["trajectory"].nunique() == 1
    assert table["TEMP_ADJUSTED (degree_Celsius)"].tolist()[:2] == [5.955, 6.2]


def test_segment_mode_split_as_new_splits_large_time_gaps(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_split.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-11-30T00:00:00Z",
                "LATITUDE (degree_north)": 10.0,
                "LONGITUDE (degree_east)": 20.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-05T00:00:00Z",
                "LATITUDE (degree_north)": 10.5,
                "LONGITUDE (degree_east)": 20.5,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-20T00:00:00Z",
                "LATITUDE (degree_north)": 11.0,
                "LONGITUDE (degree_east)": 21.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "split.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "split_as_new", "max_gap_days": 10.0},
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 2
    assert trajectories[0]["trajectory"].iloc[0] == 0
    assert trajectories[1]["trajectory"].iloc[0] == 1
    assert trajectories[0]["platform_code"].iloc[0] == 1900042
    assert trajectories[1]["platform_code"].iloc[0] == 1900042
    assert len(trajectories[0]) == 2
    assert len(trajectories[1]) == 1


def test_overlapping_platform_data_across_files_is_merged_before_segmentation(tmp_path: Path) -> None:
    csv_path_1 = tmp_path / "argo_part1.csv"
    csv_path_2 = tmp_path / "argo_part2.csv"

    _write_csv(
        csv_path_1,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-11-30T00:00:00Z",
                "LATITUDE (degree_north)": 10.0,
                "LONGITUDE (degree_east)": 20.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-05T00:00:00Z",
                "LATITUDE (degree_north)": 10.5,
                "LONGITUDE (degree_east)": 20.5,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    _write_csv(
        csv_path_2,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-05T00:00:00Z",
                "LATITUDE (degree_north)": 10.5,
                "LONGITUDE (degree_east)": 20.5,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-20T00:00:00Z",
                "LATITUDE (degree_north)": 11.0,
                "LONGITUDE (degree_east)": 21.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path_1), str(csv_path_2)]},
        "output": {"path": str(tmp_path / "merged.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["platform_code"].iloc[0] == 1900042
    assert trajectory["trajectory"].iloc[0] == 0
    assert len(trajectory) == 3
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2002-11-30T00:00:00"),
        pd.Timestamp("2002-12-05T00:00:00"),
        pd.Timestamp("2002-12-20T00:00:00"),
    ]


def test_segment_mode_split_as_new_splits_large_jump_by_speed(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_jump.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-11-30T00:00:00Z",
                "LATITUDE (degree_north)": -42.0,
                "LONGITUDE (degree_east)": 179.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-10T00:00:00Z",
                "LATITUDE (degree_north)": -42.0,
                "LONGITUDE (degree_east)": 9.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-20T00:00:00Z",
                "LATITUDE (degree_north)": -42.2,
                "LONGITUDE (degree_east)": 11.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "jump_split.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {
                "mode": "split_as_new",
                "max_gap_days": 20.0,
                "max_speed_km_per_day": 500.0,
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 2
    assert len(trajectories[0]) == 1
    assert len(trajectories[1]) == 2
    assert trajectories[0]["trajectory"].iloc[0] == 0
    assert trajectories[1]["trajectory"].iloc[0] == 1


def test_region_cut_and_resample_keep_only_points_after_entry(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_regions.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": 30.0,
                "LONGITUDE (degree_east)": 0.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-16T00:00:00Z",
                "LATITUDE (degree_north)": 36.8,
                "LONGITUDE (degree_east)": 14.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T00:00:00Z",
                "LATITUDE (degree_north)": 37.0,
                "LONGITUDE (degree_east)": 14.4,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "regions.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "regions": {
                "names_or_labels": ["sic"],
                "cut_from_first_entry": True,
            },
            "resample": {
                "enabled": True,
                "frequency": "12H",
                "interpolate": "time",
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["time"].iloc[0] == pd.Timestamp("2026-04-16T00:00:00")
    assert trajectory["time"].iloc[-1] == pd.Timestamp("2026-04-17T00:00:00")
    assert len(trajectory) == 3
    assert np.isclose(trajectory["lon"].iloc[1], 14.2)
    assert np.isclose(trajectory["lat"].iloc[1], 36.9)


def test_resample_handles_duplicate_times_within_trajectory(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_duplicate_times.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": 36.8,
                "LONGITUDE (degree_east)": 14.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": 37.0,
                "LONGITUDE (degree_east)": 14.2,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T00:00:00Z",
                "LATITUDE (degree_north)": 37.2,
                "LONGITUDE (degree_east)": 14.4,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "duplicate_times.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": True,
                "frequency": "1d",
                "interpolate": "time",
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2026-04-15T00:00:00"),
        pd.Timestamp("2026-04-16T00:00:00"),
        pd.Timestamp("2026-04-17T00:00:00"),
    ]
    assert np.isclose(trajectory["lon"].iloc[0], 14.0)
    assert np.isclose(trajectory["lat"].iloc[0], 36.8)


def test_resample_interpolates_longitude_across_dateline_without_midpoint_jump(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_dateline.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 179.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": -179.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "dateline.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": True,
                "frequency": "1d",
                "interpolate": "time",
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert len(trajectory) == 3

    lon = trajectory["lon"].to_numpy(dtype=float)
    wrapped_steps = np.abs(((np.diff(lon) + 180.0) % 360.0) - 180.0)

    assert np.isclose(abs(lon[1]), 180.0)
    assert np.all(wrapped_steps <= 1.1)


def test_shared_time_with_shift_start_to_reference_uses_common_time_grid(tmp_path: Path) -> None:
    csv_path_1 = tmp_path / "argo_align_1.csv"
    csv_path_2 = tmp_path / "argo_align_2.csv"

    _write_csv(
        csv_path_1,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 10.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T00:00:00Z",
                "LATITUDE (degree_north)": -54.0,
                "LONGITUDE (degree_east)": 12.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    _write_csv(
        csv_path_2,
        [
            {
                "PLATFORM_CODE": 1901042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-05-01T00:00:00Z",
                "LATITUDE (degree_north)": -50.0,
                "LONGITUDE (degree_east)": 20.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1901042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-05-04T00:00:00Z",
                "LATITUDE (degree_north)": -49.0,
                "LONGITUDE (degree_east)": 21.5,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path_1), str(csv_path_2)]},
        "output": {"path": str(tmp_path / "aligned.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": True,
                "frequency": "1d",
                "interpolate": "time",
                "reference_time": "2020-01-01T00:00:00Z",
                "shared_time": True,
                "shift_start_to_reference": True,
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory["time"].iloc[0] == pd.Timestamp("2020-01-01T00:00:00")

    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00"),
        pd.Timestamp("2020-01-02T00:00:00"),
        pd.Timestamp("2020-01-03T00:00:00"),
        pd.Timestamp("2020-01-04T00:00:00"),
    ]
    assert trajectories[1]["time"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00"),
        pd.Timestamp("2020-01-02T00:00:00"),
        pd.Timestamp("2020-01-03T00:00:00"),
        pd.Timestamp("2020-01-04T00:00:00"),
    ]

    shared_prefix = min(len(trajectory) for trajectory in trajectories)
    for obs_index in range(shared_prefix):
        values = [trajectory["time"].iloc[obs_index] for trajectory in trajectories]
        assert len(set(values)) == 1

    assert np.isnan(trajectories[0]["lon"].iloc[-1])
    assert np.isnan(trajectories[0]["lat"].iloc[-1])


def test_shared_time_trims_leading_empty_prefix(tmp_path: Path) -> None:
    csv_path_1 = tmp_path / "argo_shared_trim_1.csv"
    csv_path_2 = tmp_path / "argo_shared_trim_2.csv"

    _write_csv(
        csv_path_1,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-11-30T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 10.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-10T00:00:00Z",
                "LATITUDE (degree_north)": -54.0,
                "LONGITUDE (degree_east)": 12.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    _write_csv(
        csv_path_2,
        [
            {
                "PLATFORM_CODE": 1901042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-02T00:00:00Z",
                "LATITUDE (degree_north)": -50.0,
                "LONGITUDE (degree_east)": 20.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1901042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2002-12-12T00:00:00Z",
                "LATITUDE (degree_north)": -49.0,
                "LONGITUDE (degree_east)": 21.5,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path_1), str(csv_path_2)]},
        "output": {"path": str(tmp_path / "shared_trim.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": True,
                "frequency": "10d",
                "interpolate": "time",
                "reference_time": "2000-01-01T00:00:00Z",
                "shared_time": True,
                "shift_start_to_reference": False,
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory["time"].iloc[0] == pd.Timestamp("2002-12-06T00:00:00")

    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2002-12-06T00:00:00"),
    ]
    assert trajectories[1]["time"].tolist() == [
        pd.Timestamp("2002-12-06T00:00:00"),
    ]


def test_shared_time_requires_reference_time_when_enabled(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_missing_reference_time.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 10.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "align_missing_start.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": True,
                "frequency": "1d",
                "shared_time": True,
            },
        },
    }

    try:
        convert_argo_to_dataframe(config)
    except ValueError as exc:
        assert "processing.resample.reference_time" in str(exc)
    else:
        raise AssertionError("Expected shared_time without reference_time to raise ValueError")


def test_shared_time_requires_frequency_when_enabled(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_shared_time_missing_frequency.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 10.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T00:00:00Z",
                "LATITUDE (degree_north)": -54.0,
                "LONGITUDE (degree_east)": 12.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "align_missing_frequency.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": False,
                "reference_time": "2020-01-01T00:00:00Z",
                "shared_time": True,
            },
        },
    }

    try:
        convert_argo_to_dataframe(config)
    except ValueError as exc:
        assert "processing.resample.frequency" in str(exc)
    else:
        raise AssertionError("Expected shared_time without frequency to raise ValueError")


def test_reference_time_anchors_non_shared_resampling_grid(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_reference_time_grid.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T12:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 10.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T12:00:00Z",
                "LATITUDE (degree_north)": -54.0,
                "LONGITUDE (degree_east)": 12.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "reference_time_grid.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": True,
                "frequency": "1d",
                "interpolate": "time",
                "reference_time": "2026-04-15T00:00:00Z",
                "shared_time": False,
                "shift_start_to_reference": False,
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2026-04-16T00:00:00"),
        pd.Timestamp("2026-04-17T00:00:00"),
    ]


def test_shift_start_to_reference_works_without_shared_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "argo_shift_only.csv"
    _write_csv(
        csv_path,
        [
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-15T00:00:00Z",
                "LATITUDE (degree_north)": -55.0,
                "LONGITUDE (degree_east)": 10.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
            {
                "PLATFORM_CODE": 1900042,
                "DATE (YYYY-MM-DDTHH:MI:SSZ)": "2026-04-17T00:00:00Z",
                "LATITUDE (degree_north)": -54.0,
                "LONGITUDE (degree_east)": 12.0,
                "PRES_ADJUSTED (decibar)": 2.0,
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "shift_only.zarr")},
        "processing": {
            "parking_depth": {"mode": "fixed", "value": 1000.0},
            "segment": {"mode": "ignore", "max_gap_days": 10.0},
            "resample": {
                "enabled": False,
                "reference_time": "2020-01-01T00:00:00Z",
                "shift_start_to_reference": True,
            },
        },
    }

    trajectories = convert_argo_to_dataframe(config)

    assert len(trajectories) == 1
    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00"),
        pd.Timestamp("2020-01-03T00:00:00"),
    ]