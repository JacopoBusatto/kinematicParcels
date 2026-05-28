from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from kinematicparcels.postprocessing.config.models import DatasetCoordinatesConfig
from kinematicparcels.postprocessing.io.parcels import build_trajectory_table, open_parcels_dataset
from kinematicparcels.postprocessing.io.parcels import resolve_parcels_schema
from kinematicparcels.tools.drifter_to_zarr import convert_drifter_to_dataframe, convert_drifter_to_zarr


def _write_drifter_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = ["ID", "time", "latitude", "longitude", "drogue_lost_date", "DrogueLength"]
    units = ["", "UTC", "degrees_north", "degrees_east", "UTC", ""]
    frame = pd.DataFrame(rows, columns=columns)

    lines = [",".join(columns), ",".join(units)]
    lines.extend(
        ",".join("" if pd.isna(value) else str(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_convert_drifter_to_zarr_creates_parcels_compatible_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "drifter.csv"
    _write_drifter_csv(
        csv_path,
        [
            {
                "ID": 103798,
                "time": "2012-04-23T18:00:00Z",
                "latitude": 44.66,
                "longitude": -9.975,
                "drogue_lost_date": "2012-04-24T06:00:00Z",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-24T00:00:00Z",
                "latitude": 44.626,
                "longitude": -9.939,
                "drogue_lost_date": "2012-04-24T06:00:00Z",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-24T06:00:00Z",
                "latitude": 44.6,
                "longitude": -9.9,
                "drogue_lost_date": "2012-04-24T06:00:00Z",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 200001,
                "time": "2012-04-23T18:00:00Z",
                "latitude": 40.0,
                "longitude": 10.0,
                "drogue_lost_date": "",
                "DrogueLength": "",
            },
            {
                "ID": 200001,
                "time": "2012-04-24T00:00:00Z",
                "latitude": 40.2,
                "longitude": 10.2,
                "drogue_lost_date": "",
                "DrogueLength": "",
            },
        ],
    )

    output_path = tmp_path / "drifter_output.zarr"
    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(output_path)},
        "processing": {
            "drogue": {
                "clip_after_loss": True,
                "minimum_length_m": 5.0,
            },
            "segment": {
                "mode": "ignore",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
        },
    }

    convert_drifter_to_zarr(config)

    ds = open_parcels_dataset(output_path)
    schema = resolve_parcels_schema(ds, coordinates=DatasetCoordinatesConfig())
    table = build_trajectory_table(ds, schema=schema, extra_vars=["platform_code"])

    assert ds.dims["trajectory"] == 1
    assert ds.dims["obs"] == 2
    assert ds["platform_code"].dims == ("trajectory",)
    assert ds["platform_code"].values.tolist() == [103798]
    assert np.allclose(ds["z"].values[0, :2], [0.0, 0.0])
    assert zarr.open_group(str(output_path), mode="r")["lon"].chunks == (1, 1)
    assert table["trajectory"].nunique() == 1
    assert table["platform_code"].tolist()[:2] == [103798, 103798]
    assert table["time"].tolist()[:2] == [
        pd.Timestamp("2012-04-23T18:00:00"),
        pd.Timestamp("2012-04-24T00:00:00"),
    ]


def test_segment_mode_split_as_new_splits_irregular_cadence(tmp_path: Path) -> None:
    csv_path = tmp_path / "drifter_split.csv"
    _write_drifter_csv(
        csv_path,
        [
            {
                "ID": 103798,
                "time": "2012-04-23T00:00:00Z",
                "latitude": -50.0,
                "longitude": 10.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T06:00:00Z",
                "latitude": -49.8,
                "longitude": 10.3,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T12:00:00Z",
                "latitude": -49.6,
                "longitude": 10.6,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T15:00:00Z",
                "latitude": -49.5,
                "longitude": 10.7,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T21:00:00Z",
                "latitude": -49.1,
                "longitude": 11.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "split.zarr")},
        "processing": {
            "segment": {
                "mode": "split_as_new",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
        },
    }

    trajectories = convert_drifter_to_dataframe(config)

    assert len(trajectories) == 2
    assert trajectories[0]["trajectory"].iloc[0] == 0
    assert trajectories[1]["trajectory"].iloc[0] == 1
    assert trajectories[0]["platform_code"].iloc[0] == 103798
    assert trajectories[1]["platform_code"].iloc[0] == 103798
    assert len(trajectories[0]) == 3
    assert len(trajectories[1]) == 2


def test_segment_mode_longest_keeps_longest_irregular_segment(tmp_path: Path) -> None:
    csv_path = tmp_path / "drifter_longest.csv"
    _write_drifter_csv(
        csv_path,
        [
            {
                "ID": 103798,
                "time": "2012-04-23T00:00:00Z",
                "latitude": -50.0,
                "longitude": 10.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T06:00:00Z",
                "latitude": -49.8,
                "longitude": 10.3,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T12:00:00Z",
                "latitude": -49.6,
                "longitude": 10.6,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2012-04-23T15:00:00Z",
                "latitude": -49.5,
                "longitude": 10.7,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "longest.zarr")},
        "processing": {
            "segment": {
                "mode": "longest",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
        },
    }

    trajectories = convert_drifter_to_dataframe(config)

    assert len(trajectories) == 1
    assert trajectories[0]["platform_code"].iloc[0] == 103798
    assert trajectories[0]["trajectory"].iloc[0] == 0
    assert len(trajectories[0]) == 3
    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2012-04-23T00:00:00"),
        pd.Timestamp("2012-04-23T06:00:00"),
        pd.Timestamp("2012-04-23T12:00:00"),
    ]


def test_region_cut_and_resample_keep_only_points_after_entry(tmp_path: Path) -> None:
    csv_path = tmp_path / "drifter_regions.csv"
    _write_drifter_csv(
        csv_path,
        [
            {
                "ID": 103798,
                "time": "2026-04-15T00:00:00Z",
                "latitude": 30.0,
                "longitude": 0.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2026-04-15T06:00:00Z",
                "latitude": 36.8,
                "longitude": 14.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2026-04-15T12:00:00Z",
                "latitude": 37.0,
                "longitude": 14.4,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "regions.zarr")},
        "processing": {
            "segment": {
                "mode": "ignore",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
            "regions": {
                "names_or_labels": ["sic"],
                "cut_from_first_entry": True,
            },
            "resample": {
                "enabled": True,
                "frequency": "3H",
                "interpolate": "time",
            },
        },
    }

    trajectories = convert_drifter_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2026-04-15T06:00:00"),
        pd.Timestamp("2026-04-15T09:00:00"),
        pd.Timestamp("2026-04-15T12:00:00"),
    ]
    assert np.isclose(trajectory["lon"].iloc[1], 14.2)
    assert np.isclose(trajectory["lat"].iloc[1], 36.9)


def test_shared_time_with_shift_start_to_reference_uses_common_time_grid(tmp_path: Path) -> None:
    csv_path_1 = tmp_path / "drifter_align_1.csv"
    csv_path_2 = tmp_path / "drifter_align_2.csv"

    _write_drifter_csv(
        csv_path_1,
        [
            {
                "ID": 103798,
                "time": "2026-04-15T00:00:00Z",
                "latitude": -55.0,
                "longitude": 10.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2026-04-17T00:00:00Z",
                "latitude": -54.0,
                "longitude": 12.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    _write_drifter_csv(
        csv_path_2,
        [
            {
                "ID": 103799,
                "time": "2026-05-01T00:00:00Z",
                "latitude": -50.0,
                "longitude": 20.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103799,
                "time": "2026-05-04T00:00:00Z",
                "latitude": -49.0,
                "longitude": 21.5,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path_1), str(csv_path_2)]},
        "output": {"path": str(tmp_path / "aligned.zarr")},
        "processing": {
            "segment": {
                "mode": "ignore",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
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

    trajectories = convert_drifter_to_dataframe(config)

    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory["time"].iloc[0] == pd.Timestamp("2020-01-01T00:00:00")

    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00"),
        pd.Timestamp("2020-01-02T00:00:00"),
        pd.Timestamp("2020-01-03T00:00:00"),
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

    assert not trajectories[0][["lon", "lat"]].isna().any().any()
    assert not trajectories[1][["lon", "lat"]].isna().any().any()


def test_reference_time_anchors_non_shared_resampling_grid(tmp_path: Path) -> None:
    csv_path = tmp_path / "drifter_reference_time_grid.csv"
    _write_drifter_csv(
        csv_path,
        [
            {
                "ID": 103798,
                "time": "2026-04-15T06:00:00Z",
                "latitude": -55.0,
                "longitude": 10.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2026-04-17T06:00:00Z",
                "latitude": -54.0,
                "longitude": 12.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "reference_time_grid.zarr")},
        "processing": {
            "segment": {
                "mode": "ignore",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
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

    trajectories = convert_drifter_to_dataframe(config)

    assert len(trajectories) == 1
    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2026-04-16T00:00:00"),
        pd.Timestamp("2026-04-17T00:00:00"),
    ]


def test_shift_start_to_reference_works_without_shared_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "drifter_shift_only.csv"
    _write_drifter_csv(
        csv_path,
        [
            {
                "ID": 103798,
                "time": "2026-04-15T06:00:00Z",
                "latitude": -55.0,
                "longitude": 10.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
            {
                "ID": 103798,
                "time": "2026-04-17T06:00:00Z",
                "latitude": -54.0,
                "longitude": 12.0,
                "drogue_lost_date": "",
                "DrogueLength": "5.2 m",
            },
        ],
    )

    config = {
        "input": {"csv_files": [str(csv_path)]},
        "output": {"path": str(tmp_path / "shift_only.zarr")},
        "processing": {
            "segment": {
                "mode": "ignore",
                "step_hours": 6.0,
                "tolerance_minutes": 30.0,
            },
            "resample": {
                "enabled": True,
                "frequency": "1d",
                "interpolate": "time",
                "reference_time": "2020-01-01T00:00:00Z",
                "shared_time": False,
                "shift_start_to_reference": True,
            },
        },
    }

    trajectories = convert_drifter_to_dataframe(config)

    assert len(trajectories) == 1
    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00"),
        pd.Timestamp("2020-01-02T00:00:00"),
        pd.Timestamp("2020-01-03T00:00:00"),
    ]