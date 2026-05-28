from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kinematicparcels.postprocessing.config.models import DatasetCoordinatesConfig
from kinematicparcels.postprocessing.io.parcels import build_trajectory_table, open_parcels_dataset, resolve_parcels_schema
from kinematicparcels.postprocessing.workflows.base_products import _expand_memberwise_rows
from kinematicparcels.tools.couple_trajectories import build_coupled_trajectories, couple_trajectories_to_zarr
from kinematicparcels.tools.zarr_writer import build_dataset_from_trajectories


def _write_input_dataset(path: Path, trajectories: list[pd.DataFrame]) -> None:
    ds = build_dataset_from_trajectories(trajectories, trajectory_level_columns={"platform_code"})
    ds.to_zarr(path, mode="w")


def test_build_coupled_trajectories_uses_absolute_time_overlap(tmp_path: Path) -> None:
    input_path = tmp_path / "input_overlap.zarr"
    trajectories = [
        pd.DataFrame(
            {
                "trajectory": [0, 0, 0],
                "obs": [0, 1, 2],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                    ]
                ),
                "lon": [10.0, 10.1, 10.2],
                "lat": [45.0, 45.0, 45.0],
                "z": [1000.0, 1000.0, 1000.0],
                "platform_code": [1001, 1001, 1001],
            }
        ),
        pd.DataFrame(
            {
                "trajectory": [1, 1, 1],
                "obs": [0, 1, 2],
                "time": pd.to_datetime(
                    [
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                        "2026-01-04T00:00:00",
                    ]
                ),
                "lon": [10.12, 10.18, 10.3],
                "lat": [45.0, 45.0, 45.0],
                "z": [1000.0, 1000.0, 1000.0],
                "platform_code": [1002, 1002, 1002],
            }
        ),
    ]
    _write_input_dataset(input_path, trajectories)

    coupled = build_coupled_trajectories(input_path, threshold_km=20.0)

    assert len(coupled) == 1
    pair = coupled[0]
    assert pair["time"].tolist() == [
        pd.Timestamp("2026-01-02T00:00:00"),
        pd.Timestamp("2026-01-03T00:00:00"),
    ]
    assert pair["obs"].tolist() == [0, 1]
    assert pair["platform_code_1"].iloc[0] == 1001
    assert pair["platform_code_2"].iloc[0] == 1002


def test_build_coupled_trajectories_trims_to_closest_approach_and_minimum_life(tmp_path: Path) -> None:
    input_path = tmp_path / "input_min_life.zarr"
    trajectories = [
        pd.DataFrame(
            {
                "trajectory": [0, 0, 0, 0],
                "obs": [0, 1, 2, 3],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                        "2026-01-04T00:00:00",
                    ]
                ),
                "lon": [0.30, 0.05, 0.00, 0.02],
                "lat": [0.0, 0.0, 0.0, 0.0],
                "z": [1000.0, 1000.0, 1000.0, 1000.0],
                "platform_code": [2001, 2001, 2001, 2001],
            }
        ),
        pd.DataFrame(
            {
                "trajectory": [1, 1, 1, 1],
                "obs": [0, 1, 2, 3],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                        "2026-01-04T00:00:00",
                    ]
                ),
                "lon": [0.00, 0.00, 0.00, 0.00],
                "lat": [0.0, 0.0, 0.0, 0.0],
                "z": [1000.0, 1000.0, 1000.0, 1000.0],
                "platform_code": [2002, 2002, 2002, 2002],
            }
        ),
    ]
    _write_input_dataset(input_path, trajectories)

    coupled = build_coupled_trajectories(
        input_path,
        threshold_km=40.0,
        minimum_life_days=1.0,
    )

    assert len(coupled) == 1
    pair = coupled[0]
    assert pair["time"].tolist() == [
        pd.Timestamp("2026-01-03T00:00:00"),
        pd.Timestamp("2026-01-04T00:00:00"),
    ]

    rejected = build_coupled_trajectories(
        input_path,
        threshold_km=40.0,
        minimum_life_days=2.0,
    )
    assert rejected == []


def test_couple_trajectories_to_zarr_writes_grouped_entity_output(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input_grouped.zarr"
    output_path = tmp_path / "paired_output.zarr"
    trajectories = [
        pd.DataFrame(
            {
                "trajectory": [0, 0, 0],
                "obs": [0, 1, 2],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                    ]
                ),
                "lon": [179.8, -179.9, -179.8],
                "lat": [10.0, 10.0, 10.0],
                "z": [1000.0, 1000.0, 1000.0],
                "platform_code": [3001, 3001, 3001],
            }
        ),
        pd.DataFrame(
            {
                "trajectory": [1, 1, 1],
                "obs": [0, 1, 2],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                    ]
                ),
                "lon": [-179.7, -179.8, -179.7],
                "lat": [10.2, 10.2, 10.2],
                "z": [1000.0, 1000.0, 1000.0],
                "platform_code": [3002, 3002, 3002],
            }
        ),
    ]
    _write_input_dataset(input_path, trajectories)

    ds = couple_trajectories_to_zarr(input_path, output_path, threshold_km=80.0)

    assert ds.sizes["trajectory"] == 1
    assert ds["group_size"].values.tolist() == [2]
    assert ds["platform_code_1"].values.tolist() == [3001]
    assert ds["platform_code_2"].values.tolist() == [3002]
    assert ds.attrs["source"] == "Trajectory pair coupling"
    assert ds.attrs["pair_threshold_km"] == 80.0

    opened = open_parcels_dataset(output_path)
    schema = resolve_parcels_schema(opened, coordinates=DatasetCoordinatesConfig())
    table = build_trajectory_table(
        opened,
        schema=schema,
        extra_vars=[
            "group_id",
            "group_size",
            "center_lon",
            "lon_1",
            "lat_1",
            "lon_2",
            "lat_2",
            "platform_code_1",
            "platform_code_2",
        ],
    )
    expanded = _expand_memberwise_rows(table)
    captured = capsys.readouterr()

    assert set(expanded["group_member"]) == {1, 2}
    assert expanded["platform_code_1"].unique().tolist() == [3001]
    assert expanded["platform_code_2"].unique().tolist() == [3002]
    assert np.all(np.abs(table["center_lon"].to_numpy(dtype=float)) > 179.0)
    assert "Found 1 couples. Average length: 1.00 days." in captured.out


def test_build_coupled_trajectories_region_filter_uses_in_region_minimum(tmp_path: Path) -> None:
    input_path = tmp_path / "input_region_filter.zarr"
    trajectories = [
        pd.DataFrame(
            {
                "trajectory": [0, 0, 0, 0],
                "obs": [0, 1, 2, 3],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                        "2026-01-04T00:00:00",
                    ]
                ),
                "lon": [10.00, 15.000, 15.050, 15.100],
                "lat": [0.00, 36.000, 36.100, 36.200],
                "z": [1000.0, 1000.0, 1000.0, 1000.0],
                "platform_code": [4001, 4001, 4001, 4001],
            }
        ),
        pd.DataFrame(
            {
                "trajectory": [1, 1, 1, 1],
                "obs": [0, 1, 2, 3],
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-02T00:00:00",
                        "2026-01-03T00:00:00",
                        "2026-01-04T00:00:00",
                    ]
                ),
                "lon": [10.020, 15.005, 15.070, 15.150],
                "lat": [0.000, 36.002, 36.110, 36.210],
                "z": [1000.0, 1000.0, 1000.0, 1000.0],
                "platform_code": [4002, 4002, 4002, 4002],
            }
        ),
    ]
    _write_input_dataset(input_path, trajectories)

    coupled = build_coupled_trajectories(
        input_path,
        threshold_km=5.0,
        regions=("med_cpf",),
    )

    assert len(coupled) == 1
    pair = coupled[0]
    assert pair["time"].tolist() == [
        pd.Timestamp("2026-01-02T00:00:00"),
        pd.Timestamp("2026-01-03T00:00:00"),
        pd.Timestamp("2026-01-04T00:00:00"),
    ]

    rejected = build_coupled_trajectories(
        input_path,
        threshold_km=5.0,
        regions=("PFuef",),
    )
    assert rejected == []