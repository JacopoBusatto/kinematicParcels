from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels.tools.rafos_to_zarr import (
    convert_rafos_to_dataframe,
    convert_rafos_to_zarr,
)


def _write_rafos(path: Path, rows: list[dict[str, object]]) -> None:
    ds = xr.Dataset(
        data_vars={
            "trajectoryID": (
                ("row",),
                np.asarray([row["trajectoryID"] for row in rows], dtype="U64"),
            ),
            "floatID": (
                ("row",),
                np.asarray([row["floatID"] for row in rows], dtype="U32"),
            ),
            "float_type": (
                ("row",),
                np.asarray([row.get("float_type", "RAFOS") for row in rows], dtype="U32"),
            ),
            "time": (
                ("row",),
                np.asarray([row["time"] for row in rows], dtype="datetime64[ns]"),
            ),
            "latitude": (
                ("row",),
                np.asarray([row["lat"] for row in rows], dtype=float),
            ),
            "longitude": (
                ("row",),
                np.asarray([row["lon"] for row in rows], dtype=float),
            ),
            "surface_date": (
                ("row",),
                np.asarray([row.get("surface_date", "2099-01-01") for row in rows], dtype="datetime64[ns]"),
            ),
            "pressure": (
                ("row",),
                np.asarray([row["pressure"] for row in rows], dtype=float),
            ),
        }
    )
    ds.to_netcdf(path)


def test_rafos_uses_float_id_and_trajectory_id_pair_and_clips_surface_date(tmp_path: Path) -> None:
    nc_path = tmp_path / "rafos.nc"
    _write_rafos(
        nc_path,
        [
            {
                "floatID": "42",
                "trajectoryID": "float 10 of North Atlantic",
                "time": "2020-01-01",
                "lat": -50.0,
                "lon": 10.0,
                "pressure": 1000.0,
                "surface_date": "2020-01-03",
            },
            {
                "floatID": "42",
                "trajectoryID": "float 10 of North Atlantic",
                "time": "2020-01-02",
                "lat": -50.5,
                "lon": 10.5,
                "pressure": 1005.0,
                "surface_date": "2020-01-03",
            },
            {
                "floatID": "42",
                "trajectoryID": "float 10 of North Atlantic",
                "time": "2020-01-03",
                "lat": -51.0,
                "lon": 11.0,
                "pressure": 1010.0,
                "surface_date": "2020-01-03",
            },
            {
                "floatID": "42",
                "trajectoryID": "float 10 of second mission",
                "time": "2020-02-01",
                "lat": -60.0,
                "lon": 20.0,
                "pressure": 2000.0,
            },
            {
                "floatID": "42",
                "trajectoryID": "float 10 of second mission",
                "time": "2020-02-02",
                "lat": -60.5,
                "lon": 20.5,
                "pressure": 2010.0,
            },
            {
                "floatID": "N/A",
                "trajectoryID": "fallback trajectory",
                "time": "2020-03-01",
                "lat": -45.0,
                "lon": -10.0,
                "pressure": 800.0,
            },
        ],
    )

    config = {
        "input": {"netcdf_files": [str(nc_path)]},
        "output": {"path": str(tmp_path / "rafos.zarr")},
    }

    trajectories = convert_rafos_to_dataframe(config)

    assert len(trajectories) == 3
    platform_codes = sorted(trajectory["platform_code"].iloc[0] for trajectory in trajectories)
    assert platform_codes == [
        "42::float 10 of North Atlantic",
        "42::float 10 of second mission",
        "N/A::fallback trajectory",
    ]
    first = next(
        trajectory
        for trajectory in trajectories
        if trajectory["platform_code"].iloc[0] == "42::float 10 of North Atlantic"
    )
    assert first["time"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    ]
    assert first["floatID"].iloc[0] == "42"
    assert first["trajectoryID"].iloc[0] == "float 10 of North Atlantic"
    assert first["float_type"].iloc[0] == "RAFOS"
    assert np.allclose(first["z"].to_numpy(dtype=float), [1000.0, 1005.0])


def test_rafos_depth_bins_write_per_bin_zarr_with_string_metadata(tmp_path: Path) -> None:
    nc_path = tmp_path / "rafos.nc"
    _write_rafos(
        nc_path,
        [
            {
                "floatID": "42",
                "trajectoryID": "shallow",
                "time": "2020-01-01",
                "lat": -50.0,
                "lon": 10.0,
                "pressure": 1000.0,
                "float_type": "RAFOS",
            },
            {
                "floatID": "42",
                "trajectoryID": "shallow",
                "time": "2020-01-02",
                "lat": -50.5,
                "lon": 10.5,
                "pressure": 1010.0,
                "float_type": "RAFOS",
            },
            {
                "floatID": "42",
                "trajectoryID": "deep",
                "time": "2020-02-01",
                "lat": -60.0,
                "lon": 20.0,
                "pressure": 2000.0,
                "float_type": "SOFAR",
            },
            {
                "floatID": "42",
                "trajectoryID": "deep",
                "time": "2020-02-02",
                "lat": -60.5,
                "lon": 20.5,
                "pressure": 2010.0,
                "float_type": "SOFAR",
            },
        ],
    )

    output_path = tmp_path / "rafos.zarr"
    config = {
        "input": {"netcdf_files": [str(nc_path)]},
        "output": {"path": str(output_path)},
        "depth_bins": {
            "enabled": True,
            "output_mode": "per_bin",
            "bins": [
                {"label": "z0000_1100", "min": 0.0, "max": 1100.0},
                {"label": "z1900_inf", "min": 1900.0, "max": None},
            ],
        },
    }

    convert_rafos_to_zarr(config)

    shallow = xr.open_zarr(tmp_path / "rafos_z0000_1100.zarr")
    deep = xr.open_zarr(tmp_path / "rafos_z1900_inf.zarr")

    assert shallow.dims["trajectory"] == 1
    assert shallow.dims["obs"] == 2
    assert shallow["platform_code"].dims == ("trajectory",)
    assert shallow["platform_code"].values.tolist() == ["42::shallow"]
    assert shallow["floatID"].values.tolist() == ["42"]
    assert shallow["trajectoryID"].values.tolist() == ["shallow"]
    assert shallow["float_type"].values.tolist() == ["RAFOS"]
    assert shallow.attrs["depth_bin_label"] == "z0000_1100"
    assert np.allclose(shallow["z"].values[0, :2], [1000.0, 1010.0])

    assert deep["platform_code"].values.tolist() == ["42::deep"]
    assert deep["float_type"].values.tolist() == ["SOFAR"]
    assert deep.attrs["depth_bin_label"] == "z1900_inf"


def test_rafos_depth_bins_fill_missing_and_repair_isolated_outlier(tmp_path: Path) -> None:
    nc_path = tmp_path / "rafos.nc"
    _write_rafos(
        nc_path,
        [
            {
                "floatID": "42",
                "trajectoryID": "controlled",
                "time": "2020-01-01",
                "lat": -50.0,
                "lon": 10.0,
                "pressure": 1000.0,
            },
            {
                "floatID": "42",
                "trajectoryID": "controlled",
                "time": "2020-01-02",
                "lat": -50.5,
                "lon": 10.5,
                "pressure": np.nan,
            },
            {
                "floatID": "42",
                "trajectoryID": "controlled",
                "time": "2020-01-03",
                "lat": -51.0,
                "lon": 11.0,
                "pressure": 1005.0,
            },
            {
                "floatID": "42",
                "trajectoryID": "controlled",
                "time": "2020-01-04",
                "lat": -51.5,
                "lon": 11.5,
                "pressure": 5.0,
            },
            {
                "floatID": "42",
                "trajectoryID": "controlled",
                "time": "2020-01-05",
                "lat": -52.0,
                "lon": 12.0,
                "pressure": 1008.0,
            },
        ],
    )

    config = {
        "input": {"netcdf_files": [str(nc_path)]},
        "output": {"path": str(tmp_path / "rafos.zarr")},
        "depth_bins": {
            "enabled": True,
            "output_mode": "per_bin",
            "missing_depth": {
                "strategy": "bounded_neighbor",
                "max_fill_points": 1,
                "fill_between_same_bin_only": True,
            },
            "isolated_outlier": {
                "enabled": True,
                "max_run_points": 1,
                "require_same_neighbor_bin": True,
            },
            "bins": [
                {"label": "z0000_0900", "min": 0.0, "max": 900.0},
                {"label": "z0900_1100", "min": 900.0, "max": 1100.0},
            ],
        },
    }

    trajectories = convert_rafos_to_dataframe(config)

    assert len(trajectories) == 1
    assert trajectories[0]["depth_bin"].tolist() == ["z0900_1100"] * 5
    assert trajectories[0]["depth_bin_interval"].tolist() == ["[900, 1100)"] * 5
