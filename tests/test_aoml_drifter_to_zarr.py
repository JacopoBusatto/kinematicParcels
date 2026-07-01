from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels.tools.aoml_drifter_to_zarr import (
    convert_aoml_drifter_to_dataframe,
    convert_aoml_drifter_to_zarr,
)


def _write_aoml_drifter(
    path: Path,
    *,
    drifter_id: int,
    drogue_length: str,
    lat: list[float] | None = None,
    lon: list[float] | None = None,
) -> None:
    if lat is None:
        lat = [0.0, 1.0, 2.0, 3.0]
    if lon is None:
        lon = [10.0, 11.0, 12.0, 13.0]

    times = np.asarray(
        [["2020-01-01T00:00:00", "2020-01-01T06:00:00", "2020-01-01T12:00:00", "2020-01-01T18:00:00"]],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        data_vars={
            "ID": (("traj",), np.asarray([str(drifter_id).encode("ascii")], dtype="S15")),
            "WMO": (("traj",), np.asarray([4400507.0], dtype=float)),
            "start_date": (("traj",), np.asarray(["2020-01-01T06:00:00"], dtype="datetime64[ns]")),
            "end_date": (("traj",), np.asarray(["2020-01-02T00:00:00"], dtype="datetime64[ns]")),
            "drogue_lost_date": (("traj",), np.asarray(["2020-01-01T18:00:00"], dtype="datetime64[ns]")),
            "time": (("traj", "obs"), times),
            "latitude": (("traj", "obs"), np.asarray([lat], dtype=float)),
            "longitude": (("traj", "obs"), np.asarray([lon], dtype=float)),
        }
    )
    ds.attrs["DrogueLength"] = drogue_length
    ds.to_netcdf(path)


def test_aoml_drifter_filters_drogue_clips_loss_and_resamples(tmp_path: Path) -> None:
    kept_path = tmp_path / "drifter_6h_2613.nc"
    skipped_path = tmp_path / "drifter_6h_2614.nc"
    _write_aoml_drifter(kept_path, drifter_id=2613, drogue_length="5.2 m")
    _write_aoml_drifter(skipped_path, drifter_id=2614, drogue_length="3.0 m")

    config = {
        "input": {"netcdf_files": [str(kept_path), str(skipped_path)]},
        "output": {"path": str(tmp_path / "aoml.zarr")},
        "processing": {
            "drogue": {
                "clip_to_drogued_period": True,
                "minimum_length_m": 4.0,
            },
            "resample": {
                "enabled": True,
                "frequency": "3h",
                "interpolate": "time",
            },
        },
    }

    trajectories = convert_aoml_drifter_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["platform_code"].iloc[0] == 2613
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2020-01-01T06:00:00"),
        pd.Timestamp("2020-01-01T09:00:00"),
        pd.Timestamp("2020-01-01T12:00:00"),
    ]
    assert np.allclose(trajectory["lat"].to_numpy(dtype=float), [1.0, 1.5, 2.0])
    assert np.allclose(trajectory["lon"].to_numpy(dtype=float), [11.0, 11.5, 12.0])
    assert np.allclose(trajectory["z"].to_numpy(dtype=float), [0.0, 0.0, 0.0])


def test_convert_aoml_drifter_to_zarr_writes_parcels_essentials(tmp_path: Path) -> None:
    nc_path = tmp_path / "drifter_6h_2613.nc"
    output_path = tmp_path / "aoml_output.zarr"
    _write_aoml_drifter(nc_path, drifter_id=2613, drogue_length="5.2 m")

    config = {
        "input": {"netcdf_files": [str(nc_path)]},
        "output": {"path": str(output_path)},
        "processing": {
            "drogue": {
                "clip_to_drogued_period": True,
                "minimum_length_m": 4.0,
            },
        },
    }

    convert_aoml_drifter_to_zarr(config)

    ds = xr.open_zarr(output_path)
    assert ds.dims["trajectory"] == 1
    assert ds.dims["obs"] == 2
    assert set(ds.data_vars) == {"platform_code", "time", "lat", "lon", "z"}
    assert ds["platform_code"].dims == ("trajectory",)
    assert ds["platform_code"].values.tolist() == [2613]
    assert np.allclose(ds["z"].values[0, :2], [0.0, 0.0])
