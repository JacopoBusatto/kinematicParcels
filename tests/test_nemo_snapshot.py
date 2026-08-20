from __future__ import annotations

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from kinematicparcels.tools.nemo_snapshot import (
    build_parser,
    extract_nemo_snapshot,
)


FILL_VALUE = np.float32(9.96921e36)


def _write_component(
    path: Path,
    variable_name: str,
    *,
    times: tuple[float, ...] = (0.0, 3600.0, 7200.0),
    time_units: str = "seconds since 2022-08-03 00:00:00",
    time_dim: str = "time_counter",
    time_name: str | None = None,
    annotate_time: bool = True,
) -> None:
    time_name = time_name or time_dim
    ny = 2
    nx = 3

    with Dataset(path, mode="w", format="NETCDF4") as dataset:
        dataset.title = "synthetic NEMO component"
        dataset.createDimension(time_dim, None)
        dataset.createDimension("depth", 1)
        dataset.createDimension("y", ny)
        dataset.createDimension("x", nx)
        dataset.createDimension("bnds", 2)

        time = dataset.createVariable(time_name, "f8", (time_dim,))
        if annotate_time:
            time.standard_name = "time"
            time.axis = "T"
        if time_units:
            time.units = time_units
            time.calendar = "standard"
        time[:] = np.asarray(times, dtype=np.float64)

        bounds = dataset.createVariable("time_bounds", "f8", (time_dim, "bnds"))
        bounds[:] = np.column_stack((np.asarray(times), np.asarray(times) + 3600.0))

        lon = dataset.createVariable("nav_lon", "f4", ("y", "x"))
        lat = dataset.createVariable("nav_lat", "f4", ("y", "x"))
        lon[:] = np.array([[-128.0, -127.5, -127.0], [-128.1, -127.6, -127.1]], dtype=np.float32)
        lat[:] = np.array([[50.0, 50.0, 50.0], [50.5, 50.5, 50.5]], dtype=np.float32)

        mask = dataset.createVariable("wet_mask", "i1", ("y", "x"))
        mask[:] = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int8)

        quality = dataset.createVariable("quality_flag", "i2", (time_dim, "y", "x"))
        quality[:] = np.arange(len(times) * ny * nx, dtype=np.int16).reshape(len(times), ny, nx)

        time_last = dataset.createVariable("time_last", "i2", ("y", "x", time_dim))
        time_last[:] = np.arange(ny * nx * len(times), dtype=np.int16).reshape(ny, nx, len(times))

        velocity = dataset.createVariable(
            variable_name,
            "f4",
            (time_dim, "depth", "y", "x"),
            fill_value=FILL_VALUE,
            zlib=True,
            complevel=2,
            shuffle=True,
            chunksizes=(1, 1, ny, nx),
        )
        velocity.units = "m s-1"
        velocity.long_name = f"synthetic {variable_name} velocity"
        velocity.missing_value = FILL_VALUE
        values = np.arange(len(times) * ny * nx, dtype=np.float32).reshape(len(times), 1, ny, nx)
        values[1, 0, 0, 0] = FILL_VALUE
        velocity.set_auto_maskandscale(False)
        velocity[:] = values


def test_extract_keeps_single_time_record_and_metadata(tmp_path: Path) -> None:
    u_input = tmp_path / "U source.nc"
    v_input = tmp_path / "V source.nc"
    u_output = tmp_path / "out" / "U snapshot.nc"
    v_output = tmp_path / "out" / "V snapshot.nc"
    _write_component(u_input, "uo")
    _write_component(v_input, "vo")

    result = extract_nemo_snapshot(
        u_input,
        v_input,
        u_output,
        v_output,
        time_index=1,
    )

    assert result.u_output == u_output.resolve()
    assert result.v_output == v_output.resolve()
    assert result.time_index == 1
    assert "2022-08-03 01:00:00" in result.selected_time

    with Dataset(u_output, mode="r") as dataset:
        assert dataset.title == "synthetic NEMO component"
        assert len(dataset.dimensions["time_counter"]) == 1
        assert dataset.dimensions["time_counter"].isunlimited()
        assert dataset.variables["time_counter"][:].tolist() == [3600.0]
        assert dataset.variables["time_bounds"].shape == (1, 2)
        assert dataset.variables["quality_flag"].shape == (1, 2, 3)
        assert dataset.variables["time_last"].shape == (2, 3, 1)
        np.testing.assert_allclose(
            dataset.variables["nav_lon"][:],
            np.array([[-128.0, -127.5, -127.0], [-128.1, -127.6, -127.1]], dtype=np.float32),
        )

        velocity = dataset.variables["uo"]
        velocity.set_auto_maskandscale(False)
        assert velocity.shape == (1, 1, 2, 3)
        assert velocity.dtype == np.dtype("float32")
        assert velocity.getncattr("_FillValue") == FILL_VALUE
        assert velocity.getncattr("missing_value") == FILL_VALUE
        assert velocity.units == "m s-1"
        assert velocity[0, 0, 0, 0] == FILL_VALUE
        assert velocity.filters()["zlib"] is True
        assert velocity.chunking()[0] == 1


def test_extract_compares_decoded_uv_times(tmp_path: Path) -> None:
    u_input = tmp_path / "u.nc"
    v_input = tmp_path / "v.nc"
    u_output = tmp_path / "u_snapshot.nc"
    v_output = tmp_path / "v_snapshot.nc"
    _write_component(
        u_input,
        "uo",
        times=(0.0, 3600.0, 7200.0),
        time_units="seconds since 2022-08-03 00:00:00",
    )
    _write_component(
        v_input,
        "vo",
        times=(0.0, 1.0, 2.0),
        time_units="hours since 2022-08-03 00:00:00",
    )

    extract_nemo_snapshot(u_input, v_input, u_output, v_output, time_index=1)

    assert u_output.exists()
    assert v_output.exists()


def test_timestamp_mismatch_is_rejected_before_outputs_are_written(tmp_path: Path) -> None:
    u_input = tmp_path / "u.nc"
    v_input = tmp_path / "v.nc"
    u_output = tmp_path / "u_snapshot.nc"
    v_output = tmp_path / "v_snapshot.nc"
    _write_component(u_input, "uo", times=(0.0, 3600.0))
    _write_component(v_input, "vo", times=(0.0, 3660.0))

    with pytest.raises(ValueError, match="timestamps do not match"):
        extract_nemo_snapshot(u_input, v_input, u_output, v_output, time_index=1)

    assert not u_output.exists()
    assert not v_output.exists()


def test_explicit_time_dimension_supports_unannotated_coordinate(tmp_path: Path) -> None:
    u_input = tmp_path / "u.nc"
    v_input = tmp_path / "v.nc"
    u_output = tmp_path / "u_snapshot.nc"
    v_output = tmp_path / "v_snapshot.nc"
    _write_component(
        u_input,
        "uo",
        time_dim="record",
        time_name="record_id",
        time_units="",
        annotate_time=False,
    )
    _write_component(
        v_input,
        "vo",
        time_dim="record",
        time_name="record_id",
        time_units="",
        annotate_time=False,
    )

    with pytest.raises(ValueError, match="Supply --time-dim"):
        extract_nemo_snapshot(u_input, v_input, u_output, v_output)

    extract_nemo_snapshot(
        u_input,
        v_input,
        u_output,
        v_output,
        time_index=2,
        time_dim="record",
    )

    with Dataset(u_output, mode="r") as dataset:
        assert len(dataset.dimensions["record"]) == 1
        assert dataset.variables["record_id"][:].tolist() == [7200.0]


def test_existing_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    u_input = tmp_path / "u.nc"
    v_input = tmp_path / "v.nc"
    u_output = tmp_path / "u_snapshot.nc"
    v_output = tmp_path / "v_snapshot.nc"
    _write_component(u_input, "uo")
    _write_component(v_input, "vo")
    u_output.write_bytes(b"keep-u")
    v_output.write_bytes(b"keep-v")

    with pytest.raises(FileExistsError, match="--overwrite"):
        extract_nemo_snapshot(u_input, v_input, u_output, v_output)

    assert u_output.read_bytes() == b"keep-u"
    assert v_output.read_bytes() == b"keep-v"

    extract_nemo_snapshot(
        u_input,
        v_input,
        u_output,
        v_output,
        overwrite=True,
    )
    with Dataset(u_output, mode="r") as dataset:
        assert len(dataset.dimensions["time_counter"]) == 1


def test_invalid_paths_and_indices_create_no_outputs(tmp_path: Path) -> None:
    u_input = tmp_path / "u.nc"
    v_input = tmp_path / "v.nc"
    u_output = tmp_path / "u_snapshot.nc"
    v_output = tmp_path / "v_snapshot.nc"
    _write_component(u_input, "uo")
    _write_component(v_input, "vo")

    with pytest.raises(ValueError, match="must not overwrite"):
        extract_nemo_snapshot(u_input, v_input, u_input, v_output, overwrite=True)

    with pytest.raises(ValueError, match="must be different"):
        extract_nemo_snapshot(u_input, v_input, u_output, u_output)

    with pytest.raises(IndexError, match="outside dimension"):
        extract_nemo_snapshot(u_input, v_input, u_output, v_output, time_index=99)

    assert not u_output.exists()
    assert not v_output.exists()


def test_parser_accepts_paths_with_spaces_and_safety_flags() -> None:
    args = build_parser().parse_args(
        [
            "C:/data with spaces/U.nc",
            "C:/data with spaces/V.nc",
            "--output-dir",
            "C:/snapshot output",
            "--time-index",
            "4",
            "--time-dim",
            "time_counter",
            "--overwrite",
            "--allow-cloud-download",
        ]
    )

    assert args.time_index == 4
    assert args.time_dim == "time_counter"
    assert args.overwrite is True
    assert args.allow_cloud_download is True

