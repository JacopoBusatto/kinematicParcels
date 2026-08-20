from __future__ import annotations

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from kinematicparcels.tools.nemo_snapshot import (
    build_coordinates_parser,
    extract_nemo_f_coordinates,
)


FILL_VALUE = np.float32(-999.0)


def _coordinate_values(*, leading_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    lon_2d = np.array(
        [
            [-128.0, -127.5, -127.0],
            [-128.1, -127.6, -127.1],
        ],
        dtype=np.float32,
    )
    lat_2d = np.array(
        [
            [50.0, 50.0, 50.0],
            [50.5, 50.5, 50.5],
        ],
        dtype=np.float32,
    )
    if leading_size is None:
        return lon_2d, lat_2d
    return (
        np.repeat(lon_2d[np.newaxis, ...], leading_size, axis=0),
        np.repeat(lat_2d[np.newaxis, ...], leading_size, axis=0),
    )


def _write_mesh(
    path: Path,
    *,
    lon_name: str = "glamf",
    lat_name: str = "gphif",
    leading_size: int | None = 1,
    lat_dimensions: tuple[str, ...] | None = None,
    lon_dtype: str = "f4",
) -> None:
    ny = 2
    nx = 3
    dimensions = ("y", "x") if leading_size is None else ("t", "y", "x")
    lat_dimensions = lat_dimensions or dimensions
    lon_values, lat_values = _coordinate_values(leading_size=leading_size)

    with Dataset(path, mode="w", format="NETCDF4") as dataset:
        dataset.title = "synthetic NEMO F-node mesh"
        dataset.source = "unit test"

        if leading_size is not None:
            dataset.createDimension("t", leading_size)
        dataset.createDimension("y", ny)
        dataset.createDimension("x", nx)
        if "x_alt" in lat_dimensions:
            dataset.createDimension("x_alt", nx)

        # Dimension coordinate variables are cheap dependencies of the selected
        # F-node arrays and should be retained.
        if leading_size is not None:
            t_coord = dataset.createVariable("t", "i4", ("t",))
            t_coord.long_name = "mesh record"
            t_coord[:] = np.arange(leading_size, dtype=np.int32)
        y_coord = dataset.createVariable("y", "i4", ("y",))
        x_coord = dataset.createVariable("x", "i4", ("x",))
        y_coord[:] = np.arange(ny, dtype=np.int32)
        x_coord[:] = np.arange(nx, dtype=np.int32)

        lon = dataset.createVariable(
            lon_name,
            lon_dtype,
            dimensions,
            fill_value=FILL_VALUE if lon_dtype != "S1" else None,
            zlib=True,
            complevel=2,
            shuffle=True,
            chunksizes=tuple(1 if name == "t" else len(dataset.dimensions[name]) for name in dimensions),
        )
        lon.standard_name = "longitude"
        lon.long_name = "longitude at NEMO F points"
        lon.units = "degrees_east"

        lat = dataset.createVariable(
            lat_name,
            "f4",
            lat_dimensions,
            fill_value=FILL_VALUE,
            zlib=True,
            complevel=2,
            shuffle=True,
            chunksizes=tuple(
                1 if name == "t" else len(dataset.dimensions[name])
                for name in lat_dimensions
            ),
        )
        lat.standard_name = "latitude"
        lat.long_name = "latitude at NEMO F points"
        lat.units = "degrees_north"

        if lon_dtype == "S1":
            lon[:] = np.full(lon_values.shape, b"x", dtype="S1")
        else:
            lon[:] = lon_values
        lat[:] = lat_values

        # These variables make the source representative of a large mesh file.
        # None is needed by FieldSet.from_nemo once glamf/gphif are extracted.
        dataset.createDimension("depth", 2)
        tmask_dimensions = ("depth", "y", "x")
        if leading_size is not None:
            tmask_dimensions = ("t",) + tmask_dimensions
        tmask = dataset.createVariable("tmask", "i1", tmask_dimensions)
        tmask[:] = 1
        e1t = dataset.createVariable("e1t", "f4", ("y", "x"))
        e1t[:] = 500.0
        nav_lon = dataset.createVariable("nav_lon", "f4", ("y", "x"))
        nav_lat = dataset.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon[:] = lon_values.reshape((-1, ny, nx))[0]
        nav_lat[:] = lat_values.reshape((-1, ny, nx))[0]


def test_extract_f_coordinates_preserves_singleton_grid_and_omits_mesh_fields(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "source mesh.nc"
    output = tmp_path / "reduced" / "f coordinates.nc"
    _write_mesh(mesh)

    result = extract_nemo_f_coordinates(mesh, output)

    assert result.output == output.resolve()
    assert result.lon_variable == "glamf"
    assert result.lat_variable == "gphif"
    assert result.dimensions == ("t", "y", "x")
    assert result.shape == (1, 2, 3)

    expected_lon, expected_lat = _coordinate_values(leading_size=1)
    with Dataset(output, mode="r") as dataset:
        assert dataset.title == "synthetic NEMO F-node mesh"
        assert dataset.source == "unit test"
        assert set(dataset.dimensions) == {"t", "y", "x"}
        assert set(dataset.variables) == {"t", "y", "x", "glamf", "gphif"}
        assert "tmask" not in dataset.variables
        assert "e1t" not in dataset.variables
        assert "nav_lon" not in dataset.variables
        assert "nav_lat" not in dataset.variables

        np.testing.assert_allclose(dataset.variables["glamf"][:], expected_lon)
        np.testing.assert_allclose(dataset.variables["gphif"][:], expected_lat)

        lon = dataset.variables["glamf"]
        lat = dataset.variables["gphif"]
        assert lon.dimensions == ("t", "y", "x")
        assert lat.dimensions == ("t", "y", "x")
        assert lon.dtype == np.dtype("float32")
        assert lat.dtype == np.dtype("float32")
        assert lon.getncattr("_FillValue") == FILL_VALUE
        assert lat.getncattr("_FillValue") == FILL_VALUE
        assert lon.standard_name == "longitude"
        assert lon.long_name == "longitude at NEMO F points"
        assert lon.units == "degrees_east"
        assert lat.standard_name == "latitude"
        assert lat.long_name == "latitude at NEMO F points"
        assert lat.units == "degrees_north"
        assert lon.filters()["zlib"] is True
        assert lat.filters()["zlib"] is True
        assert lon.filters()["complevel"] == 2
        assert lat.filters()["complevel"] == 2
        assert lon.chunking() == [1, 2, 3]
        assert lat.chunking() == [1, 2, 3]


def test_extract_f_coordinates_supports_2d_and_explicit_variable_names(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "mesh.nc"
    output = tmp_path / "coordinates.nc"
    _write_mesh(
        mesh,
        lon_name="f_longitude",
        lat_name="f_latitude",
        leading_size=None,
    )

    result = extract_nemo_f_coordinates(
        mesh,
        output,
        lon_var="f_longitude",
        lat_var="f_latitude",
    )

    assert result.dimensions == ("y", "x")
    assert result.shape == (2, 3)
    expected_lon, expected_lat = _coordinate_values()
    with Dataset(output, mode="r") as dataset:
        assert set(dataset.dimensions) == {"y", "x"}
        assert set(dataset.variables) == {"y", "x", "f_longitude", "f_latitude"}
        np.testing.assert_allclose(dataset.variables["f_longitude"][:], expected_lon)
        np.testing.assert_allclose(dataset.variables["f_latitude"][:], expected_lat)


@pytest.mark.parametrize(
    ("lon_var", "lat_var", "message"),
    [
        ("missing_lon", "gphif", "missing_lon"),
        ("glamf", "missing_lat", "missing_lat"),
        ("glamf", "glamf", "different"),
    ],
)
def test_extract_f_coordinates_rejects_missing_or_aliased_variables(
    tmp_path: Path,
    lon_var: str,
    lat_var: str,
    message: str,
) -> None:
    mesh = tmp_path / "mesh.nc"
    output = tmp_path / "coordinates.nc"
    _write_mesh(mesh)

    with pytest.raises(ValueError, match=message):
        extract_nemo_f_coordinates(mesh, output, lon_var=lon_var, lat_var=lat_var)

    assert not output.exists()


def test_extract_f_coordinates_rejects_mismatched_dimensions(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.nc"
    output = tmp_path / "coordinates.nc"
    _write_mesh(mesh, lat_dimensions=("t", "y", "x_alt"))

    with pytest.raises(ValueError, match="dimensions"):
        extract_nemo_f_coordinates(mesh, output)

    assert not output.exists()


def test_extract_f_coordinates_rejects_non_singleton_leading_dimension(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "mesh.nc"
    output = tmp_path / "coordinates.nc"
    _write_mesh(mesh, leading_size=2)

    with pytest.raises(ValueError, match="singleton|length one"):
        extract_nemo_f_coordinates(mesh, output)

    assert not output.exists()


def test_extract_f_coordinates_rejects_nonnumeric_coordinates(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.nc"
    output = tmp_path / "coordinates.nc"
    _write_mesh(mesh, lon_dtype="S1")

    with pytest.raises(ValueError, match="numeric"):
        extract_nemo_f_coordinates(mesh, output)

    assert not output.exists()


def test_extract_f_coordinates_protects_source_and_existing_output(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.nc"
    output = tmp_path / "coordinates.nc"
    _write_mesh(mesh)
    output.write_bytes(b"keep-existing-output")

    with pytest.raises(FileExistsError, match="overwrite"):
        extract_nemo_f_coordinates(mesh, output)
    assert output.read_bytes() == b"keep-existing-output"

    extract_nemo_f_coordinates(mesh, output, overwrite=True)
    with Dataset(output, mode="r") as dataset:
        assert "glamf" in dataset.variables
        assert "gphif" in dataset.variables

    with pytest.raises(ValueError, match="source"):
        extract_nemo_f_coordinates(mesh, mesh, overwrite=True)
    with Dataset(mesh, mode="r") as dataset:
        assert "tmask" in dataset.variables


def test_coordinates_parser_accepts_paths_names_and_safety_flags() -> None:
    args = build_coordinates_parser().parse_args(
        [
            "C:/data with spaces/KIT500 mesh.nc",
            "--output",
            "C:/reduced mesh/f_coordinates.nc",
            "--lon-var",
            "custom_glamf",
            "--lat-var",
            "custom_gphif",
            "--overwrite",
            "--allow-cloud-download",
        ]
    )

    assert args.mesh_file == "C:/data with spaces/KIT500 mesh.nc"
    assert args.output == "C:/reduced mesh/f_coordinates.nc"
    assert args.lon_var == "custom_glamf"
    assert args.lat_var == "custom_gphif"
    assert args.overwrite is True
    assert args.allow_cloud_download is True
