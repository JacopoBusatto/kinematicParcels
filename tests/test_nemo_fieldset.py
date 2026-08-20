from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from netCDF4 import Dataset
import numpy as np
import pytest
from parcels import FieldSet

from kinematicparcels.runner.run_experiment import build_fieldset


def _base_nemo_config(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    u_path = tmp_path / "velocity_U.nc"
    v_path = tmp_path / "velocity_V.nc"
    coordinates_path = tmp_path / "coordinates.nc"
    for path in (u_path, v_path, coordinates_path):
        path.touch()

    config = {
        "fieldset": {
            "loader": "nemo",
            "files": {
                "U": str(u_path),
                "V": str(v_path),
                "coordinates": str(coordinates_path),
            },
            "variables": {"U": "vozocrtx", "V": "vomecrty"},
            "dimensions": {
                "lon": "glamf",
                "lat": "gphif",
                "time": "time_counter",
            },
            "mesh": "flat",
            "allow_time_extrapolation": True,
            "time_periodic": False,
            "periodic_halo": False,
        },
        "release": {"filter_land": False},
        "simulation": {"boundary_halo": {"enabled": False}},
    }
    return config, u_path, v_path, coordinates_path


def _dummy_fieldset() -> SimpleNamespace:
    fieldset = SimpleNamespace()

    def add_constant(name: str, value: float) -> None:
        setattr(fieldset, name, value)

    fieldset.add_constant = add_constant
    return fieldset


def test_build_fieldset_routes_sorted_nemo_files_and_stationary_options(tmp_path: Path) -> None:
    config, u_path, v_path, coordinates_path = _base_nemo_config(tmp_path)
    u_second = tmp_path / "velocity_U_2.nc"
    v_second = tmp_path / "velocity_V_2.nc"
    u_path.rename(tmp_path / "velocity_U_1.nc")
    v_path.rename(tmp_path / "velocity_V_1.nc")
    u_second.touch()
    v_second.touch()
    u_paths = sorted(tmp_path.glob("velocity_U_*.nc"))
    v_paths = sorted(tmp_path.glob("velocity_V_*.nc"))
    config["fieldset"]["files"]["U"] = str(tmp_path / "velocity_U_*.nc")
    config["fieldset"]["files"]["V"] = str(tmp_path / "velocity_V_*.nc")
    config["fieldset"]["chunksize"] = False

    dummy = _dummy_fieldset()
    with patch.object(FieldSet, "from_nemo", return_value=dummy) as from_nemo:
        result = build_fieldset(config)

    assert result is dummy
    from_nemo.assert_called_once_with(
        filenames={
            "U": {
                "lon": [str(coordinates_path)],
                "lat": [str(coordinates_path)],
                "data": [str(path) for path in u_paths],
            },
            "V": {
                "lon": [str(coordinates_path)],
                "lat": [str(coordinates_path)],
                "data": [str(path) for path in v_paths],
            },
        },
        variables={"U": "vozocrtx", "V": "vomecrty"},
        dimensions={"lon": "glamf", "lat": "gphif", "time": "time_counter"},
        mesh="flat",
        allow_time_extrapolation=True,
        time_periodic=False,
        chunksize=False,
    )
    assert result._kp_source_files == {
        "U": [str(path) for path in u_paths],
        "V": [str(path) for path in v_paths],
        "coordinates": [str(coordinates_path)],
    }
    assert result.bh_periodic == 0.0


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        (
            "fieldset",
            {"periodic_halo": True},
            "periodic_halo is not currently supported",
        ),
        (
            "release",
            {"filter_land": True},
            "filter_land is not currently supported",
        ),
        (
            "simulation",
            {"boundary_halo": {"enabled": True}},
            "boundary_halo.enabled must be false",
        ),
    ],
)
def test_build_fieldset_rejects_nemo_incompatible_options(
    tmp_path: Path,
    section: str,
    changes: dict,
    message: str,
) -> None:
    config, _, _, _ = _base_nemo_config(tmp_path)
    config[section].update(changes)

    with patch.object(FieldSet, "from_nemo") as from_nemo:
        with pytest.raises(ValueError, match=message):
            build_fieldset(config)

    from_nemo.assert_not_called()


def test_build_fieldset_validates_nemo_file_mapping_before_loading(tmp_path: Path) -> None:
    config, _, _, _ = _base_nemo_config(tmp_path)
    del config["fieldset"]["files"]["coordinates"]

    with patch.object(FieldSet, "from_nemo") as from_nemo:
        with pytest.raises(ValueError, match="missing fieldset.files entries: coordinates"):
            build_fieldset(config)

    from_nemo.assert_not_called()


def test_build_fieldset_rejects_unpaired_nemo_velocity_files(tmp_path: Path) -> None:
    config, _, _, _ = _base_nemo_config(tmp_path)
    (tmp_path / "velocity_U_extra.nc").touch()
    config["fieldset"]["files"]["U"] = str(tmp_path / "velocity_U*.nc")

    with patch.object(FieldSet, "from_nemo") as from_nemo:
        with pytest.raises(ValueError, match="same number of U and V data files"):
            build_fieldset(config)

    from_nemo.assert_not_called()


def test_build_fieldset_defaults_to_legacy_netcdf_loader(tmp_path: Path) -> None:
    field_path = tmp_path / "legacy.nc"
    field_path.touch()
    config = {
        "fieldset": {
            "file_pattern": str(field_path),
            "variables": {"U": "uo", "V": "vo"},
            "dimensions": {"lon": "lon", "lat": "lat", "time": "time"},
            "mesh": "flat",
            "periodic_halo": False,
        },
        "simulation": {"boundary_halo": {"enabled": False}},
    }
    dummy = _dummy_fieldset()

    with (
        patch(
            "kinematicparcels.runner.run_experiment._needs_xarray_fieldset_fallback",
            return_value=False,
        ),
        patch.object(FieldSet, "from_netcdf", return_value=dummy) as from_netcdf,
        patch.object(FieldSet, "from_nemo") as from_nemo,
    ):
        result = build_fieldset(config)

    assert result is dummy
    from_netcdf.assert_called_once()
    from_nemo.assert_not_called()


def _write_f_node_coordinates(path: Path, *, ny: int = 4, nx: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    glamf = x[None, :] + 0.1 * y[:, None]
    gphif = y[:, None] + 0.05 * x[None, :]

    with Dataset(path, mode="w", format="NETCDF4") as dataset:
        dataset.createDimension("y", ny)
        dataset.createDimension("x", nx)
        lon = dataset.createVariable("glamf", "f4", ("y", "x"))
        lat = dataset.createVariable("gphif", "f4", ("y", "x"))
        lon[:] = glamf
        lat[:] = gphif

    return glamf, gphif


def _write_singleton_velocity(path: Path, variable: str, value: float, *, ny: int = 4, nx: int = 5) -> None:
    with Dataset(path, mode="w", format="NETCDF4") as dataset:
        dataset.createDimension("time_counter", 1)
        dataset.createDimension("y", ny)
        dataset.createDimension("x", nx)

        time = dataset.createVariable("time_counter", "f8", ("time_counter",))
        time.standard_name = "time"
        time.units = "seconds since 2022-08-03 00:00:00"
        time.calendar = "standard"
        time[:] = np.array([0.0], dtype=np.float64)

        velocity = dataset.createVariable(variable, "f4", ("time_counter", "y", "x"))
        velocity.units = "m s-1"
        velocity[:] = np.full((1, ny, nx), value, dtype=np.float32)


def test_real_nemo_singleton_c_grid_extrapolates_stationary_velocity(tmp_path: Path) -> None:
    u_path = tmp_path / "nemo_U.nc"
    v_path = tmp_path / "nemo_V.nc"
    coordinates_path = tmp_path / "nemo_coordinates.nc"
    glamf, gphif = _write_f_node_coordinates(coordinates_path)
    _write_singleton_velocity(u_path, "vozocrtx", 1.0)
    _write_singleton_velocity(v_path, "vomecrty", 0.0)

    config, _, _, _ = _base_nemo_config(tmp_path)
    config["fieldset"]["files"] = {
        "U": str(u_path),
        "V": str(v_path),
        "coordinates": str(coordinates_path),
    }

    fieldset = build_fieldset(config)

    assert fieldset.U.interp_method == "cgrid_velocity"
    assert fieldset.V.interp_method == "cgrid_velocity"
    assert fieldset.U.gridindexingtype == "nemo"
    assert fieldset.U.allow_time_extrapolation is True
    assert np.asarray(fieldset.U.grid.lon).ndim == 2
    assert np.asarray(fieldset.U.grid.lat).ndim == 2

    sample_lon = float(np.mean(glamf[1:3, 1:3]))
    sample_lat = float(np.mean(gphif[1:3, 1:3]))
    at_snapshot = np.asarray(
        fieldset.UV.eval(0.0, 0.0, sample_lat, sample_lon, applyConversion=False),
        dtype=float,
    )
    one_day_later = np.asarray(
        fieldset.UV.eval(86400.0, 0.0, sample_lat, sample_lon, applyConversion=False),
        dtype=float,
    )

    assert np.all(np.isfinite(at_snapshot))
    np.testing.assert_allclose(one_day_later, at_snapshot, rtol=0.0, atol=1e-7)
