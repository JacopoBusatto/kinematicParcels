from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from parcels import AdvectionRK4, FieldSet

from kinematicparcels.runner.run_experiment import (
    _attach_boundary_halo_constants,
    _build_release_points_from_region,
    build_particleset,
    build_fieldset,
    build_release,
    run_simulation,
)
from kinematicparcels.runner.kernels import DeleteParticleIfTooOld, WrapLongitudePeriodic
from kinematicparcels.runner.grouped_kernels import AdvectionRK4_Grouped, make_grouped_rk4_lkm_kernel
from kinematicparcels.utilities.init_checks import filter_inside_domain


ROOT = Path(__file__).resolve().parents[1]


def _base_cfg(field_pattern: str) -> dict:
    return {
        "experiment": {"name": "test_continuous", "output_dir": str(ROOT / "test_output")},
        "fieldset": {
            "file_pattern": field_pattern,
            "periodic_halo": False,
            "periodic_halo_size": 5,
            "variables": {
                "U": "x_sea_water_velocity",
                "V": "y_sea_water_velocity",
            },
            "dimensions": {
                "lon": "lon",
                "lat": "lat",
                "time": "time",
            },
            "mesh": "spherical",
        },
        "release": {
            "mode": "region_grid",
            "region_label": "med_cpf",
            "dlon": 0.5,
            "dlat": 0.5,
            "filter_domain": True,
            "depth": {
                "enabled": False,
                "values": [0],
            },
            "continuous": {
                "enabled": True,
                "release_interval": "12H",
                "release_period": "1D",
            },
        },
        "simulation": {
            "start_time": "2026-04-01 00:00",
            "runtime_days": 1.0,
            "dt_hours": 1.0,
            "outputdt_hours": 6.0,
            "particle_type": "scipy",
        },
        "output": {
            "zarr_name": "test_continuous.zarr",
        },
    }


def _write_constant_velocity_field(field_path: Path, *, u_value: float, v_value: float) -> None:
    lon = np.array([14.0, 14.5, 15.0, 15.5, 16.0], dtype=np.float32)
    lat = np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float32)
    time = np.array(
        [
            np.datetime64("2026-04-01T00:00:00", "ns"),
            np.datetime64("2026-04-02T00:00:00", "ns"),
        ],
        dtype="datetime64[ns]",
    )

    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "x_sea_water_velocity": (
                ("time", "lat", "lon"),
                np.full(shape, u_value, dtype=np.float32),
            ),
            "y_sea_water_velocity": (
                ("time", "lat", "lon"),
                np.full(shape, v_value, dtype=np.float32),
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(field_path)


def test_build_fieldset_applies_configured_periodic_halo_size():
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["fieldset"]["periodic_halo"] = True
    cfg["fieldset"]["periodic_halo_size"] = 7

    with patch.object(FieldSet, "add_periodic_halo", autospec=True) as add_periodic_halo:
        fieldset = build_fieldset(cfg)

    args, kwargs = add_periodic_halo.call_args
    assert args[0] is fieldset
    assert kwargs == {"zonal": True, "halosize": 7}
    assert fieldset.bh_periodic == 1.0


def test_attach_boundary_halo_constants_is_noop_when_disabled_with_2d_grid():
    fieldset = SimpleNamespace(
        U=SimpleNamespace(
            grid=SimpleNamespace(
                lon=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
                lat=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            )
        )
    )
    fieldset.add_constant = lambda name, value: setattr(fieldset, name, value)

    cfg = {
        "fieldset": {"periodic_halo": False},
        "simulation": {"boundary_halo": {"enabled": False, "n_cells": 1}},
    }

    _attach_boundary_halo_constants(fieldset, cfg)

    assert fieldset.bh_periodic == 0.0
    assert not hasattr(fieldset, "bh_lat_min")
    assert not hasattr(fieldset, "bh_lon_min")


def test_run_simulation_skips_boundary_halo_kernel_when_disabled(tmp_path: Path):
    class KernelToken:
        def __init__(self, names):
            self.names = list(names)

        def __add__(self, other):
            return KernelToken(self.names + other.names)

    class FakeParticleSet:
        def __init__(self):
            self.kernel_names = []
            self.executed_kernels = []

        def ParticleFile(self, name, outputdt):
            return SimpleNamespace(name=name, outputdt=outputdt)

        def Kernel(self, func):
            name = getattr(func, "__name__", str(func))
            self.kernel_names.append(name)
            return KernelToken([name])

        def execute(self, kernels, runtime, dt, output_file):
            self.executed_kernels.append(kernels.names)

    cfg = {
        "experiment": {"output_dir": str(tmp_path)},
        "output": {"zarr_name": "dummy.zarr"},
            "fieldset": {"periodic_halo": True},
        "simulation": {
            "runtime_days": 1.0,
            "dt_hours": 1.0,
            "outputdt_hours": 6.0,
            "boundary_halo": {"enabled": False},
        },
        "release": {"group": {"size": 1}},
    }

    pset = FakeParticleSet()
    fieldset = SimpleNamespace(add_constant=lambda name, value: None)

    with patch("kinematicparcels.runner.run_experiment._nullify_off_grid_zarr_records"):
        run_simulation(cfg, pset, fieldset)

    assert pset.executed_kernels == [["WrapLongitudePeriodic", "AdvectionRK4"]]
    assert "BoundaryHaloKill" not in pset.kernel_names


def test_run_simulation_skips_grouped_boundary_halo_kernel_when_disabled(tmp_path: Path):
    class KernelToken:
        def __init__(self, names):
            self.names = list(names)

        def __add__(self, other):
            return KernelToken(self.names + other.names)

    class FakeParticleSet:
        def __init__(self):
            self.kernel_names = []
            self.executed_kernels = []

        def ParticleFile(self, name, outputdt):
            return SimpleNamespace(name=name, outputdt=outputdt)

        def Kernel(self, func):
            name = getattr(func, "__name__", str(func))
            self.kernel_names.append(name)
            return KernelToken([name])

        def execute(self, kernels, runtime, dt, output_file):
            self.executed_kernels.append(kernels.names)

    cfg = {
        "experiment": {"output_dir": str(tmp_path)},
        "output": {"zarr_name": "dummy.zarr"},
            "fieldset": {"periodic_halo": True},
        "simulation": {
            "runtime_days": 1.0,
            "dt_hours": 1.0,
            "outputdt_hours": 6.0,
            "boundary_halo": {"enabled": False},
        },
        "release": {"group": {"size": 2}},
    }

    pset = FakeParticleSet()
    fieldset = SimpleNamespace(add_constant=lambda name, value: None)

    with patch("kinematicparcels.runner.run_experiment._nullify_off_grid_zarr_records"):
        run_simulation(cfg, pset, fieldset)

    executed = pset.executed_kernels[0]
    assert executed[0] == "WrapLongitudePeriodic_GroupedEntity"
    assert executed[-1] == "AdvectionRK4_Grouped"
    assert "BoundaryHaloKill_GroupedEntity" not in pset.kernel_names


def test_run_simulation_uses_direct_advection_when_no_runtime_kernels_needed(tmp_path: Path):
    class FakeParticleSet:
        def __init__(self):
            self.executed = []
            self.kernel_names = []

        def ParticleFile(self, name, outputdt):
            return SimpleNamespace(name=name, outputdt=outputdt)

        def Kernel(self, func):
            name = getattr(func, "__name__", str(func))
            self.kernel_names.append(name)
            raise AssertionError("Kernel() should not be called on the direct advection fast path")

        def execute(self, kernels, runtime, dt, output_file):
            self.executed.append(kernels)

    cfg = {
        "experiment": {"output_dir": str(tmp_path)},
        "output": {"zarr_name": "dummy.zarr"},
        "fieldset": {"periodic_halo": False},
        "simulation": {
            "runtime_days": 1.0,
            "dt_hours": 1.0,
            "outputdt_hours": 6.0,
            "boundary_halo": {"enabled": False},
        },
        "release": {"group": {"size": 1}},
    }

    pset = FakeParticleSet()
    fieldset = SimpleNamespace(add_constant=lambda name, value: None)

    with patch("kinematicparcels.runner.run_experiment._nullify_off_grid_zarr_records"):
        run_simulation(cfg, pset, fieldset)

    assert pset.executed == [AdvectionRK4]


def test_run_simulation_writes_grouped_release_state_before_first_advection(tmp_path: Path):
    field_path = tmp_path / "constant_grouped_velocity.nc"
    _write_constant_velocity_field(field_path, u_value=0.2, v_value=0.0)

    cfg = _base_cfg(str(field_path))
    cfg["experiment"]["output_dir"] = str(tmp_path / "grouped_initial_output")
    cfg["release"]["mode"] = "point_list"
    cfg["release"]["points"] = [{"lon": 15.0, "lat": 36.0}]
    cfg["release"]["continuous"] = {"enabled": False}
    cfg["release"]["group"] = {
        "size": 5,
        "radius_km": 1.0,
        "placement": "equal_angles",
    }

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)
    pset = build_particleset(
        cfg,
        fieldset,
        lons,
        lats,
        depths=depths,
        metadata_dict=metadata,
        release_times=release_times,
    )

    run_simulation(cfg, pset, fieldset)

    ds = xr.open_zarr(Path(cfg["experiment"]["output_dir"]) / cfg["output"]["zarr_name"])
    finite_times = ds.time.values[0][np.isfinite(ds.time.values[0])]

    np.testing.assert_array_equal(
        finite_times[:2],
        np.array(
            [
                np.datetime64("2026-04-01T00:00:00", "ns"),
                np.datetime64("2026-04-01T06:00:00", "ns"),
            ],
            dtype="datetime64[ns]",
        ),
    )
    assert np.isclose(ds.lon.values[0, 0], lons[0])
    assert np.isclose(ds.lat.values[0, 0], lats[0])
    assert np.isclose(ds.center_lon.values[0, 0], metadata["center_lon"][0])
    assert np.isclose(ds.center_lat.values[0, 0], metadata["center_lat"][0])
    assert np.isclose(ds.lon_1.values[0, 0], metadata["lon_1"][0])
    assert np.isclose(ds.lat_1.values[0, 0], metadata["lat_1"][0])
    assert ds.lon.values[0, 1] > ds.lon.values[0, 0]
    assert ds.lon_1.values[0, 1] > ds.lon_1.values[0, 0]


def test_run_simulation_continuous_release_has_no_duplicate_first_observation(tmp_path: Path):
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["experiment"]["output_dir"] = str(tmp_path / "continuous_initial_output")
    cfg["release"]["dlon"] = 100.0
    cfg["release"]["dlat"] = 100.0

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)
    pset = build_particleset(
        cfg,
        fieldset,
        lons,
        lats,
        depths=depths,
        metadata_dict=metadata,
        release_times=release_times,
    )

    run_simulation(cfg, pset, fieldset)

    ds = xr.open_zarr(Path(cfg["experiment"]["output_dir"]) / cfg["output"]["zarr_name"])

    first_times = []
    for row in ds.time.values:
        finite = row[np.isfinite(row)]
        first_times.append(finite[0])
        if len(finite) >= 2:
            assert finite[1] != finite[0]

    np.testing.assert_array_equal(
        np.sort(np.array(first_times, dtype="datetime64[ns]")),
        np.sort(np.unique(release_times)),
    )


def test_wrap_longitude_periodic_maps_particle_back_into_domain():
    particle = SimpleNamespace(lon=181.2)
    fieldset = SimpleNamespace(
        bh_periodic=1.0,
        periodic_lon_west=-180.0,
        periodic_lon_span=360.0,
    )

    WrapLongitudePeriodic(particle, fieldset, 0.0)
    assert np.isclose(particle.lon, -178.8)

    particle.lon = -181.2
    WrapLongitudePeriodic(particle, fieldset, 0.0)
    assert np.isclose(particle.lon, 178.8)


def test_region_grid_continuous_grouped_release_has_unique_group_ids():
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["release"]["group"] = {
        "size": 3,
        "radius_km": 0.0,
        "placement": "equal_angles",
    }

    fieldset = build_fieldset(cfg)

    base_lons, base_lats = _build_release_points_from_region(cfg["release"])
    base_lons, base_lats = filter_inside_domain(base_lons, base_lats, fieldset)
    n_base = len(base_lons)

    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    expected_steps = 3  # 0h, 12h, 24h
    expected_groups = n_base * expected_steps

    assert depths is None
    assert release_times is not None
    assert len(np.unique(release_times)) == expected_steps
    assert len(np.unique(metadata["group_id"])) == expected_groups
    assert np.all(metadata["group_size"] == 3)
    assert len(lons) == len(lats) == len(release_times) == expected_groups


def test_region_grid_continuous_backward_release_uses_past_times():
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["simulation"]["start_time"] = "2026-04-03 00:00"
    cfg["simulation"]["dt_hours"] = -1.0

    fieldset = build_fieldset(cfg)
    _, _, depths, metadata, release_times = build_release(cfg, fieldset)

    expected = np.array(
        [
            np.datetime64("2026-04-03T00:00:00"),
            np.datetime64("2026-04-02T12:00:00"),
            np.datetime64("2026-04-02T00:00:00"),
        ],
        dtype="datetime64[ns]",
    )

    assert depths is None
    assert "group_id" in metadata
    np.testing.assert_array_equal(np.unique(release_times), np.sort(expected))


def test_region_grid_continuous_release_repeats_all_depth_layers(tmp_path: Path):
    field_path = tmp_path / "synthetic_3d.nc"

    lon = np.array([14.5, 15.0, 15.5, 16.0], dtype=np.float32)
    lat = np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float32)
    depth = np.array([0.0, 10.0], dtype=np.float32)
    time = np.array([np.datetime64("2026-04-01T00:00:00", "ns")], dtype="datetime64[ns]")

    shape = (len(time), len(depth), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "x_sea_water_velocity": (("time", "depth", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            "y_sea_water_velocity": (("time", "depth", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
        },
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(field_path)

    cfg = _base_cfg(str(field_path))
    cfg["fieldset"]["dimensions"]["depth"] = "depth"
    cfg["release"]["depth"] = {
        "enabled": True,
        "values": [0.0, 10.0],
        "mode": "as_requested",
        "request_convention": "positive_down",
        "snap_method": "nearest",
        "remove_duplicate_depths": True,
    }

    fieldset = build_fieldset(cfg)

    base_lons, base_lats = _build_release_points_from_region(cfg["release"])
    base_lons, base_lats = filter_inside_domain(base_lons, base_lats, fieldset)
    n_base = len(base_lons)

    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    expected_steps = 3
    expected_depths = 2
    expected_particles = n_base * expected_steps * expected_depths

    assert depths is not None
    assert release_times is not None
    assert len(lons) == len(lats) == len(depths) == len(release_times) == expected_particles
    assert len(np.unique(release_times)) == expected_steps
    assert np.all(np.isin(np.unique(depths), [0.0, 10.0]))

    _, counts_per_time = np.unique(release_times, return_counts=True)
    assert np.all(counts_per_time == n_base * expected_depths)
    assert len(np.unique(metadata["group_id"])) == n_base * expected_steps


def test_build_particleset_copies_internal_release_time_when_max_age_is_enabled():
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["release"]["mode"] = "point_list"
    cfg["release"]["points"] = [{"lon": 15.0, "lat": 36.0}]
    cfg["release"]["continuous"]["max_age"] = "12H"

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)
    pset = build_particleset(
        cfg,
        fieldset,
        lons,
        lats,
        depths=depths,
        metadata_dict=metadata,
        release_times=release_times,
    )

    particle_times = np.array([particle.time for particle in pset], dtype=float)
    release_time_values = np.array([particle.release_time for particle in pset], dtype=float)

    np.testing.assert_allclose(release_time_values, particle_times)
    assert set(np.unique(release_time_values).tolist()) == {0.0, 43200.0, 86400.0}


def test_build_fieldset_reduces_regular_2d_coordinate_variables(tmp_path: Path):
    field_path = tmp_path / "regular_2d_coords.nc"

    xi = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    eta = np.array([45.0, 46.0], dtype=np.float32)
    depth = np.array([0.0], dtype=np.float32)
    time = np.array([np.datetime64("2026-04-01T00:00:00", "ns")], dtype="datetime64[ns]")

    lon2d = np.repeat(xi[:, None], len(eta), axis=1)
    lat2d = np.repeat(eta[None, :], len(xi), axis=0)
    shape = (len(time), len(depth), len(xi), len(eta))

    ds = xr.Dataset(
        {
            "x_sea_water_velocity": (("time", "depth", "xi_rho", "eta_rho"), np.zeros(shape, dtype=np.float32)),
            "y_sea_water_velocity": (("time", "depth", "xi_rho", "eta_rho"), np.zeros(shape, dtype=np.float32)),
            "lon_rho": (("xi_rho", "eta_rho"), lon2d),
            "lat_rho": (("xi_rho", "eta_rho"), lat2d),
        },
        coords={"time": time, "depth": depth},
    )
    ds.to_netcdf(field_path)

    cfg = _base_cfg(str(field_path))
    cfg["fieldset"]["dimensions"] = {
        "lon": "lon_rho",
        "lat": "lat_rho",
        "time": "time",
        "depth": "depth",
    }
    cfg["simulation"]["boundary_halo"] = {"enabled": True, "n_cells": 1}

    fieldset = build_fieldset(cfg)

    np.testing.assert_allclose(np.asarray(fieldset.U.grid.lon), xi)
    np.testing.assert_allclose(np.asarray(fieldset.U.grid.lat), eta)
    assert np.asarray(fieldset.U.grid.lon).ndim == 1
    assert np.asarray(fieldset.U.grid.lat).ndim == 1
    assert fieldset.bh_lon_min > float(xi.min())
    assert fieldset.bh_lat_min > float(eta.min())


def test_region_grid_non_continuous_max_age_requires_scheduled_release():
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["release"]["continuous"] = {"enabled": False, "max_age": "12H"}

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    with np.testing.assert_raises_regex(
        ValueError,
        r"release\.continuous\.max_age requires release\.continuous\.enabled=true",
    ):
        build_particleset(
            cfg,
            fieldset,
            lons,
            lats,
            depths=depths,
            metadata_dict=metadata,
            release_times=release_times,
        )


def test_circle_release_assigns_circle_id_to_all_group_members(tmp_path: Path):
    field_path = tmp_path / "synthetic_circle.nc"

    lon = np.array([14.0, 14.5, 15.0, 15.5, 16.0], dtype=np.float32)
    lat = np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float32)
    time = np.array([np.datetime64("2026-04-01T00:00:00", "ns")], dtype="datetime64[ns]")

    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "x_sea_water_velocity": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            "y_sea_water_velocity": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(field_path)

    cfg = _base_cfg(str(field_path))
    cfg["release"] = {
        "mode": "circle",
        "circle": {
            "lat": [35.6, 36.4],
            "lon": [14.6, 15.4],
            "dimension": "2D",
            "radius_km": [0.1, 0.1],
            "count_per_timestep": [2, 1],
            "release_interval": ["12H", "12H"],
            "release_period": ["0D", "0D"],
            "sampling": "uniform",
            "seed": 42,
            "out_of_domain_policy": "retry",
            "bathymetry_policy": "drop",
        },
        "group": {
            "size": 2,
            "radius_km": 0.0,
            "placement": "equal_angles",
        },
        "depth": {
            "enabled": False,
            "values": [0],
        },
    }

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    assert depths is None
    assert release_times is not None
    assert "circle_id" in metadata
    assert set(np.unique(metadata["circle_id"]).tolist()) == {1, 2}
    assert len(lons) == len(lats) == len(release_times) == len(metadata["circle_id"]) == 3
    assert np.all(metadata["group_size"] == 2)

    for group_id in np.unique(metadata["group_id"]):
        group_circle_ids = np.unique(metadata["circle_id"][metadata["group_id"] == group_id])
        assert len(group_circle_ids) == 1


def test_circle_release_uses_per_circle_start_times_in_backward_mode(tmp_path: Path):
    field_path = tmp_path / "synthetic_circle_start_times.nc"

    lon = np.array([14.0, 14.5, 15.0, 15.5, 16.0], dtype=np.float32)
    lat = np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float32)
    time = np.array([np.datetime64("2026-04-01T00:00:00", "ns")], dtype="datetime64[ns]")

    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "x_sea_water_velocity": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            "y_sea_water_velocity": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(field_path)

    cfg = _base_cfg(str(field_path))
    cfg["simulation"].pop("start_time")
    cfg["simulation"]["dt_hours"] = -1.0
    cfg["release"] = {
        "mode": "circle",
        "circle": {
            "lat": [35.6, 36.4],
            "lon": [14.6, 15.4],
            "dimension": "2D",
            "radius_km": [0.1, 0.1],
            "count_per_timestep": [1, 1],
            "release_interval": ["12H", "24H"],
            "release_period": ["1D", "1D"],
            "start_time": ["2026-04-03 00:00", "2026-04-02 00:00"],
            "sampling": "uniform",
            "seed": 42,
            "out_of_domain_policy": "retry",
            "bathymetry_policy": "drop",
        },
        "group": {"size": 1},
        "depth": {
            "enabled": False,
            "values": [0],
        },
    }

    fieldset = build_fieldset(cfg)
    _, _, depths, metadata, release_times = build_release(cfg, fieldset)

    assert depths is None
    assert set(np.unique(metadata["circle_id"]).tolist()) == {1, 2}

    circle1_times = np.sort(release_times[metadata["circle_id"] == 1])
    circle2_times = np.sort(release_times[metadata["circle_id"] == 2])

    np.testing.assert_array_equal(
        circle1_times,
        np.array(
            [
                np.datetime64("2026-04-02T00:00:00"),
                np.datetime64("2026-04-02T12:00:00"),
                np.datetime64("2026-04-03T00:00:00"),
            ],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(
        circle2_times,
        np.array(
            [
                np.datetime64("2026-04-01T00:00:00"),
                np.datetime64("2026-04-02T00:00:00"),
            ],
            dtype="datetime64[ns]",
        ),
    )


def test_circle_release_accepts_continuous_max_age_for_scheduled_particles(tmp_path: Path):
    field_path = tmp_path / "synthetic_circle_max_age.nc"

    lon = np.array([14.0, 14.5, 15.0, 15.5, 16.0], dtype=np.float32)
    lat = np.array([35.0, 35.5, 36.0, 36.5, 37.0], dtype=np.float32)
    time = np.array(
        [
            np.datetime64("2026-04-01T00:00:00", "ns"),
            np.datetime64("2026-04-01T12:00:00", "ns"),
            np.datetime64("2026-04-02T00:00:00", "ns"),
        ],
        dtype="datetime64[ns]",
    )

    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "x_sea_water_velocity": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            "y_sea_water_velocity": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(field_path)

    cfg = _base_cfg(str(field_path))
    cfg["release"] = {
        "mode": "circle",
        "circle": {
            "lat": 35.6,
            "lon": 14.6,
            "dimension": "2D",
            "radius_km": 0.1,
            "count_per_timestep": 1,
            "release_interval": "12H",
            "release_period": "1D",
            "sampling": "uniform",
            "seed": 42,
            "out_of_domain_policy": "retry",
            "bathymetry_policy": "drop",
        },
        "continuous": {
            "max_age": "12H",
        },
        "group": {"size": 1},
        "depth": {
            "enabled": False,
            "values": [0],
        },
    }

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)
    pset = build_particleset(
        cfg,
        fieldset,
        lons,
        lats,
        depths=depths,
        metadata_dict=metadata,
        release_times=release_times,
    )

    particle_times = np.array([particle.time for particle in pset], dtype=float)
    release_time_values = np.array([particle.release_time for particle in pset], dtype=float)

    np.testing.assert_allclose(release_time_values, particle_times)
    assert set(np.unique(release_time_values).tolist()) == {0.0, 43200.0, 86400.0}


def test_delete_particle_if_too_old_uses_absolute_elapsed_time():
    particle = SimpleNamespace(release_time=43200.0, deleted=False)

    def _delete() -> None:
        particle.deleted = True

    particle.delete = _delete
    fieldset = SimpleNamespace(kp_max_age_seconds=43200.0)

    DeleteParticleIfTooOld(particle, fieldset, 86400.0)
    assert not particle.deleted

    DeleteParticleIfTooOld(particle, fieldset, 86401.0)
    assert particle.deleted

    particle.deleted = False
    DeleteParticleIfTooOld(particle, fieldset, -1.0)
    assert particle.deleted


def test_grouped_entity_release_supports_group_size_five_without_lkm() -> None:
    cfg = _base_cfg(str(ROOT / "test_fields" / "zero_velocity_april_2026.nc"))
    cfg["release"]["group"] = {
        "size": 5,
        "radius_km": 0.0,
        "placement": "equal_angles",
    }
    cfg["lkm"] = {"enabled": False}

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    assert depths is None
    assert release_times is not None
    assert np.all(metadata["group_size"] == 5)
    assert "lon_5" in metadata
    assert "lat_5" in metadata
    assert len(lons) == len(lats) == len(metadata["group_id"])


class _ZeroUV:
    def __getitem__(self, key):
        return 0.0, 0.0


def test_grouped_entity_kernel_supports_group_size_five_with_and_without_lkm() -> None:
    kernel = make_grouped_rk4_lkm_kernel(5)
    assert kernel is AdvectionRK4_Grouped

    particle = SimpleNamespace(
        group_size=5,
        depth=0.0,
        dt=60.0,
        lon_1=15.0,
        lat_1=36.0,
        lon_2=15.0,
        lat_2=36.0,
        lon_3=15.0,
        lat_3=36.0,
        lon_4=15.0,
        lat_4=36.0,
        lon_5=15.0,
        lat_5=36.0,
        lon=15.0,
        lat=36.0,
        center_lon=15.0,
        center_lat=36.0,
    )

    fieldset_off = SimpleNamespace(UV=_ZeroUV())
    AdvectionRK4_Grouped(particle, fieldset_off, 0.0)
    assert np.isfinite(particle.lon_5)
    assert np.isfinite(particle.lat_5)

    lkm_modes = SimpleNamespace(
        n_modes=1,
        wavenumbers_1m=np.array([1.0e-3], dtype=float),
        amplitudes_ms=np.array([0.02], dtype=float),
        frequencies_hz=np.array([1.0e-4], dtype=float),
        osc_amplitudes_m=np.array([5.0], dtype=float),
        phases_rad=np.array([0.0], dtype=float),
    )
    fieldset_on = SimpleNamespace(UV=_ZeroUV(), lkm_modes=lkm_modes)
    AdvectionRK4_Grouped(particle, fieldset_on, 0.0)

    assert np.isfinite(particle.lon_5)
    assert np.isfinite(particle.lat_5)
