from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from parcels import FieldSet

from kinematicparcels.runner.run_experiment import (
    _build_release_points_from_region,
    build_fieldset,
    build_release,
)
from kinematicparcels.runner.kernels import WrapLongitudePeriodic
from kinematicparcels.utilities.init_checks import filter_inside_domain


ROOT = Path(__file__).resolve().parent


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
    assert len(lons) == len(lats) == len(release_times) == len(metadata["circle_id"]) == 6

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
