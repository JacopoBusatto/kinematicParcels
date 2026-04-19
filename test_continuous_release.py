from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from kinematicparcels.runner.run_experiment import (
    _build_release_points_from_region,
    build_fieldset,
    build_release,
)
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
