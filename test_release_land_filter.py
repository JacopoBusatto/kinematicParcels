from pathlib import Path

import numpy as np
import xarray as xr
from parcels import FieldSet

from kinematicparcels.runner.run_experiment import build_fieldset, build_release
from kinematicparcels.utilities.init_checks import mask_inside_ocean


def _make_fieldset_with_masked_land() -> FieldSet:
    lon = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    lat = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    time = np.array([0.0], dtype=np.float64)

    u = np.ma.zeros((1, 3, 3), dtype=np.float32)
    v = np.ma.zeros((1, 3, 3), dtype=np.float32)
    u.mask = np.zeros((1, 3, 3), dtype=bool)
    v.mask = np.zeros((1, 3, 3), dtype=bool)

    # Treat the center cell as land/masked.
    u.mask[0, 1, 1] = True
    v.mask[0, 1, 1] = True

    data = {"U": u, "V": v}
    dimensions = {"lon": lon, "lat": lat, "time": time}
    return FieldSet.from_data(data=data, dimensions=dimensions, mesh="flat")


def test_build_release_filters_points_on_land_when_enabled():
    fieldset = _make_fieldset_with_masked_land()
    cfg = {
        "release": {
            "mode": "point_list",
            "filter_domain": True,
            "filter_land": True,
            "points": [
                {"lon": 0.2, "lat": 0.2},
                {"lon": 1.0, "lat": 1.0},
            ],
            "group": {"size": 1},
            "depth": {"enabled": False},
        },
        "simulation": {},
    }

    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    assert depths is None
    assert release_times is None
    assert lons.tolist() == [0.2]
    assert lats.tolist() == [0.2]
    assert metadata["group_size"].tolist() == [1]


def test_build_release_discards_group_if_a_member_is_on_land():
    fieldset = _make_fieldset_with_masked_land()
    cfg = {
        "release": {
            "mode": "point_list",
            "filter_domain": True,
            "filter_land": True,
            "points": [
                {"lon": 0.5, "lat": 1.0},
            ],
            "group": {
                "size": 2,
                "radius_km": 55.66,
                "placement": "equal_angles",
            },
            "depth": {"enabled": False},
        },
        "simulation": {},
    }

    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    assert depths is None
    assert release_times is None
    assert len(lons) == 0
    assert len(lats) == 0
    assert len(metadata["group_id"]) == 0


def test_continuous_grouped_release_times_stay_aligned_after_land_filtering():
    fieldset = _make_fieldset_with_masked_land()
    cfg = {
        "release": {
            "mode": "point_list",
            "filter_domain": True,
            "filter_land": True,
            "points": [
                {"lon": 0.5, "lat": 1.0},
            ],
            "continuous": {
                "enabled": True,
                "release_interval": "1H",
                "release_period": "1H",
            },
            "group": {
                "size": 2,
                "radius_km": 55.66,
                "placement": "equal_angles",
            },
            "depth": {"enabled": False},
        },
        "simulation": {
            "start_time": "2026-01-01 00:00",
        },
    }

    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)

    assert depths is None
    assert len(lons) == 0
    assert len(lats) == 0
    assert len(metadata["group_id"]) == 0
    assert release_times is not None
    assert len(release_times) == 0


def test_mask_inside_ocean_handles_lon_lat_variable_order(tmp_path):
    nc_path = Path(tmp_path) / "swapped_dims.nc"

    longitude = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    latitude = np.array([45.0, 46.0], dtype=np.float32)
    time = np.array([0], dtype=np.int32)

    u = np.zeros((1, 3, 2), dtype=np.float32)
    v = np.zeros((1, 3, 2), dtype=np.float32)

    # Mark only the far-east / north cell as land. Array order is (time, lon, lat).
    u[0, 2, 1] = np.nan
    v[0, 2, 1] = np.nan

    ds = xr.Dataset(
        {
            "uo": (("time", "longitude", "latitude"), u),
            "vo": (("time", "longitude", "latitude"), v),
        },
        coords={
            "time": time,
            "longitude": longitude,
            "latitude": latitude,
        },
    )
    ds.to_netcdf(nc_path)

    cfg = {
        "fieldset": {
            "file_pattern": str(nc_path),
            "variables": {"U": "uo", "V": "vo"},
            "dimensions": {"lon": "longitude", "lat": "latitude", "time": "time"},
            "mesh": "flat",
        }
    }

    fieldset = build_fieldset(cfg)
    mask = mask_inside_ocean(
        lons=np.array([10.0, 12.0]),
        lats=np.array([45.0, 46.0]),
        fieldset=fieldset,
    )

    assert mask.tolist() == [True, False]
