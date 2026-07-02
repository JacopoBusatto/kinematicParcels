from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels.tools.rtraj_to_zarr import (
    DepthBin,
    DepthBinConfig,
    RegionSelectionConfig,
    _depth_bin_output_path,
    _depth_histogram_counts,
    _apply_region_selection,
    _split_trajectory_by_depth_bin,
    convert_rtraj_to_dataframe,
    convert_rtraj_to_zarr,
)


def _char_array(value: str) -> np.ndarray:
    return np.asarray(list(value), dtype="S1")


def _write_rtraj(path: Path, *, pressure: list[float] | None = None) -> None:
    if pressure is None:
        pressure = [901.0, 902.0]

    ds = xr.Dataset(
        data_vars={
            "PLATFORM_NUMBER": (("STRING8",), _char_array("1902267")),
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0, 10.0, 20.0], dtype=float)),
            "JULD_ADJUSTED": (("N_MEASUREMENT",), np.asarray([np.nan, 11.0, np.nan], dtype=float)),
            "LATITUDE": (("N_MEASUREMENT",), np.asarray([-58.0, -60.0, -62.0], dtype=float)),
            "LONGITUDE": (("N_MEASUREMENT",), np.asarray([-80.0, -70.0, -60.0], dtype=float)),
            "CYCLE_NUMBER": (("N_MEASUREMENT",), np.asarray([1.0, 2.0, 3.0], dtype=float)),
            "CYCLE_NUMBER_ADJUSTED": (("N_MEASUREMENT",), np.asarray([np.nan, 20.0, np.nan], dtype=float)),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), np.asarray([1.0, 2.0], dtype=float)),
            "CYCLE_NUMBER_INDEX_ADJUSTED": (("N_CYCLE",), np.asarray([10.0, 20.0], dtype=float)),
            "REPRESENTATIVE_PARK_PRESSURE": (("N_CYCLE",), np.asarray(pressure, dtype=float)),
        }
    )
    for name in ["JULD", "JULD_ADJUSTED"]:
        ds[name].attrs["units"] = "days since 2000-01-01 00:00:00"
        ds[name].attrs["calendar"] = "gregorian"
    ds.to_netcdf(path)


def _write_depth_switch_rtraj(path: Path) -> None:
    z_values = np.asarray([1000.0, 1000.0, 1800.0, 1800.0, 1000.0], dtype=float)
    cycles = np.arange(1, len(z_values) + 1, dtype=float)
    ds = xr.Dataset(
        data_vars={
            "PLATFORM_NUMBER": (("STRING8",), _char_array("1902267")),
            "JULD": (("N_MEASUREMENT",), np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)),
            "JULD_ADJUSTED": (("N_MEASUREMENT",), np.full(len(z_values), np.nan, dtype=float)),
            "LATITUDE": (("N_MEASUREMENT",), np.asarray([-60.0, -60.5, -61.0, -61.5, -62.0], dtype=float)),
            "LONGITUDE": (("N_MEASUREMENT",), np.asarray([-70.0, -69.5, -69.0, -68.5, -68.0], dtype=float)),
            "CYCLE_NUMBER": (("N_MEASUREMENT",), cycles),
            "CYCLE_NUMBER_ADJUSTED": (("N_MEASUREMENT",), np.full(len(z_values), np.nan, dtype=float)),
            "CYCLE_NUMBER_INDEX": (("N_CYCLE",), cycles),
            "CYCLE_NUMBER_INDEX_ADJUSTED": (("N_CYCLE",), cycles),
            "REPRESENTATIVE_PARK_PRESSURE": (("N_CYCLE",), z_values),
        }
    )
    for name in ["JULD", "JULD_ADJUSTED"]:
        ds[name].attrs["units"] = "days since 2000-01-01 00:00:00"
        ds[name].attrs["calendar"] = "gregorian"
    ds.to_netcdf(path)


def _write_segmented_rtraj(path: Path, *, juld: list[float] | None = None, bad_cycle: int | None = None) -> None:
    cycles = np.arange(1, 6, dtype=float)
    if juld is None:
        juld = [0.0, 10.0, 20.0, 30.0, 40.0]
    juld_values = np.asarray(juld, dtype=float)
    base = cycles * 10.0
    transmission_end = base + 5.4
    if bad_cycle is not None:
        transmission_end[bad_cycle - 1] = base[bad_cycle - 1] + 7.0

    data_vars = {
        "PLATFORM_NUMBER": (("STRING8",), _char_array("1902267")),
        "JULD": (("N_MEASUREMENT",), juld_values),
        "JULD_ADJUSTED": (("N_MEASUREMENT",), np.full(len(cycles), np.nan, dtype=float)),
        "LATITUDE": (("N_MEASUREMENT",), np.asarray([-60.0, -60.5, -61.0, -61.5, -62.0], dtype=float)),
        "LONGITUDE": (("N_MEASUREMENT",), np.asarray([-70.0, -69.5, -69.0, -68.5, -68.0], dtype=float)),
        "CYCLE_NUMBER": (("N_MEASUREMENT",), cycles),
        "CYCLE_NUMBER_ADJUSTED": (("N_MEASUREMENT",), np.full(len(cycles), np.nan, dtype=float)),
        "CYCLE_NUMBER_INDEX": (("N_CYCLE",), cycles),
        "CYCLE_NUMBER_INDEX_ADJUSTED": (("N_CYCLE",), cycles),
        "REPRESENTATIVE_PARK_PRESSURE": (("N_CYCLE",), np.full(len(cycles), 1000.0, dtype=float)),
        "PRES": (("N_MEASUREMENT",), np.full(len(cycles), 2000.0, dtype=float)),
        "JULD_DESCENT_START": (("N_CYCLE",), base),
        "JULD_DESCENT_END": (("N_CYCLE",), base + 0.1),
        "JULD_DEEP_DESCENT_END": (("N_CYCLE",), base + 0.2),
        "JULD_PARK_START": (("N_CYCLE",), base + 0.3),
        "JULD_PARK_END": (("N_CYCLE",), base + 4.3),
        "JULD_DEEP_ASCENT_START": (("N_CYCLE",), base + 5.1),
        "JULD_ASCENT_START": (("N_CYCLE",), base + 5.2),
        "JULD_ASCENT_END": (("N_CYCLE",), base + 5.3),
        "JULD_TRANSMISSION_START": (("N_CYCLE",), base + 5.3),
        "JULD_TRANSMISSION_END": (("N_CYCLE",), transmission_end),
    }
    ds = xr.Dataset(data_vars=data_vars)
    for name in [
        "JULD",
        "JULD_ADJUSTED",
        "JULD_DESCENT_START",
        "JULD_DESCENT_END",
        "JULD_DEEP_DESCENT_END",
        "JULD_PARK_START",
        "JULD_PARK_END",
        "JULD_DEEP_ASCENT_START",
        "JULD_ASCENT_START",
        "JULD_ASCENT_END",
        "JULD_TRANSMISSION_START",
        "JULD_TRANSMISSION_END",
    ]:
        ds[name].attrs["units"] = "days since 2000-01-01 00:00:00"
        ds[name].attrs["calendar"] = "gregorian"
    ds.to_netcdf(path)


def test_rtraj_extracts_platform_time_fallback_and_parking_pressure(tmp_path: Path) -> None:
    rtraj_path = tmp_path / "1902267_Rtraj.nc"
    _write_rtraj(rtraj_path)

    config = {
        "input": {"rtraj_files": [str(rtraj_path)]},
        "output": {"path": str(tmp_path / "rtraj.zarr")},
        "processing": {
            "parking_depth": {
                "mode": "representative_park_pressure",
                "fallback_value": 1000.0,
            },
        },
    }

    trajectories = convert_rtraj_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["platform_code"].iloc[0] == 1902267
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2000-01-01T00:00:00"),
        pd.Timestamp("2000-01-12T00:00:00"),
        pd.Timestamp("2000-01-21T00:00:00"),
    ]
    assert np.allclose(trajectory["z"].to_numpy(dtype=float), [901.0, 902.0, 1000.0])
    assert trajectory["trajectory"].tolist() == [0, 0, 0]
    assert trajectory["obs"].tolist() == [0, 1, 2]


def test_rtraj_resampling_fills_z_without_interpolation(tmp_path: Path) -> None:
    rtraj_path = tmp_path / "1902267_Rtraj.nc"
    _write_rtraj(rtraj_path, pressure=[900.0, 1000.0])

    config = {
        "input": {"rtraj_files": [str(rtraj_path)]},
        "output": {"path": str(tmp_path / "rtraj_resampled.zarr")},
        "processing": {
            "parking_depth": {
                "mode": "representative_park_pressure",
                "fallback_value": 1000.0,
            },
            "resample": {
                "enabled": True,
                "frequency": "10d",
                "interpolate": "time",
            },
        },
    }

    trajectories = convert_rtraj_to_dataframe(config)

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["time"].tolist() == [
        pd.Timestamp("2000-01-01T00:00:00"),
        pd.Timestamp("2000-01-11T00:00:00"),
        pd.Timestamp("2000-01-21T00:00:00"),
    ]
    assert np.allclose(trajectory["z"].to_numpy(dtype=float), [900.0, 900.0, 1000.0])
    assert np.isclose(float(trajectory["lat"].iloc[1]), -59.81818181818182)


def test_rtraj_frequency_segmentation_splits_large_raw_juld_gap(tmp_path: Path) -> None:
    rtraj_path = tmp_path / "1902267_Rtraj.nc"
    _write_segmented_rtraj(rtraj_path, juld=[0.0, 10.0, 20.0, 60.0, 70.0])

    config = {
        "input": {"rtraj_files": [str(rtraj_path)]},
        "output": {"path": str(tmp_path / "rtraj_frequency.zarr")},
        "processing": {
            "parking_depth": {"mode": "representative_park_pressure"},
            "frequency": {
                "enabled": True,
                "source_max_gap_days": 29,
            },
        },
    }

    trajectories = convert_rtraj_to_dataframe(config)

    assert len(trajectories) == 2
    assert [len(trajectory) for trajectory in trajectories] == [3, 2]
    assert trajectories[0]["time"].tolist() == pd.to_datetime(["2000-01-01", "2000-01-11", "2000-01-21"]).tolist()
    assert trajectories[1]["time"].tolist() == pd.to_datetime(["2000-03-01", "2000-03-11"]).tolist()


def test_rtraj_near_surface_filter_discards_flagged_cycle_and_splits(tmp_path: Path) -> None:
    rtraj_path = tmp_path / "1902267_Rtraj.nc"
    _write_segmented_rtraj(rtraj_path, bad_cycle=3)

    config = {
        "input": {"rtraj_files": [str(rtraj_path)]},
        "output": {"path": str(tmp_path / "rtraj_near_surface.zarr")},
        "processing": {
            "parking_depth": {"mode": "representative_park_pressure"},
            "near_surface": {
                "enabled": True,
                "near_surface_max_hours": 18,
            },
        },
    }

    trajectories = convert_rtraj_to_dataframe(config)

    assert len(trajectories) == 2
    assert [len(trajectory) for trajectory in trajectories] == [2, 2]
    assert trajectories[0]["time"].tolist() == pd.to_datetime(["2000-01-01", "2000-01-11"]).tolist()
    assert trajectories[1]["time"].tolist() == pd.to_datetime(["2000-01-31", "2000-02-10"]).tolist()


def test_rtraj_region_selection_modes() -> None:
    outside_then_inside = pd.DataFrame(
        {
            "platform_code": [1, 1, 1],
            "time": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
            "lat": [-50.0, -60.0, -60.0],
            "lon": [-100.0, -70.0, -50.0],
            "z": [1000.0, 1000.0, 1000.0],
        }
    )
    starts_inside = outside_then_inside.copy()
    starts_inside.loc[0, ["lat", "lon"]] = [-60.0, -70.0]

    from_entry = _apply_region_selection(
        [outside_then_inside],
        config=RegionSelectionConfig(names_or_labels=("DP",), selection_mode="from_first_entry"),
    )
    full_if_enters = _apply_region_selection(
        [outside_then_inside],
        config=RegionSelectionConfig(names_or_labels=("DP",), selection_mode="full_if_enters"),
    )
    initial_inside_drop = _apply_region_selection(
        [outside_then_inside],
        config=RegionSelectionConfig(names_or_labels=("DP",), selection_mode="initial_inside"),
    )
    initial_inside_keep = _apply_region_selection(
        [starts_inside],
        config=RegionSelectionConfig(names_or_labels=("DP",), selection_mode="initial_inside"),
    )

    assert len(from_entry) == 1
    assert from_entry[0]["time"].tolist() == pd.to_datetime(["2000-01-02", "2000-01-03"]).tolist()
    assert len(full_if_enters) == 1
    assert len(full_if_enters[0]) == 3
    assert initial_inside_drop == []
    assert len(initial_inside_keep) == 1
    assert len(initial_inside_keep[0]) == 3


def test_depth_bin_splitting_keeps_contiguous_runs_separate() -> None:
    trajectory = pd.DataFrame(
        {
            "platform_code": [1902267] * 5,
            "time": pd.date_range("2000-01-01", periods=5, freq="10d"),
            "lat": [-60.0, -60.5, -61.0, -61.5, -62.0],
            "lon": [-70.0, -69.5, -69.0, -68.5, -68.0],
            "z": [1000.0, 1000.0, 1800.0, 1800.0, 1000.0],
        }
    )
    config = DepthBinConfig(
        enabled=True,
        bins=(
            DepthBin(label="z0750_1250", min_value=750.0, max_value=1250.0),
            DepthBin(label="z1500_2500", min_value=1500.0, max_value=2500.0),
        ),
    )

    segments = _split_trajectory_by_depth_bin(trajectory, config=config)

    assert len(segments) == 3
    assert [len(segment) for segment in segments] == [2, 2, 1]
    assert [segment["depth_bin"].iloc[0] for segment in segments] == [
        "z0750_1250",
        "z1500_2500",
        "z0750_1250",
    ]
    assert all(segment["platform_code"].iloc[0] == 1902267 for segment in segments)


def test_depth_bins_are_applied_before_dataframe_normalization(tmp_path: Path) -> None:
    rtraj_path = tmp_path / "1902267_Rtraj.nc"
    _write_depth_switch_rtraj(rtraj_path)

    config = {
        "input": {"rtraj_files": [str(rtraj_path)]},
        "output": {"path": str(tmp_path / "depth_bins.zarr")},
        "processing": {
            "parking_depth": {"mode": "representative_park_pressure"},
            "depth_bins": {
                "enabled": True,
                "bins": [
                    {"label": "z0750_1250", "min": 750.0, "max": 1250.0},
                    {"label": "z1500_2500", "min": 1500.0, "max": 2500.0},
                ],
            },
        },
    }

    trajectories = convert_rtraj_to_dataframe(config)

    assert len(trajectories) == 3
    assert [trajectory["trajectory"].iloc[0] for trajectory in trajectories] == [0, 1, 2]
    assert [trajectory["platform_code"].iloc[0] for trajectory in trajectories] == [1902267, 1902267, 1902267]
    assert [trajectory["depth_bin"].iloc[0] for trajectory in trajectories] == [
        "z0750_1250",
        "z1500_2500",
        "z0750_1250",
    ]
    assert [len(trajectory) for trajectory in trajectories] == [2, 2, 1]


def test_depth_bin_output_path_appends_label_before_zarr_suffix() -> None:
    assert _depth_bin_output_path(Path("out/DP_rtraj.zarr"), "z0750_1250") == Path("out/DP_rtraj_z0750_1250.zarr")


def test_depth_histogram_counts_use_100m_bins_and_overflow() -> None:
    trajectory = pd.DataFrame(
        {
            "z": [-10.0, 0.0, 49.9, 50.0, 149.9, 2450.0, 3000.0, np.nan],
        }
    )

    counts = _depth_histogram_counts([trajectory])

    assert counts[0] == 3
    assert counts[1] == 2
    assert counts[-1] == 2
    assert int(counts.sum()) == 7


def test_convert_rtraj_to_zarr_creates_parcels_compatible_dataset(tmp_path: Path) -> None:
    rtraj_path = tmp_path / "1902267_Rtraj.nc"
    _write_rtraj(rtraj_path)

    output_path = tmp_path / "rtraj_output.zarr"
    config = {
        "input": {"rtraj_files": [str(rtraj_path)]},
        "output": {"path": str(output_path)},
        "processing": {
            "parking_depth": {
                "mode": "representative_park_pressure",
                "fallback_value": 1000.0,
            },
        },
    }

    convert_rtraj_to_zarr(config)

    ds = xr.open_zarr(output_path)

    assert ds.dims["trajectory"] == 1
    assert ds.dims["obs"] == 3
    assert ds["platform_code"].dims == ("trajectory",)
    assert ds["platform_code"].values.tolist() == [1902267]
    assert np.allclose(ds["z"].values[0, :3], [901.0, 902.0, 1000.0])
    assert "REPRESENTATIVE_PARK_PRESSURE" in str(ds.attrs["z_source"])
