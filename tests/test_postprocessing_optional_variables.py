from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels.postprocessing.config.models import (
    DatasetConfig,
    ExportsConfig,
    OutputConfig,
    PostprocessConfig,
)
from kinematicparcels.postprocessing.core import build_particle_summary
from kinematicparcels.postprocessing.io import load_trajectory_table
from kinematicparcels.postprocessing.workflows.base_products import (
    OBSERVATION_VARIABLE_METADATA_CONTEXT_KEY,
    get_particle_summary,
    get_trajectory_table,
)
from kinematicparcels.postprocessing.workflows.run_summary import run_summary


def _write_optional_variable_zarr(tmp_path: Path) -> Path:
    path = tmp_path / "trajectories.zarr"
    ds = xr.Dataset(
        data_vars={
            "time": (
                ("trajectory", "obs"),
                np.array(
                    [
                        [
                            "2026-01-01T00:00:00",
                            "2026-01-02T00:00:00",
                            "2026-01-03T00:00:00",
                            "2026-01-04T00:00:00",
                        ],
                        [
                            "2026-02-01T00:00:00",
                            "2026-02-02T00:00:00",
                            "2026-02-03T00:00:00",
                            "2026-02-04T00:00:00",
                        ],
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            "lon": (
                ("trajectory", "obs"),
                [[10.0, 10.1, 10.2, 10.3], [20.0, 20.1, 20.2, 20.3]],
            ),
            "lat": (
                ("trajectory", "obs"),
                [[-50.0, -50.1, -50.2, -50.3], [-55.0, -55.1, -55.2, -55.3]],
            ),
            "z": (
                ("trajectory", "obs"),
                [[1000.0, 1000.0, 1000.0, 1000.0], [900.0, 900.0, 900.0, 900.0]],
            ),
            "temp": (
                ("trajectory", "obs"),
                [[np.nan, 2.0, 4.0, np.nan], [5.0, 6.0, 7.0, 8.0]],
            ),
            "psal": (
                ("trajectory", "obs"),
                [[34.0, np.inf, 35.0, 36.0], [33.0, 34.0, 35.0, 36.0]],
            ),
            "doxy": (
                ("trajectory", "obs"),
                [[200.0, 201.0, 202.0, 203.0], [210.0, 211.0, 212.0, 213.0]],
            ),
            "sample_status": (
                ("trajectory", "obs"),
                [["missing", "ok", "ok", "missing"], ["ok", "ok", "ok", "ok"]],
            ),
            "platform_code": (("trajectory",), [1900852, 1900853]),
            "depth_bin": (("trajectory",), ["z0850_1150", "z0850_1150"]),
            "depth_bin_interval": (("trajectory",), ["[850, 1150)", "[850, 1150)"]),
            "unrelated": (("component",), [1.0, 2.0]),
        },
        coords={
            "trajectory": [0, 1],
            "obs": [0, 1, 2, 3],
            "component": [0, 1],
        },
    )
    ds["temp"].attrs.update(
        units="degree_Celsius",
        long_name="Sea temperature",
        standard_name="sea_water_temperature",
    )
    ds["psal"].attrs.update(units="psu", long_name="Practical salinity")
    ds.to_zarr(path, mode="w")
    return path


def _config(input_path: Path, output_dir: Path) -> PostprocessConfig:
    return PostprocessConfig(
        dataset=DatasetConfig(input_path=str(input_path)),
        output=OutputConfig(output_dir=str(output_dir)),
        exports=ExportsConfig(
            save_trajectory_table=True,
            save_particle_summary=True,
        ),
    )


def test_base_product_discovers_compatible_optional_zarr_variables(tmp_path: Path) -> None:
    input_path = _write_optional_variable_zarr(tmp_path)
    cfg = _config(input_path, tmp_path / "out")

    context: dict = {}
    trajectory_table = get_trajectory_table(cfg, context)

    assert {
        "temp",
        "psal",
        "doxy",
        "sample_status",
        "platform_code",
        "depth_bin",
        "depth_bin_interval",
    }.issubset(trajectory_table.columns)
    assert "unrelated" not in trajectory_table.columns
    np.testing.assert_allclose(
        trajectory_table.loc[trajectory_table["trajectory"] == 0, "doxy"],
        [200.0, 201.0, 202.0, 203.0],
    )
    assert context[OBSERVATION_VARIABLE_METADATA_CONTEXT_KEY]["temp"] == {
        "units": "degree_Celsius",
        "long_name": "Sea temperature",
        "standard_name": "sea_water_temperature",
    }

    # The generic loader still includes extra fields only when explicitly asked.
    low_level_table = load_trajectory_table(input_path)
    assert "temp" not in low_level_table.columns
    assert "platform_code" not in low_level_table.columns


def test_particle_summary_aggregates_numeric_optional_series(tmp_path: Path) -> None:
    input_path = _write_optional_variable_zarr(tmp_path)
    cfg = _config(input_path, tmp_path / "out")
    trajectory_table = get_trajectory_table(cfg, {})

    summary = build_particle_summary(trajectory_table)
    first = summary.loc[summary["trajectory"] == 0].iloc[0]

    assert first["platform_code"] == 1900852
    assert first["depth_bin"] == "z0850_1150"
    assert first["depth_bin_interval"] == "[850, 1150)"
    assert np.isnan(first["temp0"])
    assert np.isnan(first["tempf"])
    assert first["temp_min"] == 2.0
    assert first["temp_max"] == 4.0
    assert first["temp_mean"] == 3.0
    assert first["psal0"] == 34.0
    assert first["psalf"] == 36.0
    assert first["psal_mean"] == 35.0
    assert first["doxy0"] == 200.0
    assert first["doxyf"] == 203.0
    assert first["doxy_mean"] == 201.5
    assert {
        "sample_status0",
        "sample_statusf",
        "sample_status_min",
        "sample_status_max",
        "sample_status_mean",
    }.isdisjoint(summary.columns)


def test_summary_rebuilds_and_replaces_stale_cached_tables(tmp_path: Path) -> None:
    input_path = _write_optional_variable_zarr(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = _config(input_path, output_dir)

    stale_trajectory = load_trajectory_table(input_path)
    stale_trajectory.to_parquet(output_dir / "trajectory_table.parquet", index=False)
    build_particle_summary(stale_trajectory).to_parquet(
        output_dir / "particle_summary.parquet",
        index=False,
    )

    context: dict = {}
    rebuilt_trajectory = get_trajectory_table(cfg, context)
    rebuilt_summary = get_particle_summary(cfg, context)

    assert "temp" in rebuilt_trajectory.columns
    assert "temp_mean" in rebuilt_summary.columns
    assert "platform_code" in rebuilt_summary.columns

    run_summary(cfg, {})

    saved_trajectory = pd.read_parquet(output_dir / "trajectory_table.parquet")
    saved_summary = pd.read_parquet(output_dir / "particle_summary.parquet")
    assert {"temp", "psal", "doxy", "sample_status"}.issubset(saved_trajectory.columns)
    assert {
        "temp0",
        "tempf",
        "temp_min",
        "temp_max",
        "temp_mean",
        "psal_mean",
        "doxy_mean",
        "platform_code",
        "depth_bin",
        "depth_bin_interval",
    }.issubset(saved_summary.columns)
