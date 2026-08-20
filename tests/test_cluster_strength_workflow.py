from __future__ import annotations

import importlib
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kinematicparcels.postprocessing.config.loader import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
    ClusterStrengthAnimationConfig,
    ClusterStrengthSnapshotsConfig,
)
from kinematicparcels.postprocessing.core.gridding import RegularGrid


workflow = importlib.import_module(
    "kinematicparcels.postprocessing.workflows.run_cluster_strength"
)


@pytest.mark.parametrize(
    ("mode", "dimension", "expected_name"),
    [
        ("time", "time", "cluster_strength_time.nc"),
        ("age", "age_days", "cluster_strength_age.nc"),
    ],
)
def test_run_cluster_strength_uses_mode_specific_netcdf_name(
    tmp_path, monkeypatch, mode, dimension, expected_name
) -> None:
    cfg_path = tmp_path / f"cluster_{mode}.yml"
    cfg_path.write_text(
        dedent(
            f"""
            dataset:
              input_path: ./dummy.zarr
            output:
              output_dir: {tmp_path.as_posix()}
            analysis:
              types: [cluster_strength]
            cluster_strength:
              scale_km: 5
              mode: {mode}
            """
        ),
        encoding="utf-8",
    )
    cfg = load_postprocess_config(cfg_path)
    coordinate = (
        np.array([np.datetime64("2026-01-01")]) if mode == "time" else np.array([0.0])
    )
    ds = xr.Dataset(
        {"cluster_strength": ((dimension, "lat", "lon"), np.ones((1, 1, 1)))},
        coords={dimension: coordinate, "lat": [0.0], "lon": [0.0]},
    )
    saved_paths: list[Path] = []
    monkeypatch.setattr(
        workflow, "get_trajectory_table", lambda cfg, context: pd.DataFrame()
    )
    monkeypatch.setattr(
        workflow, "compute_cluster_strength", lambda *args, **kwargs: ds
    )
    monkeypatch.setattr(
        workflow,
        "save_dataset_netcdf",
        lambda dataset, path: saved_paths.append(Path(path)),
    )

    workflow.run_cluster_strength(
        cfg,
        {"grid": RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)},
    )

    assert saved_paths == [tmp_path / expected_name]


def test_grouped_age_animation_uses_numeric_age_axis(tmp_path, monkeypatch) -> None:
    ds = xr.Dataset(
        {"cluster_strength": (("age_days", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"age_days": [-1.0, 0.0], "lat": [0.0], "lon": [0.0]},
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        workflow,
        "animate_density",
        lambda dataset, **kwargs: calls.append(kwargs),
    )

    workflow._save_grouped_animation(
        ds,
        mode="age",
        animation_cfg=ClusterStrengthAnimationConfig(enabled=True),
        outdir=tmp_path,
        projection="PlateCarree",
    )

    assert calls[0]["outpath"] == tmp_path / "cluster_strength_age.gif"
    assert calls[0]["frame_dim"] == "age_days"
    assert calls[0]["frame_label"] == "age"
    assert calls[0]["frame_units"] == "days"


def test_time_snapshot_matches_nearest_coordinate_and_names_file(
    tmp_path, monkeypatch
) -> None:
    ds = xr.Dataset(
        {"cluster_strength": (("time", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={
            "time": np.array(
                ["2026-01-01T00:00", "2026-01-01T02:00"], dtype="datetime64[m]"
            ),
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        workflow,
        "plot_grid_map",
        lambda dataset, **kwargs: calls.append(kwargs),
    )

    workflow._save_time_snapshots(
        ds,
        snapshot_cfg=ClusterStrengthSnapshotsConfig(
            enabled=True,
            fixed_times="2026-01-01 01:30",
            time_tolerance_hours=1.0,
        ),
        outdir=tmp_path,
        projection="PlateCarree",
        title_fontsize=None,
        colorbar_fontsize=None,
        colorbar_tick_fontsize=None,
        axis_tick_fontsize=None,
    )

    assert calls[0]["outpath"] == tmp_path / "cluster_strength_time_20260101T020000.png"


def test_time_snapshot_requires_exact_coordinate_without_tolerance() -> None:
    with pytest.raises(ValueError, match="not present in time"):
        workflow._match_time_value(
            np.array([np.datetime64("2026-01-01T00:00")]),
            requested_time="2026-01-01 00:30",
            tolerance_hours=None,
        )


def test_age_snapshot_coordinate_matching_supports_exact_and_tolerance() -> None:
    available = np.array([-2.0, -1.0, 0.0])

    assert (
        workflow._match_age_value(
            available,
            requested_age_days=-1.0,
            tolerance_days=None,
        )
        == -1.0
    )
    assert (
        workflow._match_age_value(
            available,
            requested_age_days=-1.1,
            tolerance_days=0.2,
        )
        == -1.0
    )
    with pytest.raises(ValueError, match="not present in age_days"):
        workflow._match_age_value(
            available,
            requested_age_days=-1.1,
            tolerance_days=None,
        )
