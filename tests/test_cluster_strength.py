from __future__ import annotations

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.cluster_strength import compute_cluster_strength
from kinematicparcels.postprocessing.config.loader import load_postprocess_config
from kinematicparcels.postprocessing.core.distances import haversine_km
from kinematicparcels.postprocessing.core.gridding import RegularGrid
from kinematicparcels.postprocessing.workflows.run_cluster_strength import _resolve_snapshot_indices


def test_cluster_strength_single_particle_on_grid_point_contributes_one() -> None:
    grid = RegularGrid(
        lon_min=-0.5,
        lon_max=0.5,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
    )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01")],
            "lon": [0.0],
            "lat": [0.0],
        }
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)

    assert ds["cluster_strength"].shape == (1, 1, 1)
    assert ds["cluster_strength"].values[0, 0, 0] == pytest.approx(1.0)


def test_cluster_strength_two_particles_sum_gaussian_contributions() -> None:
    grid = RegularGrid(
        lon_min=-0.5,
        lon_max=0.5,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
    )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            "lon": [0.0, 0.1],
            "lat": [0.0, 0.0],
        }
    )
    scale_km = 20.0
    distance = haversine_km(0.0, 0.0, 0.1, 0.0)
    expected = 1.0 + np.exp(-((distance / scale_km) ** 2))

    ds = compute_cluster_strength(df, grid=grid, scale_km=scale_km, mask=False)

    assert ds["cluster_strength"].values[0, 0, 0] == pytest.approx(float(expected))


def test_cluster_strength_mask_true_leaves_never_explored_cells_nan() -> None:
    grid = RegularGrid(
        lon_min=-0.5,
        lon_max=1.5,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
    )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01")],
            "lon": [0.0],
            "lat": [0.0],
        }
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=1000.0, mask=True)

    values = ds["cluster_strength"].values
    assert values[0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(values[0, 0, 1])


def test_cluster_strength_euclidean_distance_runs() -> None:
    grid = RegularGrid(
        lon_min=-0.5,
        lon_max=0.5,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
    )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01")],
            "lon": [0.0],
            "lat": [0.0],
        }
    )

    ds = compute_cluster_strength(
        df,
        grid=grid,
        scale_km=10.0,
        distance="euclidean",
        mask=False,
    )

    assert ds.attrs["distance"] == "euclidean"
    assert ds["cluster_strength"].values[0, 0, 0] == pytest.approx(1.0)


def test_cluster_strength_warns_when_scipy_is_absent(monkeypatch) -> None:
    from kinematicparcels.postprocessing.analyses import cluster_strength as cluster_module

    monkeypatch.setattr(cluster_module, "cKDTree", None)
    grid = RegularGrid(
        lon_min=-0.5,
        lon_max=0.5,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
    )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01")],
            "lon": [0.0],
            "lat": [0.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="scipy.spatial.cKDTree is unavailable"):
        compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)


def test_cluster_strength_cutoff_factor_excludes_far_particles() -> None:
    grid = RegularGrid(
        lon_min=-0.5,
        lon_max=0.5,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
    )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            "lon": [0.0, 0.1],
            "lat": [0.0, 0.0],
        }
    )

    ds = compute_cluster_strength(
        df,
        grid=grid,
        scale_km=1.0,
        cutoff_factor=1.0,
        mask=False,
    )

    assert ds.attrs["cutoff_factor"] == 1.0
    assert ds["cluster_strength"].values[0, 0, 0] == pytest.approx(1.0)


def test_load_postprocess_config_rejects_missing_cluster_strength_scale(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_cluster_missing_scale.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - cluster_strength
            cluster_strength:
              distance: haversine
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cluster_strength.scale_km"):
        load_postprocess_config(cfg_path)


def test_load_postprocess_config_rejects_non_positive_cluster_strength_scale(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_cluster_non_positive_scale.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - cluster_strength
            cluster_strength:
              scale_km: 0
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cluster_strength.scale_km must be positive"):
        load_postprocess_config(cfg_path)


def test_load_postprocess_config_parses_cluster_strength_cutoff_and_euclidean(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_cluster.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - cluster_strength
            cluster_strength:
              scale_km: 5
              distance: euclidean
              cutoff_factor: 3
              animation_fps: 5
              animation_every_n: 2
              min_mask_value: 0.05
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.cluster_strength is not None
    assert cfg.cluster_strength.distance == "euclidean"
    assert cfg.cluster_strength.cutoff_factor == 3.0
    assert cfg.cluster_strength.vmin is None
    assert cfg.cluster_strength.min_mask_value == 0.05
    assert cfg.cluster_strength.animation_fps == 5
    assert cfg.cluster_strength.animation_every_n == 2


def test_load_postprocess_config_rejects_invalid_cluster_strength_animation_cadence(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_cluster_bad_animation.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - cluster_strength
            cluster_strength:
              scale_km: 5
              animation_fps: 0
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cluster_strength.animation_fps"):
        load_postprocess_config(cfg_path)


def test_load_postprocess_config_rejects_uppercase_cluster_strength_distance(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_cluster_bad_distance.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - cluster_strength
            cluster_strength:
              scale_km: 5
              distance: Haversine
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be lowercase"):
        load_postprocess_config(cfg_path)


def test_resolve_snapshot_indices_supports_negative_indices() -> None:
    assert _resolve_snapshot_indices(-1, n_times=4) == (3,)
    assert _resolve_snapshot_indices((0, -1, 2), n_times=4) == (0, 3, 2)


def test_resolve_snapshot_indices_rejects_out_of_range_indices() -> None:
    with pytest.raises(IndexError, match="out of range"):
        _resolve_snapshot_indices(-5, n_times=4)
