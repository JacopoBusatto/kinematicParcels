from __future__ import annotations

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.cluster_strength import (
    compute_cluster_strength,
)
from kinematicparcels.postprocessing.config.loader import load_postprocess_config
from kinematicparcels.postprocessing.core.distances import haversine_km
from kinematicparcels.postprocessing.core.gridding import RegularGrid


def _trajectory_df(
    *,
    times: list[pd.Timestamp],
    lons: list[float],
    lats: list[float],
    trajectory: str = "p0",
    group_member: int | None = None,
) -> pd.DataFrame:
    data: dict[str, object] = {
        "trajectory": [trajectory] * len(times),
        "obs": list(range(len(times))),
        "time": times,
        "lon": lons,
        "lat": lats,
    }
    if group_member is not None:
        data["group_member"] = [group_member] * len(times)
    return pd.DataFrame(data)


def test_cluster_strength_single_particle_on_grid_point_contributes_one() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[pd.Timestamp("2026-01-01")],
        lons=[0.0],
        lats=[0.0],
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)

    assert ds["cluster_strength"].dims == ("release_time", "age_days", "lat", "lon")
    assert ds["cluster_strength"].shape == (1, 1, 1, 1)
    assert ds["age_days"].values.tolist() == [0.0]
    assert "units" not in ds["age_days"].attrs
    assert ds["age_days"].attrs["unit_label"] == "days"
    assert ds["cluster_strength"].values[0, 0, 0, 0] == pytest.approx(1.0)


def test_cluster_strength_two_particles_sum_gaussian_contributions() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = pd.concat(
        [
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01")],
                lons=[0.0],
                lats=[0.0],
                trajectory="p0",
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01")],
                lons=[0.1],
                lats=[0.0],
                trajectory="p1",
            ),
        ],
        ignore_index=True,
    )
    scale_km = 20.0
    distance = haversine_km(0.0, 0.0, 0.1, 0.0)
    expected = 1.0 + np.exp(-((distance / scale_km) ** 2))

    ds = compute_cluster_strength(df, grid=grid, scale_km=scale_km, mask=False)

    assert ds["cluster_strength"].values[0, 0, 0, 0] == pytest.approx(float(expected))


def test_cluster_strength_mask_true_leaves_never_explored_cells_nan() -> None:
    grid = RegularGrid(-0.5, 1.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[pd.Timestamp("2026-01-01")],
        lons=[0.0],
        lats=[0.0],
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=1000.0, mask=True)

    values = ds["cluster_strength"].values
    assert values[0, 0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(values[0, 0, 0, 1])


def test_cluster_strength_euclidean_distance_runs() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[pd.Timestamp("2026-01-01")],
        lons=[0.0],
        lats=[0.0],
    )

    ds = compute_cluster_strength(
        df,
        grid=grid,
        scale_km=10.0,
        distance="euclidean",
        mask=False,
    )

    assert ds.attrs["distance"] == "euclidean"
    assert ds["cluster_strength"].values[0, 0, 0, 0] == pytest.approx(1.0)


def test_cluster_strength_warns_when_scipy_is_absent(monkeypatch) -> None:
    from kinematicparcels.postprocessing.analyses import (
        cluster_strength as cluster_module,
    )

    monkeypatch.setattr(cluster_module, "cKDTree", None)
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[pd.Timestamp("2026-01-01")],
        lons=[0.0],
        lats=[0.0],
    )

    with pytest.warns(RuntimeWarning, match="scipy.spatial.cKDTree is unavailable"):
        compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)


def test_cluster_strength_cutoff_factor_excludes_far_particles() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = pd.concat(
        [
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01")],
                lons=[0.0],
                lats=[0.0],
                trajectory="p0",
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01")],
                lons=[0.1],
                lats=[0.0],
                trajectory="p1",
            ),
        ],
        ignore_index=True,
    )

    ds = compute_cluster_strength(
        df,
        grid=grid,
        scale_km=1.0,
        cutoff_factor=1.0,
        mask=False,
    )

    assert ds.attrs["cutoff_factor"] == 1.0
    assert ds["cluster_strength"].values[0, 0, 0, 0] == pytest.approx(1.0)


def test_cluster_strength_separates_release_time_and_signed_age() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = pd.concat(
        [
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
                lons=[0.0, 0.0],
                lats=[0.0, 0.0],
                trajectory="p0",
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
                lons=[0.0, 0.0],
                lats=[0.0, 0.0],
                trajectory="p1",
            ),
        ],
        ignore_index=True,
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)

    assert ds.sizes["release_time"] == 2
    assert ds["age_days"].values.tolist() == [0.0, 1.0]
    assert ds.attrs["simulation_direction"] == "forward"
    assert ds["cluster_strength"].sel(age_days=1.0).values[:, 0, 0] == pytest.approx(
        [1.0, 1.0]
    )


def test_cluster_strength_backward_age_is_negative() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[
            pd.Timestamp("2026-01-03"),
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-01-01"),
        ],
        lons=[0.0, 0.0, 0.0],
        lats=[0.0, 0.0, 0.0],
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)

    assert ds.attrs["simulation_direction"] == "backward"
    assert ds["age_days"].values.tolist() == [-2.0, -1.0, 0.0]
    assert ds["release_time"].values[0] == np.datetime64(
        "2026-01-03T00:00:00.000000000"
    )


def test_cluster_strength_time_mode_combines_release_cohorts() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = pd.concat(
        [
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
                lons=[0.0, 0.0],
                lats=[0.0, 0.0],
                trajectory="p0",
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-02")],
                lons=[0.0],
                lats=[0.0],
                trajectory="p1",
            ),
        ],
        ignore_index=True,
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False, mode="time")

    assert ds["cluster_strength"].dims == ("time", "lat", "lon")
    assert ds.attrs["aggregation_mode"] == "time"
    assert ds["cluster_strength"].sel(time="2026-01-02").item() == pytest.approx(2.0)
    assert ds["particle_count"].values.tolist() == [1, 2]
    assert ds["release_count"].values.tolist() == [1, 2]


def test_cluster_strength_age_mode_combines_only_exact_signed_ages() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = pd.concat(
        [
            _trajectory_df(
                times=[pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-02")],
                lons=[0.0, 0.0],
                lats=[0.0, 0.0],
                trajectory="p0",
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-04")],
                lons=[0.0, 0.0],
                lats=[0.0, 0.0],
                trajectory="p1",
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-06 12:00")],
                lons=[0.0, 0.0],
                lats=[0.0, 0.0],
                trajectory="p2",
            ),
        ],
        ignore_index=True,
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False, mode="age")

    assert ds["cluster_strength"].dims == ("age_days", "lat", "lon")
    assert ds.attrs["simulation_direction"] == "backward"
    assert ds["age_days"].values.tolist() == [-1.0, -0.5, 0.0]
    assert ds["cluster_strength"].sel(age_days=-1.0).item() == pytest.approx(2.0)
    assert ds["particle_count"].values.tolist() == [2, 1, 3]
    assert ds["release_count"].values.tolist() == [2, 1, 3]


def test_cluster_strength_counts_identical_particle_time_duplicate_once() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
        lons=[0.0, 0.0],
        lats=[0.0, 0.0],
    )

    ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False, mode="time")

    assert ds["cluster_strength"].item() == pytest.approx(1.0)
    assert ds["particle_count"].item() == 1


def test_cluster_strength_rejects_conflicting_particle_time_duplicate() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = _trajectory_df(
        times=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
        lons=[0.0, 0.1],
        lats=[0.0, 0.0],
    )

    with pytest.raises(ValueError, match="conflicting positions"):
        compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False, mode="time")


def test_cluster_strength_filters_group_members_by_default() -> None:
    grid = RegularGrid(-0.5, 0.5, -0.5, 0.5, 1.0, 1.0)
    df = pd.concat(
        [
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01")],
                lons=[0.0],
                lats=[0.0],
                trajectory="g0",
                group_member=1,
            ),
            _trajectory_df(
                times=[pd.Timestamp("2026-01-01")],
                lons=[0.0],
                lats=[0.0],
                trajectory="g0",
                group_member=2,
            ),
        ],
        ignore_index=True,
    )

    default_ds = compute_cluster_strength(df, grid=grid, scale_km=10.0, mask=False)
    all_ds = compute_cluster_strength(
        df,
        grid=grid,
        scale_km=10.0,
        mask=False,
        max_group_member=None,
    )

    assert default_ds["cluster_strength"].values[0, 0, 0, 0] == pytest.approx(1.0)
    assert all_ds["cluster_strength"].values[0, 0, 0, 0] == pytest.approx(2.0)


def test_load_postprocess_config_rejects_missing_cluster_strength_scale(
    tmp_path,
) -> None:
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


def test_load_postprocess_config_rejects_non_positive_cluster_strength_scale(
    tmp_path,
) -> None:
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


def test_load_postprocess_config_parses_cluster_strength_cutoff_and_euclidean(
    tmp_path,
) -> None:
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
              max_group_member: null
              animation:
                enabled: true
                fps: 5
                every_n: 2
                min_mask_value: 0.05
                fixed_age_days: [1, 2]
              snapshots:
                enabled: true
                fixed_age_days: 1
                vmin: 0
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.cluster_strength is not None
    assert cfg.cluster_strength.distance == "euclidean"
    assert cfg.cluster_strength.cutoff_factor == 3.0
    assert cfg.cluster_strength.max_group_member is None
    assert cfg.cluster_strength.animation.enabled is True
    assert cfg.cluster_strength.animation.min_mask_value == 0.05
    assert cfg.cluster_strength.animation.fixed_age_days == pytest.approx((1.0, 2.0))
    assert cfg.cluster_strength.animation.fps == 5
    assert cfg.cluster_strength.animation.every_n == 2
    assert cfg.cluster_strength.snapshots.enabled is True
    assert cfg.cluster_strength.snapshots.fixed_age_days == 1.0
    assert cfg.cluster_strength.snapshots.vmin == 0.0


def test_load_postprocess_config_rejects_invalid_cluster_strength_animation_cadence(
    tmp_path,
) -> None:
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
              animation:
                fps: 0
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cluster_strength.animation.fps"):
        load_postprocess_config(cfg_path)


def test_load_postprocess_config_rejects_uppercase_cluster_strength_distance(
    tmp_path,
) -> None:
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


def test_load_postprocess_config_rejects_invalid_cluster_strength_mode(
    tmp_path,
) -> None:
    cfg_path = tmp_path / "postprocess_cluster_bad_mode.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types: [cluster_strength]
            cluster_strength:
              scale_km: 5
              mode: calendar
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cluster_strength.mode"):
        load_postprocess_config(cfg_path)


def test_load_postprocess_config_parses_time_mode_snapshot_coordinates(
    tmp_path,
) -> None:
    cfg_path = tmp_path / "postprocess_cluster_time.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types: [cluster_strength]
            cluster_strength:
              scale_km: 5
              mode: time
              snapshots:
                enabled: true
                fixed_times:
                  - "2026-01-01 00:00"
                  - "2026-01-02 12:00"
                time_tolerance_hours: 1
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.cluster_strength is not None
    assert cfg.cluster_strength.mode == "time"
    assert cfg.cluster_strength.snapshots.fixed_times == (
        "2026-01-01 00:00",
        "2026-01-02 12:00",
    )
    assert cfg.cluster_strength.snapshots.time_tolerance_hours == 1.0


def test_load_postprocess_config_defaults_cluster_strength_mode_to_release(
    tmp_path,
) -> None:
    cfg_path = tmp_path / "postprocess_cluster_release.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types: [cluster_strength]
            cluster_strength:
              scale_km: 5
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.cluster_strength is not None
    assert cfg.cluster_strength.mode == "release"


def test_load_postprocess_config_rejects_age_selector_in_time_mode(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_cluster_bad_selector.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types: [cluster_strength]
            cluster_strength:
              scale_km: 5
              mode: time
              snapshots:
                enabled: true
                fixed_age_days: 1
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed_age_days is not valid in time mode"):
        load_postprocess_config(cfg_path)
