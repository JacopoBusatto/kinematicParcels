from __future__ import annotations

import math
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
  AnalysisConfig,
  DatasetConfig,
  ExponentMapsConfig,
  ExponentMapsFSLEConfig,
  ExponentMapsFTLEConfig,
    GridConfig,
  OutputConfig,
  PlottingConfig,
  PostprocessConfig,
  ReleaseConfig,
)
from kinematicparcels.postprocessing.analyses.exponent_maps import compute_exponent_maps
from kinematicparcels.postprocessing.core import build_particle_summary
from kinematicparcels.postprocessing.workflows.run_exponent_maps import run_exponent_maps


EARTH_RADIUS_KM = 6371.0


def _lat_offset_for_distance_km(distance_km: float) -> float:
    return math.degrees(distance_km / EARTH_RADIUS_KM)


def _build_group_rows(
    *,
    group_id: int,
    center_lon: float,
    center_lat: float,
    release_time: str,
    member_distances_km: dict[int, list[float]],
    day_offsets: list[int],
) -> list[dict]:
    rows: list[dict] = []
    base_time = pd.Timestamp(release_time)
    group_size = 1 + len(member_distances_km)
    for obs, day_offset in enumerate(day_offsets):
        current_time = base_time + pd.Timedelta(days=int(day_offset))
        rows.append(
            {
                "trajectory": f"{group_id}_m1",
                "group_id": group_id,
                "group_member": 1,
                "group_size": group_size,
                "obs": obs,
                "time": current_time,
                "lon": center_lon,
                "lat": center_lat,
            }
        )
        for member, distances in member_distances_km.items():
            rows.append(
                {
                    "trajectory": f"{group_id}_m{member}",
                    "group_id": group_id,
                    "group_member": member,
                    "group_size": group_size,
                    "obs": obs,
                    "time": current_time,
                    "lon": center_lon,
                    "lat": center_lat + _lat_offset_for_distance_km(distances[obs]),
                }
            )
    return rows


def _build_regular_release_df(*, release_times: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    group_id = 1
    for release_time in release_times:
        for lat0 in (0.0, 1.0):
            for lon0 in (0.0, 1.0):
                rows.extend(
                    _build_group_rows(
                        group_id=group_id,
                        center_lon=lon0,
                        center_lat=lat0,
                        release_time=release_time,
                        member_distances_km={2: [1.0, 2.0, 4.0]},
                        day_offsets=[0, 1, 2],
                    )
                )
                group_id += 1
    return pd.DataFrame(rows)


def test_load_postprocess_config_parses_exponent_maps_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_exponent_maps.yml"
    cfg_path.write_text(
                """dataset:
    input_path: ./dummy.zarr
analysis:
    types:
        - exponent_maps
exponent_maps:
    distance: meridional
    infer_grid_from_start: false
    fsle:
        enable: true
        scale: [2.0, 5.0]
        mask_zeros: true
        plot:
            enable: true
            average_on_time: false
    ftle:
        enable: true
        scale: [3.0]
        sampling_mode: max_within_window
        mask_short_windows: false
        mask_zeros: true
        plot:
            enable: false
""",
                encoding="utf-8",
        )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("exponent_maps",)
    assert cfg.exponent_maps is not None
    assert cfg.exponent_maps.distance == "meridional"
    assert cfg.exponent_maps.infer_grid_from_start is False
    assert cfg.exponent_maps.fsle.enabled is True
    assert cfg.exponent_maps.fsle.scales_km == pytest.approx((2.0, 5.0))
    assert cfg.exponent_maps.fsle.mask_zeros is True
    assert cfg.exponent_maps.fsle.plot.average_on_time is False
    assert cfg.exponent_maps.ftle.enabled is True
    assert cfg.exponent_maps.ftle.scales_days == pytest.approx((3.0,))
    assert cfg.exponent_maps.ftle.sampling_mode == "max_within_window"
    assert cfg.exponent_maps.ftle.mask_short_windows is False
    assert cfg.exponent_maps.ftle.mask_zeros is True


def test_load_postprocess_config_rejects_disabled_exponent_maps(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_exponent_maps_invalid.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - exponent_maps
            exponent_maps:
              fsle:
                enable: false
              ftle:
                enable: false
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="At least one of exponent_maps.fsle or exponent_maps.ftle"):
        load_postprocess_config(cfg_path)


def test_load_postprocess_config_rejects_invalid_ftle_sampling_mode(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_exponent_maps_sampling.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - exponent_maps
            exponent_maps:
              fsle:
                enable: true
                scale: [2.0]
              ftle:
                enable: true
                scale: [3.0]
                sampling_mode: nearest_after
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exponent_maps.ftle.sampling_mode"):
        load_postprocess_config(cfg_path)


def test_compute_exponent_maps_fsle_uses_minimum_crossing_time_across_members() -> None:
    df = pd.DataFrame(
        _build_group_rows(
            group_id=1,
            center_lon=0.0,
            center_lat=0.0,
            release_time="2026-01-01",
            member_distances_km={2: [1.0, 3.0, 5.0], 3: [1.0, 2.0, 4.0]},
            day_offsets=[0, 1, 2],
        )
    )

    result = compute_exponent_maps(
        df,
        meridional_only=True,
        fsle_scales_km=(2.5,),
    )

    row = result.fsle_points.iloc[0]
    assert row["member_b"] == 2
    assert row["age_days"] == pytest.approx(1.0)
    assert row["fsle"] == pytest.approx(math.log(3.0 / 1.0))


def test_compute_exponent_maps_ftle_respects_sampling_mode() -> None:
    df = pd.DataFrame(
        _build_group_rows(
            group_id=1,
            center_lon=0.0,
            center_lat=0.0,
            release_time="2026-01-01",
            member_distances_km={2: [1.0, 5.0, 3.0]},
            day_offsets=[0, 1, 2],
        )
    )

    last_result = compute_exponent_maps(
        df,
        meridional_only=True,
        ftle_scales_days=(2.0,),
        ftle_sampling_mode="last_before_or_at",
    )
    max_result = compute_exponent_maps(
        df,
        meridional_only=True,
        ftle_scales_days=(2.0,),
        ftle_sampling_mode="max_within_window",
    )

    assert last_result.ftle_points.iloc[0]["ftle"] == pytest.approx(math.log(3.0) / 2.0)
    assert max_result.ftle_points.iloc[0]["ftle"] == pytest.approx(math.log(5.0) / 1.0)


def test_compute_exponent_maps_ftle_last_before_or_at_requires_later_sample() -> None:
    df = pd.DataFrame(
        _build_group_rows(
            group_id=1,
            center_lon=0.0,
            center_lat=0.0,
            release_time="2026-01-01",
            member_distances_km={2: [1.0, 2.0, 4.0]},
            day_offsets=[0, 1, 2],
        )
    )

    result = compute_exponent_maps(
        df,
        meridional_only=True,
        ftle_scales_days=(3.0,),
        ftle_sampling_mode="last_before_or_at",
    )

    row = result.ftle_points.iloc[0]
    assert not bool(row["is_valid"])
    assert pd.isna(row["ftle"])


def test_compute_exponent_maps_clamps_nonstretching_points_to_zero() -> None:
    df = pd.DataFrame(
        _build_group_rows(
            group_id=1,
            center_lon=0.0,
            center_lat=0.0,
            release_time="2026-01-01",
            member_distances_km={2: [1.0, 0.5, 0.25]},
            day_offsets=[0, 1, 2],
        )
    )

    result = compute_exponent_maps(
        df,
        meridional_only=True,
        fsle_scales_km=(0.1,),
        ftle_scales_days=(2.0,),
        ftle_mask_zeros=False,
    )

    assert result.fsle_points.iloc[0]["fsle"] == pytest.approx(0.0)
    assert result.ftle_points.iloc[0]["ftle"] == pytest.approx(0.0)


def test_compute_exponent_maps_backward_growth_is_negative() -> None:
    df = pd.DataFrame(
        _build_group_rows(
            group_id=1,
            center_lon=0.0,
            center_lat=0.0,
            release_time="2026-01-03",
            member_distances_km={2: [1.0, 2.0, 4.0]},
            day_offsets=[0, -1, -2],
        )
    )

    result = compute_exponent_maps(
        df,
        meridional_only=True,
        fsle_scales_km=(1.5,),
        ftle_scales_days=(2.0,),
    )

    assert result.simulation_direction == "backward"
    assert result.fsle_points.iloc[0]["fsle"] < 0
    assert result.ftle_points.iloc[0]["ftle"] < 0


def test_run_exponent_maps_writes_gridded_netcdf_outputs(tmp_path) -> None:
    trajectory_df = _build_regular_release_df(release_times=("2026-01-01", "2026-01-05"))
    summary_df = build_particle_summary(trajectory_df)

    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="./dummy.zarr"),
        analysis=AnalysisConfig(types=("exponent_maps",)),
        output=OutputConfig(output_dir=str(tmp_path)),
        release=ReleaseConfig(mode="region_grid", continuous=True),
        exponent_maps=ExponentMapsConfig(
            distance="meridional",
            fsle=ExponentMapsFSLEConfig(enabled=True, scales_km=(2.0,), mask_zeros=False),
            ftle=ExponentMapsFTLEConfig(
                enabled=True,
                scales_days=(2.0,),
                sampling_mode="last_before_or_at",
                mask_short_windows=True,
            ),
        ),
        plotting=PlottingConfig(projection="PlateCarree"),
    )

    run_exponent_maps(
        cfg,
        {
            "trajectory_table": trajectory_df,
            "particle_summary": summary_df,
        },
    )

    fsle_path = tmp_path / "fsle_map_meridional.nc"
    ftle_path = tmp_path / "ftle_map_meridional.nc"
    assert fsle_path.exists()
    assert ftle_path.exists()

    with xr.open_dataset(fsle_path) as fsle_ds:
        assert fsle_ds["fsle"].dims == ("time", "scale_km", "lat", "lon")
        assert fsle_ds.sizes["time"] == 2
        assert fsle_ds.sizes["scale_km"] == 1
        assert np.isfinite(fsle_ds["fsle"].values).sum() == 8

    with xr.open_dataset(ftle_path) as ftle_ds:
        assert ftle_ds["ftle"].dims == ("time", "scale_days", "lat", "lon")
        assert ftle_ds.sizes["time"] == 2
        assert ftle_ds.sizes["scale_days"] == 1
        assert np.isfinite(ftle_ds["ftle"].values).sum() == 8
        assert np.isinf(ftle_ds["ftle"].values).sum() == 0


def test_run_exponent_maps_accepts_sparse_release_grid(tmp_path) -> None:
    trajectory_df = _build_regular_release_df(release_times=("2026-01-01",))
    trajectory_df = trajectory_df.loc[trajectory_df["group_id"] != 4].copy()
    summary_df = build_particle_summary(trajectory_df)

    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="./dummy.zarr"),
        analysis=AnalysisConfig(types=("exponent_maps",)),
        output=OutputConfig(output_dir=str(tmp_path)),
        release=ReleaseConfig(mode="region_grid", continuous=False),
        exponent_maps=ExponentMapsConfig(
            distance="meridional",
            infer_grid_from_start=False,
            fsle=ExponentMapsFSLEConfig(enabled=True, scales_km=(2.0,), mask_zeros=False),
            ftle=ExponentMapsFTLEConfig(enabled=False),
        ),
        plotting=PlottingConfig(projection="PlateCarree"),
    )

    run_exponent_maps(
        cfg,
        {
            "trajectory_table": trajectory_df,
            "particle_summary": summary_df,
        },
    )

    fsle_path = tmp_path / "fsle_map_meridional.nc"
    assert fsle_path.exists()


def test_run_exponent_maps_uses_configured_grid_spacing_when_present(tmp_path) -> None:
    trajectory_df = _build_regular_release_df(release_times=("2026-01-01",))
    summary_df = build_particle_summary(trajectory_df)

    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="./dummy.zarr"),
        analysis=AnalysisConfig(types=("exponent_maps",)),
        output=OutputConfig(output_dir=str(tmp_path)),
        release=ReleaseConfig(mode="region_grid", continuous=False),
        grid=GridConfig(
            mode="from_initial_centers",
            lon_min=-1.0,
            lon_max=3.0,
            lat_min=-1.0,
            lat_max=3.0,
            dlon=2.0,
            dlat=2.0,
        ),
        exponent_maps=ExponentMapsConfig(
            distance="meridional",
            fsle=ExponentMapsFSLEConfig(enabled=True, scales_km=(2.0,), mask_zeros=False),
            ftle=ExponentMapsFTLEConfig(enabled=False),
        ),
        plotting=PlottingConfig(projection="PlateCarree"),
    )

    run_exponent_maps(
        cfg,
        {
            "trajectory_table": trajectory_df,
            "particle_summary": summary_df,
        },
    )

    fsle_path = tmp_path / "fsle_map_meridional.nc"
    with xr.open_dataset(fsle_path) as fsle_ds:
        assert fsle_ds.attrs["dlon"] == pytest.approx(2.0)
        assert fsle_ds.attrs["dlat"] == pytest.approx(2.0)
