from __future__ import annotations

import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.meridional_excursion import (
    compute_meridional_excursion,
    compute_meridional_excursion_table,
)
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
    DatasetConfig,
    GridConfig,
    MeridionalExcursionConfig,
    MeridionalExcursionGriddingConfig,
    PostprocessConfig,
)
from kinematicparcels.postprocessing.core import RegularGrid
from kinematicparcels.postprocessing.io import save_dataset_netcdf


def _build_trajectory(
    trajectory: str,
    *,
    lon: list[float],
    lat: list[float],
    start: str = "2026-01-01T00:00:00",
    step_hours: float = 24.0,
    group_member: int | None = None,
) -> pd.DataFrame:
    times = pd.date_range(
        start=start,
        periods=len(lon),
        freq=pd.to_timedelta(step_hours, unit="h"),
    )
    out = pd.DataFrame(
        {
            "trajectory": [trajectory] * len(lon),
            "obs": list(range(len(lon))),
            "time": times,
            "lon": lon,
            "lat": lat,
        }
    )
    if group_member is not None:
        out["group_member"] = group_member
    return out


def _base_grid() -> RegularGrid:
    return RegularGrid(
        lon_min=0.0,
        lon_max=5.0,
        lat_min=-2.0,
        lat_max=4.0,
        dlon=1.0,
        dlat=1.0,
    )


def test_meridional_excursion_table_records_first_extrema_occurrence() -> None:
    df = _build_trajectory(
        "a",
        lon=[1.0, 2.0, 3.0, 4.0],
        lat=[1.0, -1.0, -1.0, 3.0],
    )

    table = compute_meridional_excursion_table(df)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["lat0"] == pytest.approx(1.0)
    assert row["lon0"] == pytest.approx(1.0)
    assert row["lat_min"] == pytest.approx(-1.0)
    assert row["lon_at_lat_min"] == pytest.approx(2.0)
    assert row["age_at_lat_min_days"] == pytest.approx(1.0)
    assert row["lat_max"] == pytest.approx(3.0)
    assert row["lon_at_lat_max"] == pytest.approx(4.0)
    assert row["age_at_lat_max_days"] == pytest.approx(3.0)
    assert row["southward_excursion_deg"] == pytest.approx(2.0)
    assert row["northward_excursion_deg"] == pytest.approx(2.0)
    assert row["duration_days"] == pytest.approx(3.0)


def test_meridional_excursion_filters_short_trajectories_by_duration() -> None:
    df = pd.concat(
        [
            _build_trajectory("short", lon=[1.0, 2.0], lat=[1.0, 0.0], step_hours=12.0),
            _build_trajectory("long", lon=[1.0, 2.0, 3.0], lat=[1.0, 0.0, 2.0]),
        ],
        ignore_index=True,
    )

    table = compute_meridional_excursion_table(df, min_duration_days=1.0)

    assert table["trajectory"].to_list() == ["long"]


def test_meridional_excursion_treats_group_members_as_separate_rows() -> None:
    df = pd.concat(
        [
            _build_trajectory("g", lon=[1.0, 1.0], lat=[0.0, -1.0], group_member=1),
            _build_trajectory("g", lon=[2.0, 2.0], lat=[0.0, 2.0], group_member=2),
        ],
        ignore_index=True,
    )

    table = compute_meridional_excursion_table(df)

    assert len(table) == 2
    assert table["group_member"].to_list() == [1, 2]
    assert table["southward_excursion_deg"].to_list() == pytest.approx([1.0, 0.0])
    assert table["northward_excursion_deg"].to_list() == pytest.approx([0.0, 2.0])


def test_meridional_excursion_gridded_dataset_uses_configured_anchor_and_merge() -> None:
    df = pd.concat(
        [
            _build_trajectory("a", lon=[1.1, 1.2, 1.3], lat=[1.0, 0.0, 2.0]),
            _build_trajectory("b", lon=[1.4, 1.2, 1.6], lat=[1.5, -1.0, 3.0]),
        ],
        ignore_index=True,
    )
    cfg = MeridionalExcursionConfig(
        gridding=MeridionalExcursionGriddingConfig(
            merge="mean",
            variables=("southward_excursion_deg",),
            over=("initial_position", "southmost_point"),
        )
    )

    result = compute_meridional_excursion(df, grid=_base_grid(), cfg=cfg)

    initial_var = "southward_excursion_deg_at_initial_position_mean"
    southmost_var = "southward_excursion_deg_at_southmost_point_mean"
    assert initial_var in result.dataset
    assert southmost_var in result.dataset
    assert float(result.dataset[initial_var].sel(lat=1.5, lon=1.5)) == pytest.approx(1.75)
    assert float(result.dataset[southmost_var].sel(lat=0.5, lon=1.5)) == pytest.approx(1.0)
    assert float(result.dataset[southmost_var].sel(lat=-0.5, lon=1.5)) == pytest.approx(2.5)
    assert float(result.dataset[f"{initial_var.removesuffix('_mean')}_count"].sel(lat=1.5, lon=1.5)) == 2.0


def test_meridional_excursion_normalizes_longitude_to_grid_for_binning() -> None:
    df = _build_trajectory(
        "wrap",
        lon=[-1.0, -1.0],
        lat=[1.0, 0.0],
    )
    grid = RegularGrid(
        lon_min=0.0,
        lon_max=360.0,
        lat_min=0.0,
        lat_max=2.0,
        dlon=1.0,
        dlat=1.0,
    )
    cfg = MeridionalExcursionConfig(
        gridding=MeridionalExcursionGriddingConfig(
            merge="mean",
            variables=("southward_excursion_deg",),
            over=("initial_position",),
        )
    )

    result = compute_meridional_excursion(df, grid=grid, cfg=cfg)

    var_name = "southward_excursion_deg_at_initial_position_mean"
    assert float(result.dataset[var_name].sel(lat=1.5, lon=359.5)) == pytest.approx(1.0)


def test_meridional_excursion_dataset_writes_to_netcdf(tmp_path) -> None:
    df = _build_trajectory("a", lon=[1.0, 2.0], lat=[1.0, 0.0])

    result = compute_meridional_excursion(df, grid=_base_grid(), cfg=MeridionalExcursionConfig())

    outpath = tmp_path / "meridional_excursion.nc"
    save_dataset_netcdf(result.dataset, outpath)

    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_load_postprocess_config_parses_meridional_excursion_section(tmp_path) -> None:
    cfg_path = tmp_path / "post.yml"
    cfg_path.write_text(
        """
dataset:
  input_path: ./dummy.zarr
analysis:
  types:
    - meridional_excursion
grid:
  mode: explicit_edges
  lon_min: 0
  lon_max: 5
  lat_min: -2
  lat_max: 4
  dlon: 1
  dlat: 1
meridional_excursion:
  min_duration_days: 2
  output:
    save_table: true
    save_grid_table: false
    save_netcdf: true
    save_figures: false
  gridding:
    merge: max
    variables:
      - southward_excursion_deg
      - lat_min
    over:
      - initial_position
      - southmost_point
  plotting:
    enabled: true
    type:
      - scatter
      - gridded
    variables:
      southward_excursion_deg:
        over:
          - initial_position
        vmin: 0
        vmax: 5
        cmap: viridis
        title: Southward excursion
        cbar_label: Southward excursion [deg]
""",
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("meridional_excursion",)
    assert cfg.meridional_excursion.min_duration_days == pytest.approx(2.0)
    assert cfg.meridional_excursion.output.save_grid_table is False
    assert cfg.meridional_excursion.gridding.merge == "max"
    assert cfg.meridional_excursion.gridding.variables == (
        "southward_excursion_deg",
        "lat_min",
    )
    assert cfg.meridional_excursion.gridding.over == (
        "initial_position",
        "southmost_point",
    )
    plot_cfg = cfg.meridional_excursion.plotting.variables["southward_excursion_deg"]
    assert cfg.meridional_excursion.plotting.type == ("scatter", "gridded")
    assert plot_cfg.over == ("initial_position",)
    assert plot_cfg.vmax == pytest.approx(5.0)
    assert plot_cfg.cmap == "viridis"
    assert plot_cfg.title == "Southward excursion"
    assert plot_cfg.cbar_label == "Southward excursion [deg]"


def test_postprocess_config_can_be_constructed_with_meridional_excursion() -> None:
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="dummy.zarr"),
        grid=GridConfig(
            mode="explicit_edges",
            lon_min=0.0,
            lon_max=5.0,
            lat_min=-2.0,
            lat_max=4.0,
            dlon=1.0,
            dlat=1.0,
        ),
    )

    assert cfg.meridional_excursion.gridding.merge == "mean"
