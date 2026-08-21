from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.gridded_transition_matrix import (
    _cardinal_sector_weights,
    _spherical_initial_bearing_degrees,
    compute_gridded_transition_matrix,
)
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
    DatasetConfig,
    GriddedTransitionMatrixConfig,
    GridConfig,
    OutputConfig,
    PostprocessConfig,
)
from kinematicparcels.postprocessing.core import RegularGrid
from kinematicparcels.postprocessing.io import save_dataset_netcdf
from kinematicparcels.postprocessing.workflows.run_gridded_transition_matrix import (
    _timestep_output_label,
)


def _build_trajectory(
    trajectory: str,
    *,
    lon: list[float],
    lat: list[float],
    start: str = "2026-01-01T00:00:00",
    step_hours: float = 24.0,
) -> pd.DataFrame:
    times = pd.date_range(
        start=start,
        periods=len(lon),
        freq=pd.to_timedelta(step_hours, unit="h"),
    )
    return pd.DataFrame(
        {
            "trajectory": [trajectory] * len(lon),
            "obs": list(range(len(lon))),
            "time": times,
            "lon": lon,
            "lat": lat,
        }
    )


def _base_grid() -> RegularGrid:
    return RegularGrid(
        lon_min=0.0,
        lon_max=3.0,
        lat_min=0.0,
        lat_max=3.0,
        dlon=1.0,
        dlat=1.0,
    )


def test_gridded_transition_matrix_counts_and_normalizes_native_segments() -> None:
    df = pd.concat(
        [
            _build_trajectory("north", lon=[0.2, 0.2], lat=[0.2, 1.2]),
            _build_trajectory("east", lon=[0.2, 1.2], lat=[0.2, 0.2]),
        ],
        ignore_index=True,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    ds = result.dataset
    assert int(ds["n_segments_start"].sel(lat=0.5, lon=0.5)) == 2
    assert float(ds["probability_north"].sel(lat=0.5, lon=0.5)) == 0.5
    assert float(ds["probability_east"].sel(lat=0.5, lon=0.5)) == 0.5
    assert float(ds["probability_south"].sel(lat=0.5, lon=0.5)) == 0.0
    assert float(ds["probability_west"].sel(lat=0.5, lon=0.5)) == 0.0
    assert float(ds["probability_stay"].sel(lat=0.5, lon=0.5)) == 0.0
    assert sum(
        float(ds[f"probability_{direction}"].sel(lat=0.5, lon=0.5))
        for direction in ("north", "east", "south", "west", "stay")
    ) == pytest.approx(1.0)
    assert np.isnan(float(ds["probability_north"].sel(lat=2.5, lon=2.5)))

    table = result.transition_table.sort_values(
        ["end_lat_bin", "end_lon_bin"]
    ).reset_index(drop=True)
    assert table["transition_count"].tolist() == [1, 1]
    assert table["transition_probability"].tolist() == [0.5, 0.5]


def test_gridded_transition_matrix_stay_probability_counts_same_cell_transition() -> None:
    df = _build_trajectory("stay", lon=[0.2, 0.8], lat=[0.2, 0.8])

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    ds = result.dataset
    assert float(ds["probability_stay"].sel(lat=0.5, lon=0.5)) == 1.0
    for direction in ("north", "east", "south", "west"):
        assert float(ds[f"probability_{direction}"].sel(lat=0.5, lon=0.5)) == 0.0

    table = result.transition_table.iloc[0]
    assert int(table.start_lon_bin) == int(table.end_lon_bin) == 0
    assert int(table.start_lat_bin) == int(table.end_lat_bin) == 0


def test_gridded_transition_matrix_excludes_segments_with_end_outside_grid() -> None:
    df = pd.concat(
        [
            _build_trajectory("inside", lon=[0.2, 1.2], lat=[0.2, 0.2]),
            _build_trajectory("exit", lon=[0.2, 4.2], lat=[0.2, 0.2]),
        ],
        ignore_index=True,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    assert int(result.dataset["n_segments_start"].sel(lat=0.5, lon=0.5)) == 1
    assert len(result.transition_table) == 1
    assert float(result.transition_table.iloc[0].transition_probability) == 1.0
    assert sum(
        float(result.dataset[f"probability_{direction}"].sel(lat=0.5, lon=0.5))
        for direction in ("north", "east", "south", "west", "stay")
    ) == pytest.approx(1.0)
    assert float(result.dataset["probability_east"].sel(lat=0.5, lon=0.5)) == 1.0


@pytest.mark.parametrize(
    "lon",
    (
        [0.2, 4.2],
        [-1.2, 0.2],
    ),
)
def test_gridded_transition_matrix_requires_both_endpoints_inside(
    lon: list[float],
) -> None:
    df = _build_trajectory("outside", lon=lon, lat=[0.2, 0.2])

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    assert result.transition_table.empty
    assert int(result.dataset["n_segments_start"].sum()) == 0
    for direction in ("north", "east", "south", "west", "stay"):
        assert np.isnan(result.dataset[f"probability_{direction}"].values).all()


def test_gridded_transition_matrix_smaller_timestep_interpolates_endpoint() -> None:
    df = _build_trajectory("interp", lon=[0.1, 2.1], lat=[0.1, 0.1], step_hours=24.0)

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(timestep=12.0, timestep_unit="hours"),
    )

    table = result.transition_table.iloc[0]
    assert int(table.start_lon_bin) == 0
    assert int(table.start_lat_bin) == 0
    assert int(table.end_lon_bin) == 1
    assert int(table.end_lat_bin) == 0
    assert int(table.transition_count) == 1
    assert float(table.transition_probability) == 1.0


def test_gridded_transition_matrix_larger_timestep_uses_matching_future_observation() -> None:
    df = _build_trajectory(
        "stride",
        lon=[0.2, 1.2, 2.2],
        lat=[0.2, 0.2, 0.2],
        step_hours=24.0,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(timestep=2.0, timestep_unit="days"),
    )

    assert int(result.dataset["n_segments_start"].sel(lat=0.5, lon=0.5)) == 1
    table = result.transition_table.iloc[0]
    assert int(table.start_lon_bin) == 0
    assert int(table.end_lon_bin) == 2


def test_gridded_transition_matrix_rejects_larger_non_multiple_timestep() -> None:
    df = _build_trajectory("bad", lon=[0.2, 1.2, 2.2], lat=[0.2, 0.2, 0.2])

    with pytest.raises(ValueError, match="integer multiple"):
        compute_gridded_transition_matrix(
            df,
            grid=_base_grid(),
            cfg=GriddedTransitionMatrixConfig(timestep=36.0, timestep_unit="hours"),
        )


def test_gridded_transition_matrix_east_west_are_periodic() -> None:
    grid = RegularGrid(
        lon_min=-180.0,
        lon_max=180.0,
        lat_min=0.0,
        lat_max=1.0,
        dlon=90.0,
        dlat=1.0,
    )
    df = pd.concat(
        [
            _build_trajectory("east_wrap", lon=[135.0, -135.0], lat=[0.5, 0.5]),
            _build_trajectory("west_wrap", lon=[-135.0, 135.0], lat=[0.5, 0.5]),
        ],
        ignore_index=True,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=grid,
        cfg=GriddedTransitionMatrixConfig(),
    )

    assert float(result.dataset["probability_east"].sel(lat=0.5, lon=135.0)) == 1.0
    assert float(result.dataset["probability_west"].sel(lat=0.5, lon=-135.0)) == 1.0


def test_gridded_transition_matrix_oblique_move_uses_one_cardinal_sector() -> None:
    df = _build_trajectory("east_northeast", lon=[0.2, 2.2], lat=[0.2, 1.2])

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    ds = result.dataset
    assert float(ds["probability_east"].sel(lat=0.5, lon=0.5)) == 1.0
    for direction in ("north", "south", "west", "stay"):
        assert float(ds[f"probability_{direction}"].sel(lat=0.5, lon=0.5)) == 0.0


def test_gridded_transition_matrix_uses_geographic_bearing_at_high_latitude() -> None:
    grid = RegularGrid(
        lon_min=0.0,
        lon_max=2.0,
        lat_min=60.0,
        lat_max=61.5,
        dlon=1.0,
        dlat=0.75,
    )
    df = _build_trajectory(
        "high_latitude",
        lon=[0.2, 1.2],
        lat=[60.2, 61.0],
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=grid,
        cfg=GriddedTransitionMatrixConfig(),
    )

    ds = result.dataset
    assert float(ds["probability_north"].sel(lat=60.375, lon=0.5)) == 1.0
    assert float(ds["probability_east"].sel(lat=60.375, lon=0.5)) == 0.0


def test_cardinal_sector_boundaries_split_probability_equally() -> None:
    bearings = np.asarray([45.0, 135.0, 225.0, 315.0])

    weights = _cardinal_sector_weights(bearings)

    np.testing.assert_allclose(weights["north"], [0.5, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(weights["east"], [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(weights["south"], [0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(weights["west"], [0.0, 0.0, 0.5, 0.5])
    np.testing.assert_allclose(
        sum(weights.values()),
        np.ones(bearings.shape),
    )


def test_gridded_transition_matrix_splits_antipodal_probability_four_ways() -> None:
    grid = RegularGrid(
        lon_min=-180.0,
        lon_max=180.0,
        lat_min=-1.0,
        lat_max=1.0,
        dlon=180.0,
        dlat=2.0,
    )
    df = _build_trajectory("antipodal", lon=[-90.0, 90.0], lat=[0.0, 0.0])

    bearing, undefined = _spherical_initial_bearing_degrees(
        np.asarray([-90.0]),
        np.asarray([0.0]),
        np.asarray([90.0]),
        np.asarray([0.0]),
    )
    assert np.isnan(bearing[0])
    assert bool(undefined[0])

    result = compute_gridded_transition_matrix(
        df,
        grid=grid,
        cfg=GriddedTransitionMatrixConfig(),
    )

    ds = result.dataset
    for direction in ("north", "east", "south", "west"):
        assert float(ds[f"probability_{direction}"].sel(lat=0.0, lon=-90.0)) == 0.25
    assert float(ds["probability_stay"].sel(lat=0.0, lon=-90.0)) == 0.0


def test_gridded_transition_matrix_dataset_writes_to_netcdf(tmp_path) -> None:
    df = _build_trajectory("native", lon=[0.2, 1.2], lat=[0.2, 0.2])
    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    outpath = tmp_path / "gridded_transition_matrix.nc"
    save_dataset_netcdf(result.dataset, outpath)

    assert outpath.exists()
    assert result.dataset.attrs["direction_sector_boundaries_degrees"] == (
        "45, 135, 225, 315"
    )
    assert result.dataset.attrs["segment_inclusion"] == (
        "both start and end points must lie inside the analysis grid"
    )
    assert "cardinal bearing sector" in result.dataset["probability_east"].attrs[
        "long_name"
    ]


def test_gridded_transition_matrix_timestep_output_label_uses_configured_timestep() -> None:
    df = _build_trajectory("native", lon=[0.2, 1.2], lat=[0.2, 0.2])
    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(timestep=12.0, timestep_unit="hours"),
    )

    assert _timestep_output_label(result.dataset) == "dt_12h"


def test_gridded_transition_matrix_timestep_output_label_uses_native_timestep() -> None:
    df = _build_trajectory("native", lon=[0.2, 1.2], lat=[0.2, 0.2])
    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    assert _timestep_output_label(result.dataset) == "dt_1d"


def test_load_postprocess_config_parses_gridded_transition_matrix(tmp_path) -> None:
    config_path = tmp_path / "postprocess.yml"
    config_path.write_text(
        """
dataset:
  input_path: trajectories.zarr
analysis:
  types: [gridded_transition_matrix]
output:
  output_dir: outputs
grid:
  mode: explicit_edges
  lon_min: 0
  lon_max: 3
  lat_min: 0
  lat_max: 3
  dlon: 1
  dlat: 1
gridded_transition_matrix:
  timestep: 12
  timestep_unit: hours
  output:
    save_table: true
    save_netcdf: true
    save_figures: false
  plotting:
    enabled: true
    cmap: magma
    probability:
      as_percent: true
      vmin: 0
      vmax: 1
""",
        encoding="utf-8",
    )

    cfg = load_postprocess_config(config_path)

    assert cfg.analysis.types == ("gridded_transition_matrix",)
    assert cfg.gridded_transition_matrix.timestep == 12.0
    assert cfg.gridded_transition_matrix.timestep_unit == "hours"
    assert cfg.gridded_transition_matrix.output.save_figures is False
    assert cfg.gridded_transition_matrix.plotting.cmap == "magma"
    assert cfg.gridded_transition_matrix.plotting.probability.as_percent is True


def test_postprocess_config_has_gridded_transition_matrix_default() -> None:
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="trajectories.zarr"),
        output=OutputConfig(output_dir="outputs"),
        grid=GridConfig(
            mode="explicit_edges",
            lon_min=0.0,
            lon_max=1.0,
            lat_min=0.0,
            lat_max=1.0,
            dlon=1.0,
            dlat=1.0,
        ),
    )

    assert isinstance(cfg.gridded_transition_matrix, GriddedTransitionMatrixConfig)
    assert cfg.gridded_transition_matrix.timestep is None
    assert cfg.gridded_transition_matrix.plotting.cmap == "viridis"
