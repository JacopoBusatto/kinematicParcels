from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kinematicparcels.postprocessing.analyses.gridded_transition_matrix import (
    _cardinal_sector_weights,
    _spherical_initial_bearing_degrees,
    compute_gridded_transition_matrix,
)
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
    DatasetConfig,
    GriddedTransitionMatrixConfig,
    GriddedTransitionMatrixEntropyPlottingConfig,
    GriddedTransitionMatrixMapPlottingConfig,
    GriddedTransitionMatrixOutputConfig,
    GriddedTransitionMatrixPlottingConfig,
    GridConfig,
    OutputConfig,
    PostprocessConfig,
)
from kinematicparcels.postprocessing.core import RegularGrid
from kinematicparcels.postprocessing.io import save_dataset_netcdf
from kinematicparcels.postprocessing.plotting.maps import (
    _prepare_log_scaled_grid_values,
)
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
    assert float(ds["entropy"].sel(lat=0.5, lon=0.5)) == 0.0


@pytest.mark.parametrize(
    ("log_base", "expected", "units"),
    (
        ("e", np.log(2.0), "nats"),
        (2, 1.0, "bits"),
        (10, np.log10(2.0), "hartleys"),
    ),
)
def test_gridded_transition_matrix_entropy_supports_configured_log_base(
    log_base: str | int,
    expected: float,
    units: str,
) -> None:
    df = pd.concat(
        [
            _build_trajectory("stay", lon=[0.2, 0.2], lat=[0.2, 0.2]),
            _build_trajectory("east", lon=[0.2, 1.2], lat=[0.2, 0.2]),
        ],
        ignore_index=True,
    )
    cfg = GriddedTransitionMatrixConfig(
        plotting=GriddedTransitionMatrixPlottingConfig(
            entropy=GriddedTransitionMatrixEntropyPlottingConfig(
                log_base=log_base
            )
        )
    )

    result = compute_gridded_transition_matrix(df, grid=_base_grid(), cfg=cfg)

    entropy = result.dataset["entropy"]
    assert float(entropy.sel(lat=0.5, lon=0.5)) == pytest.approx(expected)
    assert entropy.attrs["units"] == units
    assert entropy.attrs["log_base"] == log_base


def test_gridded_transition_matrix_entropy_uses_unequal_full_destination_row() -> None:
    df = pd.concat(
        [
            _build_trajectory("east_1", lon=[0.2, 1.2], lat=[0.2, 0.2]),
            _build_trajectory("east_2", lon=[0.2, 1.2], lat=[0.2, 0.2]),
            _build_trajectory("north", lon=[0.2, 0.2], lat=[0.2, 1.2]),
        ],
        ignore_index=True,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(),
    )

    expected = -(2.0 / 3.0) * np.log(2.0 / 3.0) - (1.0 / 3.0) * np.log(
        1.0 / 3.0
    )
    assert float(result.dataset["entropy"].sel(lat=0.5, lon=0.5)) == pytest.approx(
        expected
    )
    assert np.isnan(float(result.dataset["entropy"].sel(lat=2.5, lon=2.5)))


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
    assert float(result.dataset["entropy"].sel(lat=0.5, lon=0.5)) == 0.0
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
    assert np.isnan(result.dataset["entropy"].values).all()
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


def test_gridded_transition_matrix_resample_uses_non_overlapping_segments() -> None:
    grid = RegularGrid(
        lon_min=0.0,
        lon_max=5.0,
        lat_min=0.0,
        lat_max=1.0,
        dlon=1.0,
        dlat=1.0,
    )
    df = _build_trajectory(
        "resampled",
        lon=[0.2, 1.2, 2.2, 3.2, 4.2],
        lat=[0.2] * 5,
        step_hours=24.0,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=grid,
        cfg=GriddedTransitionMatrixConfig(
            timestep=2.0,
            timestep_unit="days",
            resample=True,
        ),
    )

    table = result.transition_table.sort_values("start_lon_bin").reset_index(drop=True)
    assert table["start_lon_bin"].tolist() == [0, 2]
    assert table["end_lon_bin"].tolist() == [2, 4]
    assert table["transition_count"].tolist() == [1, 1]
    assert int(result.dataset["n_segments_start"].sum()) == 2
    assert result.dataset.attrs["segment_start_policy"] == "regular_non_overlapping"


def test_gridded_transition_matrix_resample_interpolates_regular_grid() -> None:
    df = _build_trajectory(
        "resampled_interp",
        lon=[0.1, 2.1],
        lat=[0.1, 0.1],
        step_hours=24.0,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=_base_grid(),
        cfg=GriddedTransitionMatrixConfig(
            timestep=12.0,
            timestep_unit="hours",
            resample=True,
        ),
    )

    table = result.transition_table.sort_values("start_lon_bin").reset_index(drop=True)
    assert table["start_lon_bin"].tolist() == [0, 1]
    assert table["end_lon_bin"].tolist() == [1, 2]
    assert int(result.dataset["n_segments_start"].sum()) == 2


def test_gridded_transition_matrix_default_timestep_segments_can_overlap() -> None:
    grid = RegularGrid(
        lon_min=0.0,
        lon_max=5.0,
        lat_min=0.0,
        lat_max=1.0,
        dlon=1.0,
        dlat=1.0,
    )
    df = _build_trajectory(
        "overlapping",
        lon=[0.2, 1.2, 2.2, 3.2, 4.2],
        lat=[0.2] * 5,
        step_hours=24.0,
    )

    result = compute_gridded_transition_matrix(
        df,
        grid=grid,
        cfg=GriddedTransitionMatrixConfig(timestep=2.0, timestep_unit="days"),
    )

    table = result.transition_table.sort_values("start_lon_bin").reset_index(drop=True)
    assert table["start_lon_bin"].tolist() == [0, 1, 2]
    assert table["end_lon_bin"].tolist() == [2, 3, 4]
    assert int(result.dataset["n_segments_start"].sum()) == 3
    assert result.dataset.attrs["segment_start_policy"] == "every_observation"


def test_gridded_transition_matrix_resample_requires_timestep() -> None:
    with pytest.raises(ValueError, match="timestep must be set"):
        GriddedTransitionMatrixConfig(resample=True)


def test_gridded_transition_matrix_resample_rejects_non_boolean() -> None:
    with pytest.raises(ValueError, match="resample must be true or false"):
        GriddedTransitionMatrixConfig(
            timestep=1.0,
            resample="false",  # type: ignore[arg-type]
        )


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
    assert result.dataset["entropy"].attrs["formula"] == "-sum_j P_ij log_b(P_ij)"
    assert "cardinal bearing sector" in result.dataset["probability_east"].attrs[
        "long_name"
    ]
    with xr.open_dataset(outpath) as reopened:
        xr.testing.assert_equal(reopened["entropy"], result.dataset["entropy"])
        assert reopened["entropy"].attrs["units"] == "nats"
        assert reopened["entropy"].attrs["log_base"] == "e"


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
  resample: true
  output:
    save_table: true
    save_netcdf: true
    save_figures: false
  plotting:
    enabled: true
    probability:
      cmap: cividis
      as_percent: true
      vmin: 0
      vmax: 1
    entropy:
      enabled: false
      log_base: 2
      cmap: plasma
      log_scale: true
      zero_color: cyan
      vmin: 0.01
      vmax: 4
""",
        encoding="utf-8",
    )

    cfg = load_postprocess_config(config_path)

    assert cfg.analysis.types == ("gridded_transition_matrix",)
    assert cfg.gridded_transition_matrix.timestep == 12.0
    assert cfg.gridded_transition_matrix.timestep_unit == "hours"
    assert cfg.gridded_transition_matrix.resample is True
    assert cfg.gridded_transition_matrix.output.save_figures is False
    assert cfg.gridded_transition_matrix.plotting.probability.cmap == "cividis"
    assert cfg.gridded_transition_matrix.plotting.probability.as_percent is True
    assert cfg.gridded_transition_matrix.plotting.entropy.enabled is False
    assert cfg.gridded_transition_matrix.plotting.entropy.log_base == 2
    assert cfg.gridded_transition_matrix.plotting.entropy.cmap == "plasma"
    assert cfg.gridded_transition_matrix.plotting.entropy.log_scale is True
    assert cfg.gridded_transition_matrix.plotting.entropy.zero_color == "cyan"
    assert cfg.gridded_transition_matrix.plotting.entropy.vmax == 4.0


def test_load_postprocess_config_rejects_legacy_transition_cmap(tmp_path) -> None:
    config_path = tmp_path / "postprocess.yml"
    config_path.write_text(
        """
dataset:
  input_path: trajectories.zarr
output:
  output_dir: outputs
grid:
  mode: explicit_edges
  lon_min: 0
  lon_max: 1
  lat_min: 0
  lat_max: 1
  dlon: 1
  dlat: 1
gridded_transition_matrix:
  plotting:
    cmap: viridis
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probability.cmap"):
        load_postprocess_config(config_path)


@pytest.mark.parametrize("log_base", ("2", "E", 3, True))
def test_transition_entropy_rejects_invalid_log_base(log_base) -> None:
    with pytest.raises(ValueError, match="entropy.log_base"):
        GriddedTransitionMatrixEntropyPlottingConfig(log_base=log_base)


@pytest.mark.parametrize(
    "config_factory",
    (
        lambda: GriddedTransitionMatrixMapPlottingConfig(cmap=""),
        lambda: GriddedTransitionMatrixEntropyPlottingConfig(cmap=""),
        lambda: GriddedTransitionMatrixMapPlottingConfig(vmin=2.0, vmax=1.0),
        lambda: GriddedTransitionMatrixEntropyPlottingConfig(vmin=2.0, vmax=1.0),
        lambda: GriddedTransitionMatrixEntropyPlottingConfig(zero_color=""),
        lambda: GriddedTransitionMatrixEntropyPlottingConfig(
            log_scale=True, vmin=0.0
        ),
        lambda: GriddedTransitionMatrixEntropyPlottingConfig(
            log_scale=True, vmax=0.0
        ),
        lambda: GriddedTransitionMatrixEntropyPlottingConfig(
            log_scale=True, vmin=1.0, vmax=1.0
        ),
    ),
)
def test_transition_plot_sections_validate_colormap_and_limits(config_factory) -> None:
    with pytest.raises(ValueError):
        config_factory()


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
    assert cfg.gridded_transition_matrix.resample is False
    assert cfg.gridded_transition_matrix.plotting.probability.cmap == "viridis"
    assert cfg.gridded_transition_matrix.plotting.entropy.enabled is True
    assert cfg.gridded_transition_matrix.plotting.entropy.log_base == "e"
    assert cfg.gridded_transition_matrix.plotting.entropy.cmap == "magma"
    assert cfg.gridded_transition_matrix.plotting.entropy.log_scale is False
    assert cfg.gridded_transition_matrix.plotting.entropy.zero_color == "lightgray"


def test_entropy_log_scale_masks_positive_field_and_tracks_exact_zeros() -> None:
    values = np.asarray([[0.0, np.nan], [0.1, 1.0]])

    plot_values, zero_mask, norm = _prepare_log_scaled_grid_values(
        values,
        vmin=None,
        vmax=None,
    )

    np.testing.assert_array_equal(
        np.ma.getmaskarray(plot_values),
        [[True, True], [False, False]],
    )
    np.testing.assert_array_equal(
        zero_mask,
        [[True, False], [False, False]],
    )
    assert norm.vmin == pytest.approx(0.1)
    assert norm.vmax == pytest.approx(1.0)


def test_entropy_log_scale_requires_a_positive_value() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        _prepare_log_scaled_grid_values(
            np.asarray([[0.0, np.nan]]),
            vmin=None,
            vmax=None,
        )


def test_gridded_transition_matrix_plots_use_per_product_colormaps(
    tmp_path,
    monkeypatch,
) -> None:
    workflow = importlib.import_module(
        "kinematicparcels.postprocessing.workflows.run_gridded_transition_matrix"
    )
    df = _build_trajectory("east", lon=[0.2, 1.2], lat=[0.2, 0.2])
    plot_calls: list[dict] = []
    monkeypatch.setattr(workflow, "get_trajectory_table", lambda cfg, context: df)
    monkeypatch.setattr(
        workflow,
        "plot_grid_map",
        lambda *args, **kwargs: plot_calls.append(kwargs),
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="trajectories.zarr"),
        output=OutputConfig(output_dir=str(tmp_path)),
        grid=GridConfig(
            mode="explicit_edges",
            lon_min=0.0,
            lon_max=3.0,
            lat_min=0.0,
            lat_max=3.0,
            dlon=1.0,
            dlat=1.0,
        ),
        gridded_transition_matrix=GriddedTransitionMatrixConfig(
            output=GriddedTransitionMatrixOutputConfig(
                save_table=False,
                save_netcdf=False,
                save_figures=True,
            ),
            plotting=GriddedTransitionMatrixPlottingConfig(
                probability=GriddedTransitionMatrixMapPlottingConfig(
                    cmap="cividis"
                ),
                entropy=GriddedTransitionMatrixEntropyPlottingConfig(
                    cmap="plasma",
                    log_scale=True,
                    zero_color="cyan",
                    vmin=0.01,
                    vmax=2.0,
                ),
            ),
        ),
    )

    workflow.run_gridded_transition_matrix(cfg, {"grid": _base_grid()})

    probability_calls = [
        call for call in plot_calls if call["var_name"].startswith("probability_")
    ]
    entropy_calls = [call for call in plot_calls if call["var_name"] == "entropy"]
    assert len(probability_calls) == 5
    assert all(call["cmap"] == "cividis" for call in probability_calls)
    assert len(entropy_calls) == 1
    entropy_call = entropy_calls[0]
    assert entropy_call["cmap"] == "plasma"
    assert entropy_call["vmin"] == 0.01
    assert entropy_call["vmax"] == 2.0
    assert entropy_call["log_scale"] is True
    assert entropy_call["zero_color"] == "cyan"
    assert entropy_call["outpath"].name == "gridded_transition_entropy_dt_1d.png"
    assert entropy_call["colorbar_label"] == "Entropy [nats]"

    plot_calls.clear()
    disabled_cfg = replace(
        cfg,
        gridded_transition_matrix=replace(
            cfg.gridded_transition_matrix,
            plotting=replace(
                cfg.gridded_transition_matrix.plotting,
                entropy=replace(
                    cfg.gridded_transition_matrix.plotting.entropy,
                    enabled=False,
                ),
            ),
        ),
    )
    workflow.run_gridded_transition_matrix(disabled_cfg, {"grid": _base_grid()})
    assert not any(call["var_name"] == "entropy" for call in plot_calls)
