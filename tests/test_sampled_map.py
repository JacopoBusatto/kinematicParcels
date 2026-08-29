from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Geod

from kinematicparcels.postprocessing.analyses.sampled_map import (
    compute_sampled_map,
    compute_sphere_aware_gradients,
    gaussian_smooth_supported_field,
)
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
    AnalysisConfig,
    DatasetConfig,
    ExportsConfig,
    GridConfig,
    OutputConfig,
    PlottingConfig,
    PostprocessConfig,
    SampledMapConfig,
    SampledMapGradientsConfig,
    SampledMapPlotConfig,
    SampledMapVariableConfig,
    SampledMapVariablePlottingConfig,
)
from kinematicparcels.postprocessing.core.gridding import RegularGrid
from kinematicparcels.postprocessing.io import save_dataset_netcdf, save_grid_table
from kinematicparcels.postprocessing.workflows.run_sampled_map import (
    resolve_sampled_map_plot_limits,
)


def _grid() -> RegularGrid:
    return RegularGrid(
        lon_min=0.0,
        lon_max=3.0,
        lat_min=0.0,
        lat_max=3.0,
        dlon=1.0,
        dlat=1.0,
    )


def _config(
    *,
    weighting: str = "points",
    gradients: bool = False,
    variables: dict[str, SampledMapVariableConfig] | None = None,
    max_group_member: int | None = None,
) -> SampledMapConfig:
    return SampledMapConfig(
        variables=variables or {"temp": SampledMapVariableConfig()},
        weighting=weighting,
        max_group_member=max_group_member,
        gradients=SampledMapGradientsConfig(
            enabled=gradients,
            smoothing_sigma_km=100.0 if gradients else None,
        ),
    )


def test_point_and_trajectory_weighting_use_distinct_effective_samples() -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["a", "a", "a", "b"],
            "lon": [0.2, 0.3, 0.4, 0.5],
            "lat": [0.2, 0.3, 0.4, 0.5],
            "temp": [0.0, 0.0, 0.0, 10.0],
        }
    )

    point_result = compute_sampled_map(df, grid=_grid(), cfg=_config())
    trajectory_result = compute_sampled_map(
        df,
        grid=_grid(),
        cfg=_config(weighting="trajectories"),
    )

    assert float(point_result.dataset.temp_mean[0, 0]) == pytest.approx(2.5)
    assert float(point_result.dataset.temp_std[0, 0]) == pytest.approx(5.0)
    assert float(trajectory_result.dataset.temp_mean[0, 0]) == pytest.approx(5.0)
    assert float(trajectory_result.dataset.temp_std[0, 0]) == pytest.approx(
        np.sqrt(50.0)
    )
    assert int(point_result.dataset.temp_point_count[0, 0]) == 4
    assert int(point_result.dataset.temp_trajectory_count[0, 0]) == 2


def test_filters_ranges_support_and_group_members_without_losing_counts() -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["a", "b", "c", "outside", "nan"],
            "group_member": [1, 2, 3, 1, 1],
            "lon": [0.2, 0.3, 0.4, 9.0, 0.5],
            "lat": [0.2, 0.3, 0.4, 0.3, 0.5],
            "temp": [1.0, 2.0, 30.0, 3.0, np.nan],
        }
    )
    variable_cfg = SampledMapVariableConfig(
        valid_min=0.0,
        valid_max=10.0,
        minimum_point_count=2,
        minimum_trajectory_count=2,
    )

    result = compute_sampled_map(
        df,
        grid=_grid(),
        cfg=_config(
            variables={"temp": variable_cfg},
            max_group_member=2,
        ),
    )

    assert int(result.dataset.temp_point_count[0, 0]) == 2
    assert int(result.dataset.temp_trajectory_count[0, 0]) == 2
    assert float(result.dataset.temp_mean[0, 0]) == pytest.approx(1.5)

    masked = compute_sampled_map(
        df,
        grid=_grid(),
        cfg=_config(
            variables={
                "temp": replace(variable_cfg, minimum_point_count=3)
            },
            max_group_member=2,
        ),
    )
    assert int(masked.dataset.temp_point_count[0, 0]) == 2
    assert np.isnan(masked.dataset.temp_mean[0, 0])
    assert np.isnan(masked.dataset.temp_std[0, 0])


def test_multiple_variables_and_yaml_only_third_variable_share_outputs() -> None:
    df = pd.DataFrame(
        {
            "trajectory": [1, 2],
            "lon": [0.2, 1.2],
            "lat": [0.2, 1.2],
            "temp": [1.0, 2.0],
            "psal": [34.5, 34.7],
            "oxygen": [210.0, 220.0],
        }
    )
    variables = {
        name: SampledMapVariableConfig() for name in ("temp", "psal", "oxygen")
    }

    result = compute_sampled_map(
        df,
        grid=_grid(),
        cfg=_config(variables=variables),
    )

    assert len(result.table) == 9
    for name in variables:
        assert f"{name}_mean" in result.dataset
        assert f"{name}_std" in result.table
        assert f"{name}_point_count" in result.dataset
        assert f"{name}_trajectory_count" in result.dataset


def test_nonnumeric_and_missing_variables_fail_clearly() -> None:
    df = pd.DataFrame(
        {
            "trajectory": [1],
            "lon": [0.2],
            "lat": [0.2],
            "label": ["bad"],
        }
    )
    with pytest.raises(TypeError, match="must be numeric"):
        compute_sampled_map(
            df,
            grid=_grid(),
            cfg=_config(variables={"label": SampledMapVariableConfig()}),
        )
    with pytest.raises(KeyError, match="not present"):
        compute_sampled_map(
            df,
            grid=_grid(),
            cfg=_config(variables={"oxygen": SampledMapVariableConfig()}),
        )


def test_gaussian_smoothing_is_normalized_and_does_not_fill_gaps() -> None:
    field = np.full((3, 3), np.nan)
    field[0, 0] = 2.0
    field[0, 1] = 4.0

    smoothed = gaussian_smooth_supported_field(
        field,
        grid=_grid(),
        sigma_km=100.0,
    )

    assert np.isfinite(smoothed[0, 0])
    assert np.isfinite(smoothed[0, 1])
    assert 2.0 < smoothed[0, 0] < 4.0
    assert 2.0 < smoothed[0, 1] < 4.0
    assert np.isnan(smoothed[1:, :]).all()
    assert np.isnan(smoothed[0, 2])

    constant = np.full((3, 3), 7.0)
    np.testing.assert_allclose(
        gaussian_smooth_supported_field(
            constant,
            grid=_grid(),
            sigma_km=100.0,
        ),
        constant,
    )


def test_gaussian_smoothing_uses_physical_cell_area_weights() -> None:
    grid = RegularGrid(-0.5, 0.5, 60.0, 63.0, 1.0, 1.0)
    field = np.asarray([[0.0], [10.0], [0.0]])
    smoothed = gaussian_smooth_supported_field(field, grid=grid, sigma_km=200.0)

    geod = Geod(ellps="WGS84")
    target_lon = np.full(3, grid.lon_centers[0])
    target_lat = np.full(3, grid.lat_centers[1])
    _, _, distance_m = geod.inv(
        target_lon,
        target_lat,
        np.full(3, grid.lon_centers[0]),
        grid.lat_centers,
    )
    areas = []
    for south, north in zip(grid.lat_edges[:-1], grid.lat_edges[1:]):
        area_m2, _ = geod.polygon_area_perimeter(
            [0.0, 1.0, 1.0, 0.0],
            [south, south, north, north],
        )
        areas.append(abs(area_m2) / 1.0e6)
    weights = np.exp(-0.5 * np.square(distance_m / 1000.0 / 200.0)) * areas
    expected = float(np.average(field[:, 0], weights=weights))
    assert smoothed[1, 0] == pytest.approx(expected)


def test_sphere_aware_gradients_have_physical_units_and_correct_signs() -> None:
    grid = _grid()
    geod = Geod(ellps="WGS84")

    zonal_field = np.empty((grid.nlat, grid.nlon), dtype=float)
    for lat_bin, lat in enumerate(grid.lat_centers):
        zonal_field[lat_bin, 0] = 0.0
        for lon_bin in range(1, grid.nlon):
            _, _, distance_m = geod.inv(
                grid.lon_centers[lon_bin - 1],
                lat,
                grid.lon_centers[lon_bin],
                lat,
            )
            zonal_field[lat_bin, lon_bin] = (
                zonal_field[lat_bin, lon_bin - 1] + distance_m / 1000.0
            )
    zonal = compute_sphere_aware_gradients(zonal_field, grid=grid)
    np.testing.assert_allclose(zonal["zonal_gradient"][:, 1], 1.0, rtol=1.0e-6)
    assert np.all(zonal["zonal_gradient_distance_km"][:, 1] > 200.0)

    meridional_field = np.empty((grid.nlat, grid.nlon), dtype=float)
    meridional_field[0, :] = 0.0
    for lat_bin in range(1, grid.nlat):
        _, _, distance_m = geod.inv(
            grid.lon_centers[0],
            grid.lat_centers[lat_bin - 1],
            grid.lon_centers[0],
            grid.lat_centers[lat_bin],
        )
        meridional_field[lat_bin, :] = (
            meridional_field[lat_bin - 1, :] + distance_m / 1000.0
        )
    meridional = compute_sphere_aware_gradients(meridional_field, grid=grid)
    np.testing.assert_allclose(
        meridional["meridional_gradient"][1, :], 1.0, rtol=1.0e-12
    )
    assert np.all(meridional["meridional_gradient_distance_km"][1, :] > 200.0)


def test_gradients_are_missing_aware_one_sided_and_periodic() -> None:
    grid = _grid()
    field = np.asarray(
        [
            [0.0, 1.0, np.nan],
            [2.0, 3.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    result = compute_sphere_aware_gradients(field, grid=grid)
    assert result["zonal_gradient"][0, 0] > 0.0
    assert result["meridional_gradient"][0, 0] > 0.0
    assert result["gradient_magnitude"][0, 0] == pytest.approx(
        np.hypot(
            result["zonal_gradient"][0, 0],
            result["meridional_gradient"][0, 0],
        )
    )
    assert np.isnan(result["zonal_gradient"][0, 2])

    global_grid = RegularGrid(-180.0, 180.0, -1.0, 1.0, 90.0, 1.0)
    global_field = np.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    periodic = compute_sphere_aware_gradients(global_field, grid=global_grid)
    assert np.isfinite(periodic["zonal_gradient"][:, 0]).all()
    assert (periodic["zonal_gradient"][:, 0] < 0.0).all()


def test_combined_outputs_roundtrip_with_metadata(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trajectory": [1, 2, 1, 2],
            "lon": [0.2, 0.3, 1.2, 1.3],
            "lat": [0.2, 0.3, 1.2, 1.3],
            "temp": [1.0, 3.0, 2.0, 4.0],
            "psal": [34.4, 34.6, 34.5, 34.7],
        }
    )
    result = compute_sampled_map(
        df,
        grid=_grid(),
        cfg=_config(
            variables={
                "temp": SampledMapVariableConfig(),
                "psal": SampledMapVariableConfig(),
            },
            gradients=True,
        ),
        variable_metadata={
            "temp": {
                "units": "degree_Celsius",
                "long_name": "Sea temperature",
                "standard_name": "sea_water_temperature",
            },
            "psal": {"units": "psu", "long_name": "Practical salinity"},
        },
    )
    netcdf_path = tmp_path / "sampled_map.nc"
    table_path = tmp_path / "sampled_map_table.parquet"
    save_dataset_netcdf(result.dataset, netcdf_path)
    save_grid_table(result.table, table_path, format="parquet")

    with xr.open_dataset(netcdf_path) as reopened:
        assert reopened.temp_mean.attrs["units"] == "degree_Celsius"
        assert reopened.temp_mean.attrs["standard_name"] == "sea_water_temperature"
        assert reopened.temp_zonal_gradient.attrs["units"] == "degree_Celsius km-1"
        assert reopened.psal_mean.attrs["units"] == "psu"
        np.testing.assert_array_equal(
            reopened.temp_point_count.values,
            result.dataset.temp_point_count.values,
        )
    reopened_table = pd.read_parquet(table_path)
    assert len(reopened_table) == 9
    assert {"temp_mean", "psal_mean", "temp_gradient_magnitude"}.issubset(
        reopened_table.columns
    )


def test_percentile_limits_and_signed_zero_centering() -> None:
    values = np.asarray([[-100.0, -2.0, 0.0, 3.0, 100.0]])
    cfg = SampledMapPlotConfig(percentile_limits=(20.0, 80.0))
    vmin, vmax = resolve_sampled_map_plot_limits(values, plot_cfg=cfg)
    expected_min, expected_max = np.percentile(values, [20.0, 80.0])
    assert vmin == pytest.approx(expected_min)
    assert vmax == pytest.approx(expected_max)

    vmin, vmax = resolve_sampled_map_plot_limits(
        values, plot_cfg=cfg, signed=True
    )
    magnitude = max(abs(expected_min), abs(expected_max))
    assert vmin == pytest.approx(-magnitude)
    assert vmax == pytest.approx(magnitude)

    one_explicit = SampledMapPlotConfig(
        vmin=0.0,
        percentile_limits=(20.0, 80.0),
    )
    assert resolve_sampled_map_plot_limits(
        values, plot_cfg=one_explicit
    ) == pytest.approx((0.0, expected_max))


def test_load_config_parses_generic_sampled_map(tmp_path: Path) -> None:
    config_path = tmp_path / "postprocess.yml"
    config_path.write_text(
        """
dataset:
  input_path: trajectories.zarr
analysis:
  types: [sampled_map]
grid:
  mode: explicit_edges
  lon_min: -180
  lon_max: 180
  lat_min: -80
  lat_max: -30
  dlon: 1
  dlat: 1
sampled_map:
  weighting: trajectories
  max_group_member: 2
  gradients:
    enabled: true
    smoothing_sigma_km: 100
  output:
    save_table: true
    save_netcdf: false
    save_figures: true
  variables:
    temp:
      valid_min: -3
      valid_max: 10
      minimum_point_count: 5
      minimum_trajectory_count: 2
      plotting:
        mean:
          enabled: true
          cmap: rainbow
          vmin: 0
          vmax: 6
          colorbar_label: Temperature mean [degree_Celsius]
        std:
          enabled: true
          cmap: viridis
          vmin: 0
          vmax: null
          percentile_limits: [2, 98]
        smoothed_mean:
          enabled: true
        zonal_gradient:
          enabled: true
          cmap: RdBu_r
          percentile_limits: [2, 98]
        meridional_gradient:
          enabled: true
        gradient_magnitude:
          enabled: true
          cmap: magma
    oxygen:
      valid_min: null
      valid_max: null
      minimum_point_count: 1
      minimum_trajectory_count: 1
""",
        encoding="utf-8",
    )

    cfg = load_postprocess_config(config_path)
    assert cfg.analysis.types == ("sampled_map",)
    assert cfg.sampled_map is not None
    assert cfg.sampled_map.weighting == "trajectories"
    assert cfg.sampled_map.max_group_member == 2
    assert cfg.sampled_map.gradients.smoothing_sigma_km == 100.0
    assert tuple(cfg.sampled_map.variables) == ("temp", "oxygen")
    temp = cfg.sampled_map.variables["temp"]
    assert temp.minimum_point_count == 5
    assert temp.plotting.std.percentile_limits == (2.0, 98.0)
    assert temp.plotting.zonal_gradient.cmap == "RdBu_r"
    assert (
        temp.plotting.mean.colorbar_label
        == "Temperature mean [degree_Celsius]"
    )
    assert cfg.sampled_map.output.save_netcdf is False


@pytest.mark.parametrize(
    "sampled_map_section,match",
    (
        (
            """
  weighting: invalid
  variables: {temp: {}}
""",
            "weighting",
        ),
        (
            """
  gradients: {enabled: true, smoothing_sigma_km: 0}
  variables: {temp: {}}
""",
            "smoothing_sigma_km",
        ),
        (
            """
  variables:
    temp:
      minimum_point_count: 0
""",
            "minimum_point_count",
        ),
        (
            """
  variables:
    temp:
      plotting:
        mean: {enabled: true, percentile_limits: [99, 2]}
""",
            "percentile_limits",
        ),
        (
            """
  variables:
    temp:
      plotting:
        zonal_gradient: {enabled: true}
""",
            "gradients.enabled",
        ),
    ),
)
def test_invalid_sampled_map_config_is_rejected(
    tmp_path: Path,
    sampled_map_section: str,
    match: str,
) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(
        f"""
dataset:
  input_path: trajectories.zarr
sampled_map:
{sampled_map_section}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        load_postprocess_config(config_path)


def test_gradient_product_config_requires_enabled_gradients() -> None:
    plotting = SampledMapVariablePlottingConfig(
        zonal_gradient=SampledMapPlotConfig(enabled=True, cmap="RdBu_r")
    )
    with pytest.raises(ValueError, match="gradients.enabled"):
        SampledMapConfig(
            variables={"temp": SampledMapVariableConfig(plotting=plotting)}
        )
    with pytest.raises(ValueError, match="colorbar_label"):
        SampledMapPlotConfig(colorbar_label="  ")


def test_sampled_map_workflow_writes_combined_products_and_dispatches_plots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import kinematicparcels.postprocessing.workflows.run_sampled_map as workflow

    df = pd.DataFrame(
        {
            "trajectory": [1, 2, 3, 4, 5, 6],
            "obs": [0, 0, 0, 0, 0, 0],
            "time": pd.to_datetime(
                ["2026-01-01"] * 6
            ),
            "lon": [0.2, 1.2, 2.2, 0.2, 1.2, 2.2],
            "lat": [0.2, 0.2, 0.2, 1.2, 1.2, 1.2],
            "temp": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        }
    )
    variable_plotting = SampledMapVariablePlottingConfig(
        mean=SampledMapPlotConfig(
            enabled=True,
            cmap="rainbow",
            percentile_limits=(2.0, 98.0),
            colorbar_label="Custom temperature label",
        ),
        std=SampledMapPlotConfig(enabled=False),
        smoothed_mean=SampledMapPlotConfig(enabled=False),
        zonal_gradient=SampledMapPlotConfig(enabled=True, cmap="RdBu_r"),
        meridional_gradient=SampledMapPlotConfig(enabled=False, cmap="RdBu_r"),
        gradient_magnitude=SampledMapPlotConfig(enabled=False, cmap="magma"),
    )
    sampled_cfg = SampledMapConfig(
        variables={
            "temp": SampledMapVariableConfig(plotting=variable_plotting)
        },
        gradients=SampledMapGradientsConfig(
            enabled=True,
            smoothing_sigma_km=100.0,
        ),
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="unused.zarr"),
        analysis=AnalysisConfig(types=("sampled_map",)),
        output=OutputConfig(output_dir=str(tmp_path)),
        exports=ExportsConfig(table_format="parquet"),
        grid=GridConfig(
            mode="explicit_edges",
            lon_min=0.0,
            lon_max=3.0,
            lat_min=0.0,
            lat_max=3.0,
            dlon=1.0,
            dlat=1.0,
        ),
        plotting=PlottingConfig(projection="PlateCarree"),
        sampled_map=sampled_cfg,
    )
    plot_calls: list[dict] = []
    monkeypatch.setattr(workflow, "get_trajectory_table", lambda cfg, context: df)
    monkeypatch.setattr(
        workflow,
        "plot_grid_map",
        lambda *args, **kwargs: plot_calls.append(kwargs),
    )

    context = {
        "observation_variable_metadata": {
            "temp": {"units": "degree_Celsius", "long_name": "Temperature"}
        }
    }
    workflow.run_sampled_map(cfg, context)

    assert (tmp_path / "sampled_map_table.parquet").exists()
    assert (tmp_path / "sampled_map.nc").exists()
    assert "sampled_map" in context
    assert [call["var_name"] for call in plot_calls] == [
        "temp_mean",
        "temp_zonal_gradient",
    ]
    assert plot_calls[0]["cmap"] == "rainbow"
    assert plot_calls[0]["colorbar_label"] == "Custom temperature label"
    assert plot_calls[1]["cmap"] == "RdBu_r"
    assert plot_calls[1]["colorbar_label"] == "temp zonal gradient [degree_Celsius km-1]"
    assert plot_calls[1]["vmin"] == pytest.approx(-plot_calls[1]["vmax"])
    assert plot_calls[1]["outpath"].name == "sampled_map_temp_zonal_gradient.png"


def test_runner_dispatches_sampled_map(monkeypatch) -> None:
    import kinematicparcels.postprocessing.workflows.run_sampled_map as workflow
    from kinematicparcels.postprocessing.runner.run_postprocessing import (
        _run_single_analysis,
    )

    calls: list[tuple[PostprocessConfig, dict]] = []
    monkeypatch.setattr(
        workflow,
        "run_sampled_map",
        lambda cfg, context: calls.append((cfg, context)),
    )
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="unused.zarr"),
        sampled_map=_config(),
    )
    context: dict = {}
    _run_single_analysis(cfg, "sampled_map", context)
    assert calls == [(cfg, context)]
