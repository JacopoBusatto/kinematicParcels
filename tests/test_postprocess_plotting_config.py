from __future__ import annotations

from textwrap import dedent

import numpy as np
import pytest
import xarray as xr

from kinematicparcels.postprocessing.config.loader import load_postprocess_config
from kinematicparcels.postprocessing.plotting.masking import mask_values_below
from kinematicparcels.postprocessing.workflows.snapshots import resolve_snapshot_indices


def test_load_postprocess_config_parses_meridional_crossing_plotting_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_meridional.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - meridional_crossing
            grid:
              mode: explicit_edges
              lon_min: 10.0
              lon_max: 15.0
              lat_min: 0.0
              lat_max: 4.0
              dlon: 1.0
              dlat: 1.0
            meridional_crossing:
              direction: both
              segmentation:
                lat_filter: rolling_mean
                filter_window: 5
                direction_threshold_deg: auto
                min_segment_duration_days: 1.5
                min_segment_displacement_deg: auto
                valid_if: duration_or_displacement
              crossing:
                crossing_latitude_reference: center
                count_once_per_segment_per_lat_bin: true
              output:
                save_netcdf: true
                save_grid_table: true
                save_figures: false
              plotting:
                enabled: true
                probability:
                  enabled: true
                  vmin: 0.0
                  vmax: 1.0
                  as_percent: true
                count:
                  enabled: false
                  vmin: 0.0
                  vmax: 10.0
            plotting:
              title_fontsize: 15
              colorbar_fontsize: 13
              colorbar_tick_fontsize: 11
              axis_tick_fontsize: 12
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("meridional_crossing",)
    assert cfg.meridional_crossing.direction == "both"
    assert cfg.meridional_crossing.segmentation.lat_filter == "rolling_mean"
    assert cfg.meridional_crossing.segmentation.direction_threshold_deg == "auto"
    assert cfg.meridional_crossing.crossing.crossing_latitude_reference == "center"
    assert cfg.meridional_crossing.output.save_figures is False
    assert cfg.meridional_crossing.plotting.probability.enabled is True
    assert cfg.meridional_crossing.plotting.probability.vmin == 0.0
    assert cfg.meridional_crossing.plotting.probability.vmax == 1.0
    assert cfg.meridional_crossing.plotting.probability.as_percent is True
    assert cfg.meridional_crossing.plotting.count.enabled is False
    assert cfg.meridional_crossing.plotting.count.vmax == 10.0
    assert cfg.plotting.title_fontsize == 15
    assert cfg.plotting.colorbar_fontsize == 13
    assert cfg.plotting.colorbar_tick_fontsize == 11
    assert cfg.plotting.axis_tick_fontsize == 12


def test_load_postprocess_config_parses_beaching_times_plotting_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_beaching.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - beaching_times
            grid:
              mode: explicit_edges
              lon_min: 10.0
              lon_max: 15.0
              lat_min: 0.0
              lat_max: 4.0
              dlon: 1.0
              dlat: 1.0
            beaching_times:
              infer_grid_from_start: false
              lon_col: lon0
              lat_col: lat0
              value_col: lifetime_seconds
              statistic: min
              plotting:
                enabled: true
                vmin: 0.0
                vmax: 365.0
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("beaching_times",)
    assert cfg.beaching_times.infer_grid_from_start is False
    assert cfg.beaching_times.plotting.enabled is True
    assert cfg.beaching_times.plotting.vmin == 0.0
    assert cfg.beaching_times.plotting.vmax == 365.0


def test_load_postprocess_config_parses_density_snapshot_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_density_snapshots.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - density
            grid:
              mode: explicit_edges
              lon_min: 10.0
              lon_max: 15.0
              lat_min: 0.0
              lat_max: 4.0
              dlon: 1.0
              dlat: 1.0
            density:
              animation_var: particle_count
              animation_label: particle_count
              animation_vmin: null
              animation_vmax: 5.0
              min_mask_value: 0.01
              plot_snaps: true
              timestep_snaps: [0, 6, -1]
            """
        ),
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.density.plot_snaps is True
    assert cfg.density.timestep_snaps == (0, 6, -1)
    assert cfg.density.animation_var == "particle_count"
    assert cfg.density.animation_vmin is None
    assert cfg.density.animation_vmax == 5.0
    assert cfg.density.min_mask_value == 0.01


def test_load_postprocess_config_rejects_density_snapshots_without_indices(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_density_snapshots_invalid.yml"
    cfg_path.write_text(
        dedent(
            """
            dataset:
              input_path: ./dummy.zarr
            analysis:
              types:
                - density
            density:
              plot_snaps: true
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="density.timestep_snaps"):
        load_postprocess_config(cfg_path)


def test_resolve_snapshot_indices_supports_density_negative_indices() -> None:
    assert resolve_snapshot_indices((0, -1), n_times=5, config_name="density") == (0, 4)


def test_mask_values_below_turns_low_values_into_nan() -> None:
    da = xr.DataArray([0.0, 0.2, 1.0], dims=("x",))

    masked = mask_values_below(da, 0.2)

    assert np.isnan(masked.values[0])
    assert masked.values[1] == 0.2
    assert masked.values[2] == 1.0
    assert da.values[0] == 0.0
