from __future__ import annotations

from textwrap import dedent

from kinematicparcels.postprocessing.config.loader import load_postprocess_config


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
