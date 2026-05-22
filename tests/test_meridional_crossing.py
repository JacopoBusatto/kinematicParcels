from __future__ import annotations

import pandas as pd

from kinematicparcels.postprocessing.analyses.meridional_crossing import compute_meridional_crossing
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import (
    DatasetConfig,
    GridConfig,
    MeridionalCrossingConfig,
    MeridionalCrossingCrossingConfig,
    MeridionalCrossingPlottingConfig,
    MeridionalCrossingSegmentationConfig,
    OutputConfig,
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
        lon_min=10.0,
        lon_max=15.0,
        lat_min=0.0,
        lat_max=4.0,
        dlon=1.0,
        dlat=1.0,
    )


def _base_cfg(**kwargs) -> MeridionalCrossingConfig:
    segmentation = kwargs.pop(
        "segmentation",
        MeridionalCrossingSegmentationConfig(
            lat_filter="none",
            filter_window=1,
            direction_threshold_deg=0.0,
            min_segment_duration_days=0.0,
            min_segment_displacement_deg=0.0,
        ),
    )
    crossing = kwargs.pop(
        "crossing",
        MeridionalCrossingCrossingConfig(crossing_latitude_reference="center"),
    )
    plotting = kwargs.pop("plotting", MeridionalCrossingPlottingConfig())
    return MeridionalCrossingConfig(
        segmentation=segmentation,
        crossing=crossing,
        plotting=plotting,
        **kwargs,
    )


def test_meridional_crossing_counts_simple_northward_track() -> None:
    df = _build_trajectory(
        "north",
        lon=[10.0, 11.0, 12.0, 13.0],
        lat=[0.1, 0.9, 1.9, 2.9],
    )

    result = compute_meridional_crossing(df, grid=_base_grid(), cfg=_base_cfg())

    assert int(result.dataset["n_segments_northward"].item()) == 1
    assert int(result.dataset["n_segments_southward"].item()) == 0
    assert float(result.dataset["crossing_count_northward"].sel(lat=0.5, lon=10.5)) == 1.0
    assert float(result.dataset["crossing_count_northward"].sel(lat=1.5, lon=11.5)) == 1.0
    assert float(result.dataset["crossing_count_northward"].sel(lat=2.5, lon=12.5)) == 1.0
    assert float(result.dataset["crossing_probability_northward"].sel(lat=1.5, lon=11.5)) == 1.0


def test_meridional_crossing_counts_simple_southward_track() -> None:
    df = _build_trajectory(
        "south",
        lon=[13.0, 12.0, 11.0, 10.0],
        lat=[2.9, 1.9, 0.9, 0.1],
    )

    result = compute_meridional_crossing(df, grid=_base_grid(), cfg=_base_cfg())

    assert int(result.dataset["n_segments_northward"].item()) == 0
    assert int(result.dataset["n_segments_southward"].item()) == 1
    assert float(result.dataset["crossing_count_southward"].sel(lat=2.5, lon=12.5)) == 1.0
    assert float(result.dataset["crossing_count_southward"].sel(lat=1.5, lon=11.5)) == 1.0
    assert float(result.dataset["crossing_count_southward"].sel(lat=0.5, lon=10.5)) == 1.0
    assert float(result.dataset["crossing_probability_southward"].sel(lat=1.5, lon=11.5)) == 1.0


def test_meridional_crossing_zonal_track_produces_no_directional_segments() -> None:
    df = _build_trajectory(
        "zonal",
        lon=[10.0, 11.0, 12.0, 13.0],
        lat=[0.60, 0.62, 0.61, 0.63],
    )

    result = compute_meridional_crossing(
        df,
        grid=_base_grid(),
        cfg=_base_cfg(
            segmentation=MeridionalCrossingSegmentationConfig(
                lat_filter="none",
                filter_window=1,
                direction_threshold_deg=0.25,
                min_segment_duration_days=0.0,
                min_segment_displacement_deg=0.0,
            )
        ),
    )

    assert int(result.dataset["n_segments_northward"].item()) == 0
    assert int(result.dataset["n_segments_southward"].item()) == 0
    assert result.grid_table.empty


def test_meridional_crossing_small_meanders_count_each_lat_bin_once() -> None:
    df = _build_trajectory(
        "meander",
        lon=[10.0, 11.0, 12.0, 13.0],
        lat=[0.1, 1.6, 1.4, 2.2],
    )

    result = compute_meridional_crossing(
        df,
        grid=_base_grid(),
        cfg=_base_cfg(
            segmentation=MeridionalCrossingSegmentationConfig(
                lat_filter="rolling_mean",
                filter_window=3,
                direction_threshold_deg=0.0,
                min_segment_duration_days=0.0,
                min_segment_displacement_deg=0.0,
            )
        ),
    )

    assert int(result.dataset["n_segments_northward"].item()) == 1
    assert result.grid_table["crossing_count_northward"].sum() == 2.0
    assert float(result.dataset["crossing_count_northward"].sel(lat=1.5, lon=10.5)) == 1.0


def test_meridional_crossing_fast_short_segment_passes_displacement_filter() -> None:
    df = _build_trajectory(
        "fast",
        lon=[10.0, 12.0],
        lat=[0.0, 2.2],
        step_hours=12.0,
    )

    result = compute_meridional_crossing(
        df,
        grid=_base_grid(),
        cfg=_base_cfg(
            segmentation=MeridionalCrossingSegmentationConfig(
                lat_filter="none",
                filter_window=1,
                direction_threshold_deg=0.0,
                min_segment_duration_days=1.5,
                min_segment_displacement_deg=1.0,
            )
        ),
    )

    assert int(result.dataset["n_segments_northward"].item()) == 1
    assert result.grid_table["crossing_count_northward"].sum() == 2.0


def test_meridional_crossing_stationary_track_produces_no_valid_segments() -> None:
    df = _build_trajectory(
        "stationary",
        lon=[10.0, 10.0, 10.0],
        lat=[1.0, 1.0, 1.0],
    )

    result = compute_meridional_crossing(df, grid=_base_grid(), cfg=_base_cfg())

    assert int(result.dataset["n_segments_northward"].item()) == 0
    assert int(result.dataset["n_segments_southward"].item()) == 0
    assert result.grid_table.empty


def test_meridional_crossing_invalid_short_segments_leave_probability_masked() -> None:
    df = pd.concat(
        [
            _build_trajectory(
                "north_short",
                lon=[10.0, 10.8],
                lat=[0.0, 0.4],
                step_hours=12.0,
            ),
            _build_trajectory(
                "south_short",
                lon=[11.0, 10.2],
                lat=[2.0, 1.6],
                step_hours=12.0,
            ),
        ],
        ignore_index=True,
    )

    result = compute_meridional_crossing(
        df,
        grid=_base_grid(),
        cfg=_base_cfg(
            segmentation=MeridionalCrossingSegmentationConfig(
                lat_filter="none",
                filter_window=1,
                direction_threshold_deg=0.0,
                min_segment_duration_days=1.5,
                min_segment_displacement_deg=1.0,
            )
        ),
    )

    assert int(result.dataset["n_segments_northward"].item()) == 0
    assert int(result.dataset["n_segments_southward"].item()) == 0
    assert result.dataset["crossing_probability_northward"].isnull().all().item()
    assert result.dataset["crossing_probability_southward"].isnull().all().item()


def test_load_postprocess_config_parses_meridional_crossing_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_meridional.yml"
    cfg_path.write_text(
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
            show_probability: true
            show_counts: false
        """,
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("meridional_crossing",)
    assert cfg.meridional_crossing.direction == "both"
    assert cfg.meridional_crossing.segmentation.lat_filter == "rolling_mean"
    assert cfg.meridional_crossing.segmentation.direction_threshold_deg == "auto"
    assert cfg.meridional_crossing.crossing.crossing_latitude_reference == "center"
    assert cfg.meridional_crossing.output.save_figures is False


def test_meridional_crossing_result_metadata_records_release_dependence() -> None:
    cfg = PostprocessConfig(
        dataset=DatasetConfig(input_path="dummy.zarr"),
        output=OutputConfig(output_dir="./outputs/test"),
        grid=GridConfig(
            mode="explicit_edges",
            lon_min=10.0,
            lon_max=15.0,
            lat_min=0.0,
            lat_max=4.0,
            dlon=1.0,
            dlat=1.0,
        ),
    )
    df = _build_trajectory(
        "north",
        lon=[10.0, 11.0, 12.0, 13.0],
        lat=[0.1, 0.9, 1.9, 2.9],
    )

    result = compute_meridional_crossing(df, grid=_base_grid(), cfg=cfg.meridional_crossing)

    assert "release-dependent" in result.dataset.attrs["summary"]
    assert result.dataset.attrs["lat_filter"] == "rolling_mean"


def test_meridional_crossing_dataset_writes_to_netcdf(tmp_path) -> None:
    df = _build_trajectory(
        "north",
        lon=[10.0, 11.0, 12.0, 13.0],
        lat=[0.1, 0.9, 1.9, 2.9],
    )

    result = compute_meridional_crossing(df, grid=_base_grid(), cfg=_base_cfg())

    outpath = tmp_path / "meridional_crossing.nc"
    save_dataset_netcdf(result.dataset, outpath)

    assert outpath.exists()
    assert outpath.stat().st_size > 0