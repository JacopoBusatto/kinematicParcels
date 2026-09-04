from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pyproj import Geod

from research.transition_branches.config import (
    BranchConfig,
    CartesianGridConfig,
    CompactConfig,
    DirectionalConfig,
    EdgeConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
    PlotConfig,
    SpatialGeometryConfig,
    StatisticsConfig,
    load_config,
)
from research.transition_branches.cores import compute_current_cores
from research.transition_branches.directional_corridors import (
    compute_directional_corridors,
)
from research.transition_branches.geometry import (
    CartesianGeometry,
    GeographicGeometry,
    physical_cell_scales,
)
from research.transition_branches.io import externalize_table
from research.transition_branches.statistics import (
    compute_transition_statistics,
    normalize_transition_table,
)
from research.transition_branches.workflow import run


def _cartesian_config(
    *,
    grid: CartesianGridConfig | None = None,
    length_unit: str = "cm",
    time_unit: str = "s",
    timestep: float = 1.0,
) -> CompactConfig:
    return CompactConfig(
        input=InputConfig("unused.parquet", "tank", timestep, time_unit),
        output=OutputConfig("unused", "tank"),
        geometry=SpatialGeometryConfig("cartesian", length_unit),
        grid=grid or CartesianGridConfig(0.0, 3.0, 0.0, 3.0, 1.0, 1.0),
        statistics=StatisticsConfig(min_moving_support=1),
        plotting=PlotConfig(enabled=False),
    )


def _cartesian_table(grid: CartesianGridConfig) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for y_bin in range(grid.ny):
        for x_bin in range(grid.nx):
            end_x_bin = x_bin + 1 if x_bin + 1 < grid.nx else x_bin - 1
            for selected_end_x, count in ((x_bin, 20), (end_x_bin, 80)):
                records.append(
                    {
                        "start_x_bin": x_bin,
                        "start_y_bin": y_bin,
                        "end_x_bin": selected_end_x,
                        "end_y_bin": y_bin,
                        "start_x_center": grid.x_min + (x_bin + 0.5) * grid.dx,
                        "start_y_center": grid.y_min + (y_bin + 0.5) * grid.dy,
                        "end_x_center": grid.x_min + (selected_end_x + 0.5) * grid.dx,
                        "end_y_center": grid.y_min + (y_bin + 0.5) * grid.dy,
                        "transition_count": count,
                        "transition_probability": count / 100.0,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    for column in (
        "start_x_bin",
        "start_y_bin",
        "end_x_bin",
        "end_y_bin",
        "transition_count",
    ):
        frame[column] = frame[column].astype("int64")
    for column in (
        "start_x_center",
        "start_y_center",
        "end_x_center",
        "end_y_center",
        "transition_probability",
    ):
        frame[column] = frame[column].astype("float64")
    return frame


def _tank_table(grid: CartesianGridConfig) -> pd.DataFrame:
    profile = np.asarray([0.05, 0.10, 0.20, 0.40, 0.65, 0.90, 0.65, 0.40, 0.20, 0.10, 0.05])
    records: list[dict[str, float | int]] = []
    for y_bin in range(grid.ny):
        moving_count = round(100 * profile[y_bin])
        for x_bin in range(grid.nx):
            end_x_bin = x_bin + 1 if x_bin + 1 < grid.nx else x_bin - 1
            for selected_end_x, count in (
                (x_bin, 100 - moving_count),
                (end_x_bin, moving_count),
            ):
                records.append(
                    {
                        "start_x_bin": x_bin,
                        "start_y_bin": y_bin,
                        "end_x_bin": selected_end_x,
                        "end_y_bin": y_bin,
                        "start_x_center": (x_bin + 0.5) * grid.dx,
                        "start_y_center": (y_bin + 0.5) * grid.dy,
                        "end_x_center": (selected_end_x + 0.5) * grid.dx,
                        "end_y_center": (y_bin + 0.5) * grid.dy,
                        "transition_count": count,
                        "transition_probability": count / 100.0,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    integer = [name for name in frame if name.endswith("_bin")]
    integer.append("transition_count")
    frame[integer] = frame[integer].astype("int64")
    floating = [name for name in frame if name.endswith("_center")]
    floating.append("transition_probability")
    frame[floating] = frame[floating].astype("float64")
    return frame


def test_cartesian_distance_bearing_forward_and_unrestricted_coordinates() -> None:
    geometry = CartesianGeometry("cm")
    forward, reverse, distance = geometry.inverse(2.0, 1000.0, 5.0, 1004.0)

    assert distance == pytest.approx(5.0)
    assert forward == pytest.approx(np.degrees(np.arctan2(3.0, 4.0)))
    assert reverse == pytest.approx(forward + 180.0)
    x, y, back = geometry.forward(2.0, 1000.0, forward, distance)
    assert (x, y, back) == pytest.approx((5.0, 1004.0, reverse))


def test_cartesian_rectangular_cell_scales_use_grid_length_unit() -> None:
    grid = CartesianGridConfig(-2.0, 2.0, 100.0, 106.0, 2.0, 3.0)
    cells = pd.DataFrame({"x": [-1.0, 1.0], "y": [101.5, 104.5]})
    x_scale, y_scale, effective = physical_cell_scales(
        cells, grid, CartesianGeometry("cm")
    )

    assert x_scale == pytest.approx([2.0, 2.0])
    assert y_scale == pytest.approx([3.0, 3.0])
    assert effective == pytest.approx([np.sqrt(6.0), np.sqrt(6.0)])


def test_geographic_backend_matches_existing_wgs84_geodesics() -> None:
    geometry = GeographicGeometry("WGS84", "km")
    expected = Geod(ellps="WGS84")
    lon = np.asarray([0.0, 12.5])
    lat = np.asarray([-45.0, -60.0])
    target_lon = np.asarray([1.0, 11.0])
    target_lat = np.asarray([-44.0, -59.5])

    forward, reverse, distance = geometry.inverse(lon, lat, target_lon, target_lat)
    old_forward, old_reverse, old_distance_m = expected.inv(
        lon, lat, target_lon, target_lat
    )
    assert forward == pytest.approx(old_forward)
    assert reverse == pytest.approx(old_reverse)
    assert distance == pytest.approx(old_distance_m / 1000.0)
    new_lon, new_lat, _ = geometry.forward(lon, lat, forward, distance)
    assert new_lon == pytest.approx(target_lon)
    assert new_lat == pytest.approx(target_lat)


@pytest.mark.parametrize("length_unit", ["mm", "cm", "m", "km"])
@pytest.mark.parametrize("time_unit", ["s", "min", "h", "day"])
def test_output_suffixes_cover_every_supported_unit(
    length_unit: str, time_unit: str
) -> None:
    config = _cartesian_config(length_unit=length_unit, time_unit=time_unit)
    table = pd.DataFrame(
        {
            "distance_length": [1.0],
            "speed_rate": [2.0],
            "transport_area_rate": [3.0],
            "gradient_rate_per_length": [4.0],
        }
    )
    output = externalize_table(table, config)

    assert set(output) == {
        f"distance_{length_unit}",
        f"speed_{length_unit}_{time_unit}",
        f"transport_{length_unit}2_{time_unit}",
        f"gradient_{length_unit}_{time_unit}_per_{length_unit}",
    }


def test_geographic_km_day_public_schema_is_preserved() -> None:
    config = CompactConfig(
        input=InputConfig("unused.parquet", "geo", 30.0, "day"),
        output=OutputConfig("unused"),
        geometry=SpatialGeometryConfig("geographic", "km", "WGS84"),
        grid=GridConfig(-180.0, 180.0, -80.0, -30.0, 1.0, 1.0, True),
        plotting=PlotConfig(enabled=False),
    )
    output = externalize_table(
        pd.DataFrame(
            {
                "start_x_bin": [0],
                "x": [-179.5],
                "y": [-79.5],
                "U_out_all_x_rate": [1.0],
                "U_out_all_y_rate": [2.0],
                "distance_from_core_length": [3.0],
            }
        ),
        config,
    )

    assert set(output) == {
        "start_lon_bin",
        "lon",
        "lat",
        "U_out_all_east_km_day",
        "U_out_all_north_km_day",
        "distance_from_core_km",
    }


def test_geographic_and_cartesian_input_names_are_mode_specific() -> None:
    cartesian = _cartesian_table(CartesianGridConfig(0, 3, 0, 3, 1, 1))
    normalized = normalize_transition_table(cartesian, "cartesian")
    assert set(normalized) == set(cartesian)
    with pytest.raises(ValueError, match="start_lon_bin"):
        normalize_transition_table(cartesian, "geographic")

    geographic = cartesian.rename(
        columns={
            "start_x_bin": "start_lon_bin",
            "start_y_bin": "start_lat_bin",
            "end_x_bin": "end_lon_bin",
            "end_y_bin": "end_lat_bin",
            "start_x_center": "start_lon_center",
            "start_y_center": "start_lat_center",
            "end_x_center": "end_lon_center",
            "end_y_center": "end_lat_center",
        }
    )
    assert set(normalize_transition_table(geographic, "geographic")) == set(cartesian)
    with pytest.raises(ValueError, match="start_x_bin"):
        normalize_transition_table(geographic, "cartesian")


def test_cartesian_statistics_are_unit_equivalent_in_metres_and_centimetres() -> None:
    metre_grid = CartesianGridConfig(0.0, 3.0, 0.0, 3.0, 1.0, 1.0)
    cm_grid = CartesianGridConfig(0.0, 300.0, 0.0, 300.0, 100.0, 100.0)
    metre_config = _cartesian_config(grid=metre_grid, length_unit="m")
    centimetre_config = _cartesian_config(grid=cm_grid, length_unit="cm")
    metre = compute_transition_statistics(
        _cartesian_table(metre_grid), metre_config
    ).cells.sort_values("cell_id")
    centimetre = compute_transition_statistics(
        _cartesian_table(cm_grid), centimetre_config
    ).cells.sort_values("cell_id")

    for field in ("P_move", "R1_out", "R2_out", "theta1_out", "D_out_all_magnitude"):
        assert centimetre[field].to_numpy() == pytest.approx(
            metre[field].to_numpy()
        )
    assert centimetre.U_out_all_magnitude_rate.to_numpy() == pytest.approx(
        100.0 * metre.U_out_all_magnitude_rate.to_numpy()
    )
    metre_cores = compute_current_cores(metre, metre_config)
    centimetre_cores = compute_current_cores(centimetre, centimetre_config)
    assert set(metre_cores.cores.cell_id) == set(centimetre_cores.cores.cell_id)
    metre_corridors = compute_directional_corridors(metre, metre_config)
    centimetre_corridors = compute_directional_corridors(
        centimetre, centimetre_config
    )
    assert set(metre_corridors.corridors.cell_id) == set(
        centimetre_corridors.corridors.cell_id
    )


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _raw_cartesian_config() -> dict:
    return {
        "input": {
            "transition_table": "matrix.parquet",
            "matrix_id": "tank",
            "timestep": 1.0,
            "time_unit": "s",
        },
        "output": {"root": "outputs"},
        "geometry": {"coordinate_system": "cartesian", "length_unit": "cm"},
        "grid": {
            "x_min": 0.0,
            "x_max": 3.0,
            "y_min": 0.0,
            "y_max": 3.0,
            "dx": 1.0,
            "dy": 1.0,
        },
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: raw["input"].update(timestep_days=1.0), "unknown input keys"),
        (lambda raw: raw.update(ellipsoid="WGS84"), "unknown configuration keys"),
        (
            lambda raw: raw.update(
                statistics={"direction_zero_tolerance_km": 1.0e-12}
            ),
            "unknown statistics keys",
        ),
        (
            lambda raw: raw.update(
                branches={"ridge_comparison_tolerance_km_day": 1.0e-12}
            ),
            "unknown branches keys",
        ),
        (
            lambda raw: raw.update(validation={"center_atol_degrees": 1.0e-9}),
            "unknown validation keys",
        ),
        (
            lambda raw: raw.update(plotting={"vector_reference_km_day": 5.0}),
            "unknown plotting keys",
        ),
        (lambda raw: raw["geometry"].update(crs="EPSG:32632"), "unknown geometry keys"),
        (lambda raw: raw["geometry"].update(ellipsoid="WGS84"), "must not define"),
        (lambda raw: raw["geometry"].update(length_unit="inch"), "unsupported length_unit"),
        (lambda raw: raw["input"].update(time_unit="week"), "unsupported time_unit"),
        (lambda raw: raw["grid"].update(lon_min=0.0), "unknown grid keys"),
        (lambda raw: raw["grid"].update(x_max=3.2), "integer multiple"),
        (lambda raw: raw.update(unexpected=True), "unknown configuration keys"),
    ],
)
def test_strict_config_rejects_old_unknown_mixed_and_invalid_values(
    tmp_path: Path, mutate, match: str
) -> None:
    raw = _raw_cartesian_config()
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        load_config(_write_config(tmp_path, raw))


def test_end_to_end_tank_run_writes_tables_manifest_debug_and_seven_figures(
    tmp_path: Path,
) -> None:
    grid = CartesianGridConfig(0.0, 11.0, 0.0, 11.0, 1.0, 1.0)
    input_path = tmp_path / "tank.parquet"
    _tank_table(grid).to_parquet(input_path, index=False)
    config = replace(
        _cartesian_config(grid=grid),
        input=InputConfig(str(input_path), "tank_e2e", 1.0, "s"),
        output=OutputConfig(str(tmp_path / "outputs"), "tank"),
        branches=BranchConfig(transport_percentile=0.9),
        directional=DirectionalConfig(
            minimum_P_move=0.5,
            minimum_R1=0.8,
            minimum_strength=0.5,
            minimum_component_cells=3,
        ),
        edges=EdgeConfig(half_width_grid_scales=5),
        plotting=PlotConfig(enabled=True, draw_coastlines=False, dpi=40),
        write_debug_outputs=True,
        run_validation=True,
    )

    run_dir = run(config)
    figures = sorted((run_dir / "figures").glob("*.png"))
    assert len(figures) == 7
    for name in (
        "cell_statistics.parquet",
        "branch_cores.parquet",
        "fronts.parquet",
        "directional_corridors.parquet",
        "directional_fronts.parquet",
        "gradient_validation.parquet",
        "raw_cross_sections.parquet",
        "directional_raw_cross_sections.parquet",
        "resolved_config.yaml",
        "manifest.json",
    ):
        assert (run_dir / name).is_file()

    cells = pd.read_parquet(run_dir / "cell_statistics.parquet")
    cores = pd.read_parquet(run_dir / "branch_cores.parquet")
    fronts = pd.read_parquet(run_dir / "fronts.parquet")
    corridors = pd.read_parquet(run_dir / "directional_corridors.parquet")
    directional_fronts = pd.read_parquet(run_dir / "directional_fronts.parquet")
    components = pd.read_parquet(run_dir / "component_graph_details.parquet")
    gradients = pd.read_parquet(run_dir / "gradient_validation.parquet")
    sections = pd.read_parquet(run_dir / "raw_cross_sections.parquet")
    assert {"x", "y", "U_out_all_magnitude_cm_s"} <= set(cells)
    assert len(cores) > 0
    assert fronts.front_detected.any()
    assert len(corridors) > 0
    assert directional_fronts.front_detected.any()
    assert "integrated_transport_cm2_s" in components
    assert "G_perp_at_flank_cm_s_per_cm" in gradients
    assert {"d_from_refined_core_cm", "U_parallel_raw_cm_s"} <= set(sections)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    for metadata in (manifest["geometry"], resolved["resolved_geometry"]):
        assert metadata["coordinate_system"] == "cartesian"
        assert metadata["coordinate_unit"] == "cm"
        assert metadata["length_unit"] == "cm"
        assert metadata["time_unit"] == "s"
        assert metadata["rate_unit"] == "cm/s"
        assert metadata["geometry_backend"] == "Euclidean planar"
        assert "0=+y" in metadata["bearing_convention"]
    assert load_config(run_dir / "resolved_config.yaml").geometry_metadata == manifest[
        "geometry"
    ]
