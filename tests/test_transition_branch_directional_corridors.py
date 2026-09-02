from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.transition_branches.comparison import (
    compare_transport_and_directional_structures,
)
from research.transition_branches.config import (
    CompactConfig,
    DirectionalConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
)
from research.transition_branches.cores import CoreSolution
from research.transition_branches.directional_corridors import (
    compute_directional_corridors,
)
from research.transition_branches.directional_fronts import (
    compute_probable_directional_fronts,
)


def _config(nlon: int = 7, nlat: int = 7) -> CompactConfig:
    return CompactConfig(
        input=InputConfig("unused.parquet", "synthetic", 10.0),
        output=OutputConfig("unused"),
        grid=GridConfig(
            lon_min=0.0,
            lon_max=float(nlon),
            lat_min=-3.5,
            lat_max=-3.5 + nlat,
            dlon=1.0,
            dlat=1.0,
            periodic_longitude=False,
        ),
        directional=DirectionalConfig(
            minimum_P_move=0.5,
            minimum_R1=0.8,
            minimum_strength=0.5,
            maximum_neighbor_direction_difference_degrees=45.0,
            maximum_step_direction_mismatch_degrees=45.0,
            minimum_component_cells=3,
            transverse_scale_grid=1.0,
        ),
    )


def _cells(config: CompactConfig) -> pd.DataFrame:
    lat_bin, lon_bin = np.indices((config.grid.nlat, config.grid.nlon))
    size = lat_bin.size
    theta = np.full(size, 90.0)
    r1 = np.full(size, 0.2)
    p_move = np.ones(size)
    strength = p_move * r1
    return pd.DataFrame(
        {
            "cell_id": (lat_bin * config.grid.nlon + lon_bin).ravel(),
            "lon_bin": lon_bin.ravel(),
            "lat_bin": lat_bin.ravel(),
            "lon": (
                config.grid.lon_min + (lon_bin + 0.5) * config.grid.dlon
            ).ravel(),
            "lat": (
                config.grid.lat_min + (lat_bin + 0.5) * config.grid.dlat
            ).ravel(),
            "N_out_move": np.full(size, 20),
            "U_out_all_magnitude_km_day": strength,
            "P_move": p_move,
            "R1_out": r1,
            "theta1_out": theta,
            "D_out_all_east": strength * np.sin(np.deg2rad(theta)),
            "D_out_all_north": strength * np.cos(np.deg2rad(theta)),
            "D_out_all_magnitude": strength,
        }
    )


def _set_directional_cells(
    cells: pd.DataFrame,
    config: CompactConfig,
    prescribed: dict[tuple[int, int], float],
    *,
    strength: float = 0.85,
) -> pd.DataFrame:
    output = cells.copy()
    for (lon_bin, lat_bin), theta in prescribed.items():
        cell_id = lat_bin * config.grid.nlon + lon_bin
        mask = output.cell_id.eq(cell_id)
        output.loc[mask, "R1_out"] = strength
        output.loc[mask, "theta1_out"] = theta
        output.loc[mask, "D_out_all_east"] = strength * np.sin(np.deg2rad(theta))
        output.loc[mask, "D_out_all_north"] = strength * np.cos(np.deg2rad(theta))
        output.loc[mask, "D_out_all_magnitude"] = strength
    return output


def test_one_cell_wide_corridor_is_retained_with_two_observable_sides() -> None:
    config = _config()
    prescribed = {(lon, 3): 90.0 for lon in range(1, 6)}
    result = compute_directional_corridors(
        _set_directional_cells(_cells(config), config, prescribed), config
    )

    assert len(result.components) == 1
    assert len(result.corridors) == 5
    assert result.corridors.corridor_observability.eq("two_sided").all()
    assert result.summary["two_sided_corridor_cells"] == 5


def test_smoothly_turning_local_directions_form_one_curved_corridor() -> None:
    config = _config()
    prescribed = {
        (6, 2): 270.0,
        (5, 2): 280.0,
        (4, 2): 300.0,
        (3, 3): 330.0,
        (3, 4): 0.0,
        (3, 5): 0.0,
    }
    result = compute_directional_corridors(
        _set_directional_cells(_cells(config), config, prescribed), config
    )

    assert len(result.components) == 1
    assert set(result.corridors.cell_id) == {
        lat * config.grid.nlon + lon for lon, lat in prescribed
    }
    assert result.components.iloc[0].maximum_neighbor_direction_difference_degrees <= 30
    assert result.components.iloc[0].latitude_span_degrees == 3.0


def test_opposite_directions_do_not_connect_despite_spatial_adjacency() -> None:
    config = _config()
    config = replace(
        config,
        directional=replace(config.directional, minimum_component_cells=2),
    )
    prescribed = {(2, 3): 90.0, (3, 3): 270.0}
    result = compute_directional_corridors(
        _set_directional_cells(_cells(config), config, prescribed), config
    )

    assert result.corridors.empty
    assert result.components.empty
    assert result.summary["discarded_short_components"] == 2


def test_configured_support_threshold_controls_directional_eligibility() -> None:
    config = _config()
    config = replace(
        config,
        directional=replace(config.directional, minimum_component_cells=2),
    )
    prescribed = {(lon, 3): 90.0 for lon in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    cells.loc[cells.lon_bin.eq(3) & cells.lat_bin.eq(3), "N_out_move"] = 9
    result = compute_directional_corridors(cells, config)

    assert result.summary["support_threshold"] == 10
    assert 3 * config.grid.nlon + 3 not in set(result.corridors.cell_id)
    assert sorted(result.components.n_cells) == [2, 2]


def test_two_cell_wide_directional_band_is_not_rejected_by_width() -> None:
    config = _config()
    prescribed = {
        (lon, lat): 90.0 for lat in (3, 4) for lon in range(1, 6)
    }
    result = compute_directional_corridors(
        _set_directional_cells(_cells(config), config, prescribed), config
    )

    assert len(result.corridors) == 10
    assert result.corridors.corridor_observability.eq("two_sided").all()


def test_narrow_corridor_produces_persistent_two_sided_directional_fronts() -> None:
    config = _config()
    prescribed = {(lon, 3): 90.0 for lon in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    corridors = compute_directional_corridors(cells, config)
    result = compute_probable_directional_fronts(cells, corridors, config)

    assert result.summary["sections"] == 5
    assert result.summary["sections_with_two_retained_fronts"] == 5
    assert result.summary["probable_directional_fronts"] == 10
    assert result.fronts.front_status.eq("probable_directional_front").all()
    assert result.fronts.absolute_directional_drop.gt(0.6).all()


def test_cross_section_projects_onto_each_central_local_direction() -> None:
    config = _config()
    prescribed = {(lon, 3): 90.0 for lon in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    north = cells.lat_bin.eq(4)
    cells.loc[north, "theta1_out"] = 270.0
    cells.loc[north, "D_out_all_east"] = -cells.loc[
        north, "D_out_all_magnitude"
    ]
    cells.loc[north, "D_out_all_north"] = 0.0
    corridors = compute_directional_corridors(cells, config)
    result = compute_probable_directional_fronts(cells, corridors, config)
    center_id = 3 * config.grid.nlon + 3
    section = result.cross_sections.loc[
        result.cross_sections.corridor_cell_id.eq(center_id)
    ]
    axis = section.loc[section.offset_index_from_corridor_cell.eq(0)].iloc[0]
    left = section.loc[section.offset_index_from_corridor_cell.eq(-1)].iloc[0]

    assert axis.theta1_out_center == 90.0
    assert axis.D_parallel_raw == pytest.approx(0.85)
    assert left.D_parallel_raw < 0.0


def test_missing_support_is_unobservable_and_never_a_directional_drop() -> None:
    config = _config()
    prescribed = {(lon, 3): 90.0 for lon in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    north = cells.lat_bin.eq(4)
    cells.loc[north, "N_out_move"] = 0
    cells.loc[
        north,
        ["D_out_all_east", "D_out_all_north", "D_out_all_magnitude"],
    ] = np.nan
    corridors = compute_directional_corridors(cells, config)
    result = compute_probable_directional_fronts(cells, corridors, config)
    left = result.fronts.loc[result.fronts.side.eq("left")]

    assert left.front_status.eq("side_not_observable").all()
    assert not left.front_detected.any()
    assert left.absolute_directional_drop.isna().all()


def test_transport_directional_comparison_does_not_match_component_geometry() -> None:
    config = _config()
    prescribed = {(lon, 3): 90.0 for lon in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    directional = compute_directional_corridors(cells, config)
    transport_ids = [3 * config.grid.nlon + 1, 3 * config.grid.nlon + 2, 0]
    transport = CoreSolution(
        cores=pd.DataFrame(
            {
                "cell_id": transport_ids,
                "component_id": ["transport_a", "transport_a", "transport_b"],
            }
        ),
        components=pd.DataFrame(),
        segment_members=pd.DataFrame(),
        segments=pd.DataFrame(),
        threshold_km_day=1.0,
        selection_label="q90",
    )
    result = compare_transport_and_directional_structures(
        cells, transport, directional, config
    )

    assert result.summary["transport_and_directional"] == 2
    assert result.summary["directional_only"] == 3
    assert result.summary["transport_only"] == 1
    assert result.summary["neither"] == 43
    assert not result.summary["component_matching_performed"]
    assert set(result.components.structure_type) == {"transport", "directional"}
