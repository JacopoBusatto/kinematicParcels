from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.transition_branches.config import (
    CompactConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
)
from research.transition_branches.fronts import _canonical_fronts
from research.transition_branches.io import STANDARD_TABLES
from research.transition_branches.plotting import (
    CORE_MARKER_STYLES,
    FRONT_MARKER_STYLES,
    _finite_percentile_max,
)
from research.transition_branches.statistics import (
    compact_cell_table,
    compute_transition_statistics,
)


def _config() -> CompactConfig:
    return CompactConfig(
        input=InputConfig("unused.parquet", "synthetic", 30.0),
        output=OutputConfig("unused"),
        grid=GridConfig(
            lon_min=0.0,
            lon_max=3.0,
            lat_min=0.0,
            lat_max=3.0,
            dlon=1.0,
            dlat=1.0,
            periodic_longitude=False,
        ),
    )


def _four_direction_table() -> pd.DataFrame:
    rows = [(1, 2), (2, 1), (1, 0), (0, 1)]
    return pd.DataFrame(
        {
            "start_lon_bin": pd.Series([1] * 4, dtype="int64"),
            "start_lat_bin": pd.Series([1] * 4, dtype="int64"),
            "end_lon_bin": pd.Series([row[0] for row in rows], dtype="int64"),
            "end_lat_bin": pd.Series([row[1] for row in rows], dtype="int64"),
            "start_lon_center": pd.Series([1.5] * 4, dtype="float64"),
            "start_lat_center": pd.Series([1.5] * 4, dtype="float64"),
            "end_lon_center": pd.Series(
                [row[0] + 0.5 for row in rows], dtype="float64"
            ),
            "end_lat_center": pd.Series(
                [row[1] + 0.5 for row in rows], dtype="float64"
            ),
            "transition_count": pd.Series([5] * 4, dtype="int64"),
            "transition_probability": pd.Series([0.25] * 4, dtype="float64"),
        }
    )


def test_compact_workflow_enforces_normalized_source_probability_contract() -> None:
    table = _four_direction_table()
    table.loc[0, "transition_probability"] = 0.20

    with pytest.raises(ValueError, match="row_normalization_failure"):
        compute_transition_statistics(table, _config())


def test_compact_output_preserves_normalized_angular_entropy() -> None:
    statistics = compute_transition_statistics(_four_direction_table(), _config())
    compact = compact_cell_table(statistics.cells)
    source = compact.loc[compact.cell_id.eq(4)].iloc[0]

    assert source.angular_entropy_out == pytest.approx(np.log(4) / np.log(36))
    assert "H_out" not in compact
    assert {"start_lon_bin", "start_lat_bin", "R1_in", "R2_in"} <= set(compact)


def test_compact_defaults_define_only_one_production_realization() -> None:
    config = _config()

    assert config.statistics.min_moving_support == 10
    assert config.branches.transport_percentile == pytest.approx(0.9)
    assert config.branches.ridge_field == "raw"
    assert not config.run_validation
    assert not config.write_debug_outputs


def test_standard_outputs_and_overlay_labels_are_scientist_facing() -> None:
    assert STANDARD_TABLES == (
        "cell_statistics.parquet",
        "branch_cores.parquet",
        "fronts.parquet",
    )
    labels = {
        style[2]
        for style in (*CORE_MARKER_STYLES.values(), *FRONT_MARKER_STYLES.values())
    }
    assert labels == {
        "Current core (two-sided observed)",
        "Current core (one-sided observed)",
        "Left transport front",
        "Right transport front",
    }


def test_structure_map_colormap_percentile_is_configurable_and_validated() -> None:
    config = _config()

    assert config.plotting.structure_map_max_percentile == 100.0
    assert _finite_percentile_max([0.0, 1.0, 2.0, 100.0, np.nan], 100) == 100.0
    assert _finite_percentile_max([0.0, 1.0, 2.0, 100.0, np.nan], 50) == 1.5
    with pytest.raises(ValueError, match="structure_map_max_percentile"):
        replace(
            config,
            plotting=replace(config.plotting, structure_map_max_percentile=0.0),
        )


def test_incoming_statistics_are_always_preserved_for_future_topology() -> None:
    statistics = compute_transition_statistics(_four_direction_table(), _config())
    compact = compact_cell_table(statistics.cells)

    assert {
        "R1_in",
        "R2_in",
        "theta1_in_motion_destination",
        "theta_mu_in_motion_destination",
        "delta_theta_mu1_in",
        "delta_theta_io",
    } <= set(compact)


def test_front_product_preserves_observability_and_physical_quantities() -> None:
    cores = pd.DataFrame(
        {
            "cell_id": [10, 11],
            "component_id": ["component_0001", "component_0001"],
            "lon": [1.5, 2.5],
            "lat": [-40.5, -40.5],
            "ridge_type": ["two_sided", "one_sided"],
            "missing_side": ["none", "right"],
        }
    )
    segment_fronts = pd.DataFrame(
        {
            "cell_id": [10, 10, 11, 11],
            "side": ["left", "left", "left", "right"],
            "candidate_lon": [1.0, 1.2, 2.0, 3.0],
            "candidate_lat": [-40.0, -40.2, -40.0, -40.0],
            "candidate_distance_km": [50.0, 60.0, 70.0, 80.0],
            "absolute_drop": [2.0, 4.0, 5.0, 9.0],
            "relative_drop": [0.2, 0.4, 0.5, 0.9],
        }
    )

    fronts = _canonical_fronts(cores, segment_fronts)
    left = fronts.loc[fronts.core_cell_id.eq(10) & fronts.side.eq("left")].iloc[0]
    missing = fronts.loc[fronts.core_cell_id.eq(11) & fronts.side.eq("right")].iloc[0]
    no_front = fronts.loc[fronts.core_cell_id.eq(10) & fronts.side.eq("right")].iloc[0]

    assert left.front_lon == pytest.approx(1.1)
    assert left.distance_from_core_km == pytest.approx(55.0)
    assert left.transport_loss_km_day == pytest.approx(3.0)
    assert left.front_status == "probable_transport_front"
    assert missing.front_status == "side_not_observable"
    assert no_front.front_status == "observable_no_retained_front"
    assert not missing.observable
    assert np.isnan(missing.front_lon)
