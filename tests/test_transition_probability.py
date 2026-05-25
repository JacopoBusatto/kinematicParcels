from __future__ import annotations

import pandas as pd
import pytest

from kinematicparcels.postprocessing.analyses.transition_probability import compute_transition_probability
from kinematicparcels.postprocessing.config import load_postprocess_config
from kinematicparcels.postprocessing.config.models import TransitionProbabilityConfig
from kinematicparcels.regions import Region, RegionManager


def test_load_postprocess_config_parses_transition_probability_section(tmp_path) -> None:
    cfg_path = tmp_path / "postprocess_transition_probability.yml"
    cfg_path.write_text(
        """
        dataset:
          input_path: ./dummy.zarr
        analysis:
          types:
            - transition_probability
        transition_probability:
          region_labels:
            - sesc-mod
            - sesc-sir
          time_frequency: 3
          how_many: priority_max
          priority_level: 7
          priority_mode: exact
          input_lon_mode: "-180_180"
          max_group_member: 2
          filter_isolated: true
        """,
        encoding="utf-8",
    )

    cfg = load_postprocess_config(cfg_path)

    assert cfg.analysis.types == ("transition_probability",)
    assert cfg.transition_probability.region_labels == ("sesc-mod", "sesc-sir")
    assert cfg.transition_probability.time_frequency == 3
    assert cfg.transition_probability.priority_level == 7
    assert cfg.transition_probability.max_group_member == 2
    assert cfg.transition_probability.filter_isolated is True


def _region_manager(priority_r2: int = 1) -> RegionManager:
    return RegionManager(
        [
            Region(
                name="Region 1",
                label="r1",
                numericLabel=1,
                polygons=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
                priority=1,
            ),
            Region(
                name="Region 2",
                label="r2",
                numericLabel=2,
                polygons=[[(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]],
                priority=priority_r2,
            ),
        ]
    )


def _base_cfg(**kwargs) -> TransitionProbabilityConfig:
    return TransitionProbabilityConfig(
        region_labels=("r1", "r2"),
        **kwargs,
    )


def _trajectory_rows(
    trajectory: str,
    coords: list[tuple[float, float]],
    *,
    group_member: int | None = None,
 ) -> list[dict]:
    times = pd.date_range("2026-01-01", periods=len(coords), freq="1D")
    rows: list[dict] = []
    for obs, ((lon, lat), time) in enumerate(zip(coords, times, strict=True)):
        row = {
            "trajectory": trajectory,
            "obs": obs,
            "time": time,
            "lon": lon,
            "lat": lat,
        }
        if group_member is not None:
            row["group_member"] = group_member
        rows.append(row)
    return rows


def test_compute_transition_probability_counts_and_excludes_outside_starts() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (0.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (2.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("c", [(5.0, 5.0), (2.5, 0.5), (2.5, 0.5)])
        + _trajectory_rows("d", [(2.5, 0.5), (2.5, 0.5), (0.5, 0.5)])
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(),
        region_manager=_region_manager(),
    )

    assert result["time"].tolist() == list(pd.date_range("2026-01-01", periods=3, freq="1D"))
    assert result["p_r1__r1"].tolist() == [1.0, 0.5, 0.0]
    assert result["p_r1__r2"].tolist() == [0.0, 0.5, 1.0]
    assert result["p_r2__r1"].tolist() == [0.0, 0.0, 1.0]
    assert result["p_r2__r2"].tolist() == [1.0, 1.0, 0.0]


def test_compute_transition_probability_filter_isolated_reclassifies_single_symbol() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (2.5, 0.5), (0.5, 0.5)])
        + _trajectory_rows("b", [(0.6, 0.5), (0.6, 0.5), (0.6, 0.5)])
    )

    no_filter = compute_transition_probability(
        df,
        cfg=_base_cfg(filter_isolated=False),
        region_manager=_region_manager(),
    )
    filtered = compute_transition_probability(
        df,
        cfg=_base_cfg(filter_isolated=True),
        region_manager=_region_manager(),
    )

    assert no_filter.loc[1, "p_r1__r2"] == 0.5
    assert filtered.loc[1, "p_r1__r2"] == 0.0
    assert filtered.loc[1, "p_r1__r1"] == 1.0


def test_compute_transition_probability_supports_group_member_filter_and_stride() -> None:
    df = pd.DataFrame(
        _trajectory_rows("g1", [(0.5, 0.5), (0.5, 0.5), (2.5, 0.5)], group_member=1)
        + _trajectory_rows("g1", [(2.5, 0.5), (2.5, 0.5), (0.5, 0.5)], group_member=2)
    )

    result = compute_transition_probability(
        df,
        cfg=_base_cfg(time_frequency=2, max_group_member=1),
        region_manager=_region_manager(),
    )

    assert result["time"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")]
    assert result["p_r1__r1"].tolist() == [1.0, 0.0]
    assert result["p_r1__r2"].tolist() == [0.0, 1.0]
    assert result["p_r2__r1"].isna().all()
    assert result["p_r2__r2"].isna().all()


def test_compute_transition_probability_warns_on_mixed_priorities() -> None:
    df = pd.DataFrame(
        _trajectory_rows("a", [(0.5, 0.5), (0.5, 0.5)])
        + _trajectory_rows("b", [(2.5, 0.5), (2.5, 0.5)])
    )

    with pytest.warns(UserWarning, match="priority"):
        compute_transition_probability(
            df,
            cfg=_base_cfg(),
            region_manager=_region_manager(priority_r2=2),
        )