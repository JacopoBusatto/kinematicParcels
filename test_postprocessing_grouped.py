from __future__ import annotations

import pandas as pd

from kinematicparcels.postprocessing.analyses.density import compute_time_density
from kinematicparcels.postprocessing.core.gridding import RegularGrid
from kinematicparcels.postprocessing.core.summaries import build_particle_summary


def test_build_particle_summary_separates_group_members() -> None:
    df = pd.DataFrame(
        {
            "trajectory": [10, 10, 10, 10],
            "group_member": [1, 1, 2, 2],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                    "2026-04-15T00:00:00",
                    "2026-04-15T06:00:00",
                ]
            ),
            "lon": [12.00, 12.10, 12.20, 12.30],
            "lat": [37.00, 37.05, 37.20, 37.25],
        }
    )

    summary = build_particle_summary(df)

    assert len(summary) == 2
    assert set(summary["group_member"]) == {1, 2}
    assert set(summary["lat0"]) == {37.0, 37.2}


def test_compute_time_density_accepts_pandas_timestamps() -> None:
    df = pd.DataFrame(
        {
            "trajectory": ["1_m1", "1_m1", "1_m2", "1_m2"],
            "group_member": [1, 1, 2, 2],
            "obs": [0, 1, 0, 1],
            "time": pd.to_datetime(
                [
                    "2026-04-15T12:00:00",
                    "2026-04-15T18:00:00",
                    "2026-04-15T12:00:00",
                    "2026-04-15T18:00:00",
                ]
            ),
            "lon": [12.0, 12.1, 12.2, 12.3],
            "lat": [37.0, 37.0, 37.1, 37.1],
        }
    )

    grid = RegularGrid(
        lon_min=11.9,
        lon_max=12.4,
        lat_min=36.9,
        lat_max=37.2,
        dlon=0.1,
        dlat=0.1,
    )

    table, ds = compute_time_density(df, grid=grid)

    assert len(table) == 4
    assert ds.sizes["time"] == 2
    assert float(table["particle_count"].sum()) == 4.0
