from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kinematicparcels.postprocessing.config.models import DatasetCoordinatesConfig
from kinematicparcels.postprocessing.io.parcels import build_trajectory_table, open_parcels_dataset, resolve_parcels_schema
from kinematicparcels.tools.drf_to_zarr import convert_drf_to_dataframe, convert_drf_to_zarr


def _write_drf(path: Path, *, instrument_id: str, rows: list[tuple[float, str, str, float, float, int]]) -> None:
    header = [
        "*2021/12/06 11:39:36.28",
        "*INSTRUMENT",
        "    TYPE                : Oceanetic Measurement",
        "    ID                  : " + instrument_id,
        "*END OF HEADER",
    ]

    body = [f" {record:.1f} {date} {time} {lat:.5f} {lon:.5f} {flag}" for record, date, time, lat, lon, flag in rows]
    content = "\n".join(header + body) + "\n"
    path.write_text(content, encoding="utf-8")


def test_convert_drf_to_zarr_creates_parcels_compatible_dataset(tmp_path: Path) -> None:
    drf_path = tmp_path / "sct0277_20150727_20150815.drf"
    _write_drf(
        drf_path,
        instrument_id="277",
        rows=[
            (1.0, "2015/07/27", "18:21:36", 53.53889, -129.01738, 1),
            (2.0, "2015/07/27", "18:26:36", 53.53818, -129.01799, 1),
            (3.0, "2015/07/27", "18:31:36", 53.53726, -129.01865, 1),
            (4.0, "2015/07/27", "18:36:36", 53.53650, -129.01944, 2),
        ],
    )

    output_path = tmp_path / "drf_output.zarr"
    config = {
        "input": {"drf_files": [str(drf_path)]},
        "output": {"path": str(output_path)},
        "processing": {
            "quality": {
                "keep_at_sea_flags": [1],
            },
            "segment": {
                "mode": "ignore",
                "step_hours": 1.0,
                "tolerance_minutes": 59.0,
            },
            "resample": {
                "enabled": True,
                "frequency": "10min",
                "interpolate": "time",
            },
        },
    }

    convert_drf_to_zarr(config)

    ds = open_parcels_dataset(output_path)
    schema = resolve_parcels_schema(ds, coordinates=DatasetCoordinatesConfig())
    table = build_trajectory_table(ds, schema=schema, extra_vars=["platform_code"])

    assert ds.dims["trajectory"] == 1
    assert ds["platform_code"].dims == ("trajectory",)
    assert ds["platform_code"].values.tolist() == [277]
    assert np.allclose(ds["z"].values[0, :2], [0.0, 0.0])
    assert table["trajectory"].nunique() == 1
    assert table["platform_code"].iloc[0] == 277
    assert pd.Timestamp(table["time"].iloc[0]) == pd.Timestamp("2015-07-27T18:21:36")

    cadence_steps = json.loads(str(ds.attrs["cadence_common_steps_seconds"]))
    assert 300 in cadence_steps
    assert int(ds.attrs["cadence_mode_step_seconds"]) == 300


def test_convert_drf_to_dataframe_reports_mixed_input_cadence(tmp_path: Path) -> None:
    drf_1 = tmp_path / "sct1700_202507271543_202507280310.drf"
    drf_2 = tmp_path / "sct1695_202507271626_202507280151.drf"

    _write_drf(
        drf_1,
        instrument_id="1700",
        rows=[
            (1.0, "2025/07/27", "15:43:00", 53.00000, -129.00000, 1),
            (2.0, "2025/07/27", "15:48:00", 53.00010, -129.00010, 1),
            (3.0, "2025/07/27", "15:53:00", 53.00020, -129.00020, 1),
        ],
    )
    _write_drf(
        drf_2,
        instrument_id="1695",
        rows=[
            (1.0, "2025/07/27", "16:26:00", 53.10000, -129.10000, 1),
            (2.0, "2025/07/27", "16:36:00", 53.10010, -129.10010, 1),
            (3.0, "2025/07/27", "16:46:00", 53.10020, -129.10020, 1),
        ],
    )

    config = {
        "input": {"drf_files": [str(drf_1), str(drf_2)]},
        "output": {"path": str(tmp_path / "mixed.zarr")},
        "processing": {
            "quality": {"keep_at_sea_flags": [1]},
            "segment": {
                "mode": "ignore",
                "step_hours": 1.0,
                "tolerance_minutes": 59.0,
            },
            "resample": {
                "enabled": True,
                "frequency": "10min",
            },
        },
    }

    _, cadence_summary = convert_drf_to_dataframe(config)

    assert cadence_summary["n_trajectories"] == 2
    assert cadence_summary["n_with_deltas"] == 2
    assert 300 in cadence_summary["common_steps_seconds"]
    assert 600 in cadence_summary["common_steps_seconds"]


def test_convert_drf_to_dataframe_filters_at_sea_flags(tmp_path: Path) -> None:
    drf_path = tmp_path / "sct1124_20200803_20200820.drf"
    _write_drf(
        drf_path,
        instrument_id="1124",
        rows=[
            (1.0, "2020/08/03", "00:00:00", 52.00000, -128.00000, 1),
            (2.0, "2020/08/03", "00:05:00", 52.50000, -128.50000, 2),
            (3.0, "2020/08/03", "00:10:00", 52.00040, -128.00040, 1),
        ],
    )

    config = {
        "input": {"drf_files": [str(drf_path)]},
        "output": {"path": str(tmp_path / "at_sea_filter.zarr")},
        "processing": {
            "quality": {"keep_at_sea_flags": [1]},
            "segment": {
                "mode": "ignore",
                "step_hours": 1.0,
                "tolerance_minutes": 59.0,
            },
            "resample": {
                "enabled": True,
                "frequency": "5min",
            },
        },
    }

    trajectories, _ = convert_drf_to_dataframe(config)
    assert len(trajectories) == 1
    assert trajectories[0]["time"].tolist() == [
        pd.Timestamp("2020-08-03T00:00:00"),
        pd.Timestamp("2020-08-03T00:05:00"),
        pd.Timestamp("2020-08-03T00:10:00"),
    ]
    assert np.isclose(float(trajectories[0]["lat"].iloc[1]), 52.0002)
    assert np.isclose(float(trajectories[0]["lon"].iloc[1]), -128.0002)
