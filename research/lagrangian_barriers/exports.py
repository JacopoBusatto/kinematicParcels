from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr


def save_dataset_netcdf(dataset: xr.Dataset, path: Path) -> None:
    """Write an eager dataset without probing an unused Dask Distributed client.

    Recent xarray releases ask Dask which scheduler is active even when every
    variable is an in-memory NumPy array.  Importing ``distributed`` during
    that probe can fail for reasons unrelated to NetCDF output (for example a
    malformed Windows certificate).  The analysis datasets are deliberately
    eager, so scheduler discovery is unnecessary and is disabled only for the
    duration of this write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(variable.chunks is not None for variable in dataset.variables.values()):
        dataset.to_netcdf(path, engine="netcdf4")
        return

    try:
        from xarray.backends import locks, writers
    except ImportError:  # Compatibility with xarray versions predating writers.py.
        dataset.to_netcdf(path, engine="netcdf4")
        return

    with (
        patch.object(writers, "get_dask_scheduler", return_value=None),
        patch.object(locks, "get_dask_scheduler", return_value=None),
    ):
        dataset.to_netcdf(path, engine="netcdf4")


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _split_antimeridian(coordinates: Iterable[tuple[float, float]]) -> list[list[list[float]]]:
    coords = [(float(((lon + 180) % 360) - 180), float(lat)) for lon, lat in coordinates]
    if not coords:
        return []
    parts: list[list[list[float]]] = [[[coords[0][0], coords[0][1]]]]
    for previous, current in zip(coords[:-1], coords[1:]):
        if abs(current[0] - previous[0]) > 180:
            parts.append([])
        parts[-1].append([current[0], current[1]])
    return [part for part in parts if len(part) >= 2]


def line_geojson(
    frame: pd.DataFrame, group_columns: list[str], path: Path,
    *, robust_only: bool = False,
) -> None:
    features = []
    work = frame
    if work.empty or any(column not in work for column in [*group_columns, "lon", "lat"]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": []}, indent=2), encoding="utf-8")
        return
    if robust_only and "robust_segment" in work:
        work = work.loc[work.robust_segment]
    for keys, group in work.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        group = group.sort_values("point_order" if "point_order" in group else "barrier_point_order")
        parts = _split_antimeridian(zip(group.lon, group.lat))
        if not parts: continue
        props = {name: value for name, value in zip(group_columns, keys)}
        features.append({
            "type": "Feature", "properties": props,
            "geometry": {"type": "MultiLineString", "coordinates": parts},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def save_graphml(graph: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def save_cross_sections_netcdf(frame: pd.DataFrame, path: Path) -> None:
    numeric = frame.select_dtypes(include=[np.number, bool]).copy()
    ds = xr.Dataset({column: (("cross_section",), numeric[column].to_numpy()) for column in numeric},
                    coords={"cross_section": np.arange(len(frame), dtype=np.int64)})
    ds.attrs["representation"] = "one record per branch point and requested offset"
    save_dataset_netcdf(ds, path)
