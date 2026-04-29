from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.models import PostprocessConfig
from ..core import build_particle_summary
from ..io import load_trajectory_table, open_parcels_dataset


def _candidate_extra_vars(available_vars: set[str]) -> list[str]:
    candidate_extra_vars = [
        "circle_id",
        "group_id",
        "group_member",
        "group_size",
        "lon_1",
        "lat_1",
        "lon_2",
        "lat_2",
        "lon_3",
        "lat_3",
        "lon_4",
        "lat_4",
    ]
    return [v for v in candidate_extra_vars if v in available_vars]


def _expand_memberwise_rows(df: pd.DataFrame) -> pd.DataFrame:
    has_group_member = "group_member" in df.columns
    has_member_columns = ("lon_1" in df.columns) and ("lat_1" in df.columns)
    if has_group_member or (not has_member_columns):
        return df

    member_chunks: list[pd.DataFrame] = []
    for m in (1, 2, 3, 4):
        lon_col = f"lon_{m}"
        lat_col = f"lat_{m}"
        if lon_col not in df.columns or lat_col not in df.columns:
            continue

        chunk = df.copy()
        if "group_size" in chunk.columns:
            chunk = chunk[chunk["group_size"] >= m]

        if chunk.empty:
            continue

        chunk["source_trajectory"] = chunk["trajectory"]
        chunk["trajectory"] = chunk["trajectory"].astype(str) + f"_m{m}"
        chunk["lon"] = chunk[lon_col]
        chunk["lat"] = chunk[lat_col]
        chunk["group_member"] = m
        member_chunks.append(chunk)

    if not member_chunks:
        return df

    out = pd.concat(member_chunks, ignore_index=True)
    drop_cols = [
        c
        for c in ("lon_1", "lat_1", "lon_2", "lat_2", "lon_3", "lat_3", "lon_4", "lat_4")
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out.sort_values(["trajectory", "group_member", "obs"]).reset_index(drop=True)


def _table_path(
    cfg: PostprocessConfig,
    name: str,
) -> Path:
    return Path(cfg.output.output_dir) / f"{name}.{cfg.exports.table_format}"


def _load_exported_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format for '{path}'")


def get_trajectory_table(
    cfg: PostprocessConfig,
    context: dict,
) -> pd.DataFrame:
    """
    Return the trajectory table from:
    1. context
    2. exported file
    3. fresh computation
    """
    if "trajectory_table" in context:
        return context["trajectory_table"]

    path = _table_path(cfg, "trajectory_table")

    ds = open_parcels_dataset(cfg.dataset.input_path)
    try:
        available_vars = set(str(v) for v in ds.variables)
    finally:
        ds.close()

    extra_vars = _candidate_extra_vars(available_vars)

    if path.exists():
        df = _expand_memberwise_rows(_load_exported_table(path))
        if ("group_member" in df.columns) or (not extra_vars):
            context["trajectory_table"] = df
            return df

    df = load_trajectory_table(
        cfg.dataset.input_path,
        truncate_stagnant=cfg.cleaning.truncate_stagnant,
        stagnant_tol=cfg.cleaning.stagnant_tol,
        stagnant_min_consecutive=cfg.cleaning.stagnant_min_consecutive,
        extra_vars=extra_vars,
    )
    df = _expand_memberwise_rows(df)

    context["trajectory_table"] = df
    return df


def get_particle_summary(
    cfg: PostprocessConfig,
    context: dict,
) -> pd.DataFrame:
    """
    Return the particle summary from:
    1. context
    2. exported file
    3. fresh computation
    """
    if "particle_summary" in context:
        return context["particle_summary"]

    path = _table_path(cfg, "particle_summary")
    if path.exists():
        df = _load_exported_table(path)
        context["particle_summary"] = df
        return df

    traj = get_trajectory_table(cfg, context)
    summary = build_particle_summary(traj)
    context["particle_summary"] = summary
    return summary
