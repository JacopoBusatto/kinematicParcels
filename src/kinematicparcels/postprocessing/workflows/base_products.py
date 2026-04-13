from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.models import PostprocessConfig
from ..core import build_particle_summary
from ..io import load_trajectory_table


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
    if path.exists():
        df = _load_exported_table(path)
        context["trajectory_table"] = df
        return df

    df = load_trajectory_table(
        cfg.dataset.input_path,
        truncate_stagnant=cfg.cleaning.truncate_stagnant,
        stagnant_tol=cfg.cleaning.stagnant_tol,
        stagnant_min_consecutive=cfg.cleaning.stagnant_min_consecutive,
        extra_vars=['group_id', 'group_member', 'group_size'],
    )
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