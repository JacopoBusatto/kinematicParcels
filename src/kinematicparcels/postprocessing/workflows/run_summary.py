from __future__ import annotations

from pathlib import Path

from ..config.models import PostprocessConfig
from ..core import build_particle_summary
from ..io import save_particle_summary, save_trajectory_table
from .base_products import get_trajectory_table


def run_summary(cfg: PostprocessConfig, context: dict) -> None:
    """
    Build and optionally save the base products:
    - trajectory_table
    - particle_summary

    This is the only workflow allowed to persist these products.
    """
    print("Building trajectory table")

    trajectory_table = get_trajectory_table(cfg, context)

    print("Building particle summary")
    particle_summary = build_particle_summary(trajectory_table)
    context["particle_summary"] = particle_summary

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.exports.save_trajectory_table:
        traj_path = outdir / f"trajectory_table.{cfg.exports.table_format}"
        print("Saving trajectory table:", traj_path)
        save_trajectory_table(
            trajectory_table,
            traj_path,
            format=cfg.exports.table_format,
        )

    if cfg.exports.save_particle_summary:
        summary_path = outdir / f"particle_summary.{cfg.exports.table_format}"
        print("Saving particle summary:", summary_path)
        save_particle_summary(
            particle_summary,
            summary_path,
            format=cfg.exports.table_format,
        )