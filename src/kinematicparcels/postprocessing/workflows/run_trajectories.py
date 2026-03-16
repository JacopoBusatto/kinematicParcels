from __future__ import annotations

from pathlib import Path

from ..config.models import PostprocessConfig
from ..io import load_trajectory_table, save_trajectory_table
from ..plotting import plot_trajectories_map


def run_trajectories(cfg: PostprocessConfig, context: dict) -> None:
    """
    Trajectory plotting workflow.
    """
    if "trajectory_table" not in context:
        print("Loading trajectory table")

        df = load_trajectory_table(
            cfg.dataset.input_path,
            truncate_stagnant=cfg.cleaning.truncate_stagnant,
            stagnant_tol=cfg.cleaning.stagnant_tol,
            stagnant_min_consecutive=cfg.cleaning.stagnant_min_consecutive,
        )
        context["trajectory_table"] = df
    else:
        df = context["trajectory_table"]

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.exports.save_trajectory_table:
        traj_path = outdir / f"trajectory_table.{cfg.exports.table_format}"
        print("Saving trajectory table:", traj_path)
        save_trajectory_table(
            df,
            traj_path,
            format=cfg.exports.table_format,
        )

    if cfg.trajectories.plot:
        plot_path = outdir / "trajectories.png"
        print("Saving trajectories plot:", plot_path)
        plot_trajectories_map(
            df,
            outpath=plot_path,
            title=cfg.trajectories.title,
            show_start=cfg.trajectories.show_start,
            show_end=cfg.trajectories.show_end,
            projection=cfg.plotting.projection,
        )