from __future__ import annotations

from pathlib import Path

from ..config.models import PostprocessConfig
from ..plotting import plot_trajectories_map
from .base_products import get_trajectory_table


def run_trajectories(cfg: PostprocessConfig, context: dict) -> None:
    """
    Trajectory plotting workflow.
    """
    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.trajectories.plot:
        plot_path = outdir / "trajectories.png"
        print("Saving trajectories plot:", plot_path)
        plot_trajectories_map(
            df,
            outpath=plot_path,
            projection=cfg.plotting.projection,
            title=cfg.trajectories.title,
            show_start=cfg.trajectories.show_start,
            show_end=cfg.trajectories.show_end,
        )