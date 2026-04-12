from __future__ import annotations

from pathlib import Path

from ..animations import animate_trajectories
from ..config.models import PostprocessConfig
from ..plotting import plot_trajectories_map
from .base_products import get_particle_summary, get_trajectory_table


def run_trajectories(cfg: PostprocessConfig, context: dict) -> None:
    """
    Trajectory plotting workflow.
    """
    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    summary = None
    if cfg.trajectories.animate:
        try:
            summary = get_particle_summary(cfg, context)
        except Exception:
            summary = None

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
            max_group_member=cfg.trajectories.max_group_member,
        )

    if cfg.trajectories.animate:
        gif_path = outdir / "trajectories.gif"
        print("Saving trajectories animation:", gif_path)
        animate_trajectories(
            df,
            outpath=gif_path,
            projection=cfg.plotting.projection,
            fps=cfg.trajectories.animation_fps,
            title=cfg.trajectories.title,
            color_by=cfg.trajectories.animation_color_by,
            colorbar_label=cfg.trajectories.animation_label,
            vmin=cfg.trajectories.animation_vmin,
            vmax=cfg.trajectories.animation_vmax,
            show_time_bar=cfg.trajectories.show_time_bar,
            trail=cfg.trajectories.trail,
            trail_steps=cfg.trajectories.trail_steps,
            summary_df=summary,
            max_group_member=cfg.trajectories.max_group_member,
        )