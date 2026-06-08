from __future__ import annotations

from pathlib import Path

from ..analyses import compute_transition_probability
from ..config.models import PostprocessConfig
from ..io import save_table
from ..plotting import plot_transition_probability_by_source, plot_transition_probability_overview
from .base_products import get_trajectory_table


def run_transition_probability(cfg: PostprocessConfig, context: dict) -> None:
    """
    Transition-probability workflow.
    """
    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    print("Computing transition probability matrix")
    transition_table = compute_transition_probability(
        df,
        cfg=cfg.transition_probability,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / "transition_probability.csv"
    print("Saving transition probability table:", outpath)
    save_table(transition_table, outpath, format="csv")

    if cfg.transition_probability.plotting.enabled:
        print("Saving transition probability plots")
        plot_transition_probability_overview(
            transition_table,
            region_labels=list(cfg.transition_probability.region_labels),
            outpath=outdir / "transition_probability_plot.png",
            x_log_scale=cfg.transition_probability.plotting.x_log_scale,
            y_log_scale=cfg.transition_probability.plotting.y_log_scale,
            colormap=cfg.transition_probability.plotting.colormap,
        )
        plot_transition_probability_by_source(
            transition_table,
            region_labels=list(cfg.transition_probability.region_labels),
            outdir=outdir,
            x_log_scale=cfg.transition_probability.plotting.x_log_scale,
            y_log_scale=cfg.transition_probability.plotting.y_log_scale,
            colormap=cfg.transition_probability.plotting.colormap,
        )