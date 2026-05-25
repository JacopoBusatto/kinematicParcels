from __future__ import annotations

from pathlib import Path

from ..analyses import compute_transition_probability
from ..config.models import PostprocessConfig
from ..io import save_table
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