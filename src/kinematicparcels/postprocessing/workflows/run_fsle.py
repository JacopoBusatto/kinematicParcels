from __future__ import annotations

from pathlib import Path

from ..analyses import compute_fsle
from ..config.models import PostprocessConfig
from ..io import save_table
from ..plotting import plot_fsle_spectrum
from .base_products import get_trajectory_table


def run_fsle(cfg: PostprocessConfig, context: dict) -> None:
    """
    FSLE workflow.
    """
    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    print("Computing FSLE")
    result = compute_fsle(
        df,
        pair_mode=cfg.fsle.pair_mode,
        min_scale=cfg.fsle.min_scale,
        max_scale=cfg.fsle.max_scale,
        rho_increment=cfg.fsle.rho_increment,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    spectrum_path = outdir / f"fsle_spectrum_table.{cfg.exports.table_format}"
    print("Saving FSLE spectrum table:", spectrum_path)
    save_table(result.spectrum, spectrum_path, format=cfg.exports.table_format)

    if cfg.fsle.save_crossing_events:
        events_path = outdir / f"fsle_crossing_events.{cfg.exports.table_format}"
        print("Saving FSLE crossing-event table:", events_path)
        save_table(result.crossing_events, events_path, format=cfg.exports.table_format)

    if cfg.fsle.plot and not result.spectrum.empty:
        plot_path = outdir / "fsle_spectrum.png"
        print("Saving FSLE plot:", plot_path)
        plot_fsle_spectrum(
            result.spectrum,
            outpath=plot_path,
            reference_slopes=cfg.fsle.reference_slopes,
            reference_slope_anchor_scales=cfg.fsle.reference_slope_anchor_scales,
            x_min=cfg.fsle.x_min,
            x_max=cfg.fsle.x_max,
            y_min=cfg.fsle.y_min,
            y_max=cfg.fsle.y_max,
        )