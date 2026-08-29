from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..analyses import compute_alive_latitude_fraction
from ..config.models import PostprocessConfig
from ..io import save_table
from ..plotting import plot_alive_latitude_fraction
from .base_products import get_trajectory_table


def _long_form_table(ds: xr.Dataset) -> pd.DataFrame:
    axis_name = "time" if ds.attrs.get("time_axis") == "time" else "age_days"
    columns = [
        axis_name,
        "latitude_bin",
        "lat_lower",
        "lat_center",
        "lat_upper",
        "latitude_bin_count",
        "alive_tracer_count",
        "alive_tracer_fraction",
        "meets_minimum_alive",
    ]
    if ds.sizes.get(axis_name, 0) == 0:
        return pd.DataFrame(columns=columns)
    return ds.to_dataframe().reset_index()[columns]


def run_alive_latitude_fraction(cfg: PostprocessConfig, context: dict) -> None:
    """Run and export the alive latitude fraction diagnostic."""
    print("Getting trajectory table")
    trajectory_table = get_trajectory_table(cfg, context)

    print("Computing alive latitude fraction")
    result = compute_alive_latitude_fraction(
        trajectory_table,
        cfg=cfg.alive_latitude_fraction,
    )
    context["alive_latitude_fraction"] = result

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.alive_latitude_fraction.output.save_csv:
        table_path = outdir / "alive_latitude_fraction.csv"
        print("Saving alive latitude fraction table:", table_path)
        save_table(_long_form_table(result), table_path, format="csv")

    if not cfg.alive_latitude_fraction.output.save_figure:
        return

    if not np.isfinite(result["alive_tracer_fraction"].values).any():
        warnings.warn(
            "No alive_latitude_fraction values meet the minimum tracer support; "
            "the PNG heatmap will not be written.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    plotting = cfg.alive_latitude_fraction.plotting
    figure_path = outdir / "alive_latitude_fraction.png"
    print("Saving alive latitude fraction heatmap:", figure_path)
    plot_alive_latitude_fraction(
        result,
        outpath=figure_path,
        cmap=plotting.cmap,
        vmin=plotting.vmin,
        vmax=plotting.vmax,
        min_mask_value=plotting.min_mask_value,
        as_percent=plotting.as_percent,
        masked_color=plotting.masked_color,
        title_fontsize=cfg.plotting.title_fontsize,
        colorbar_fontsize=cfg.plotting.colorbar_fontsize,
        colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
        axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
    )
