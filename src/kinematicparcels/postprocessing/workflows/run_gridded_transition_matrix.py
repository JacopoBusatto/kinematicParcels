from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analyses import compute_gridded_transition_matrix
from ..config.models import PostprocessConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_table
from ..plotting import plot_grid_map
from .base_products import get_trajectory_table


def _has_finite_values(ds, var_name: str) -> bool:
    return bool(np.isfinite(ds[var_name].values).any())


def _format_duration_label(seconds: float) -> str:
    for suffix, unit_seconds in (
        ("d", 86400.0),
        ("h", 3600.0),
        ("m", 60.0),
        ("s", 1.0),
    ):
        value = seconds / unit_seconds
        if np.isclose(value, round(value), rtol=0.0, atol=1.0e-9):
            return f"{int(round(value))}{suffix}"

    value = f"{seconds:.6g}".replace(".", "p").replace("-", "m")
    return f"{value}s"


def _timestep_output_label(ds) -> str:
    timestep_seconds = ds.attrs.get("timestep_seconds", "native")
    if isinstance(timestep_seconds, (int, float, np.integer, np.floating)):
        return f"dt_{_format_duration_label(float(timestep_seconds))}"

    source_timestep_seconds = ds.attrs.get("source_timestep_seconds", "unknown")
    if isinstance(source_timestep_seconds, (int, float, np.integer, np.floating)):
        return f"dt_{_format_duration_label(float(source_timestep_seconds))}"

    return "dt_native"


def run_gridded_transition_matrix(cfg: PostprocessConfig, context: dict) -> None:
    """
    Sparse gridded transition-matrix workflow.
    """
    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    if "grid" not in context:
        print("Building grid")
        context["grid"] = build_grid_from_config(
            cfg,
            df,
            lon_col="lon",
            lat_col="lat",
            time_col="time",
        )

    grid = context["grid"]

    print("Computing gridded transition matrix")
    result = compute_gridded_transition_matrix(
        df,
        grid=grid,
        cfg=cfg.gridded_transition_matrix,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestep_label = _timestep_output_label(result.dataset)

    if cfg.gridded_transition_matrix.output.save_table:
        table_path = (
            outdir
            / f"gridded_transition_matrix_{timestep_label}_table.{cfg.exports.table_format}"
        )
        print("Saving gridded transition matrix table:", table_path)
        save_table(result.transition_table, table_path, format=cfg.exports.table_format)

    if cfg.gridded_transition_matrix.output.save_netcdf:
        nc_path = outdir / f"gridded_transition_matrix_{timestep_label}.nc"
        print("Saving gridded transition matrix dataset:", nc_path)
        save_dataset_netcdf(result.dataset, nc_path)

    if (
        not cfg.gridded_transition_matrix.output.save_figures
        or not cfg.gridded_transition_matrix.plotting.enabled
    ):
        return

    probability_cfg = cfg.gridded_transition_matrix.plotting.probability
    for direction in ("north", "south", "east", "west", "stay"):
        var_name = f"probability_{direction}"
        if not _has_finite_values(result.dataset, var_name):
            continue

        vmin = probability_cfg.vmin
        vmax = probability_cfg.vmax
        ds_plot = result.dataset
        if probability_cfg.as_percent:
            ds_plot = result.dataset.copy(deep=True)
            ds_plot[var_name] = ds_plot[var_name] * 100.0
            if vmin is not None:
                vmin = vmin * 100.0
            if vmax is not None:
                vmax = vmax * 100.0
            colorbar_label = "Probability [%]"
        else:
            colorbar_label = None

        plot_path = (
            outdir
            / f"gridded_transition_probability_{direction}_{timestep_label}.png"
        )
        print(f"Saving {direction} transition probability plot:", plot_path)
        plot_grid_map(
            ds_plot,
            var_name=var_name,
            outpath=plot_path,
            projection=cfg.plotting.projection,
            title=f"Gridded transition probability ({direction})",
            vmin=vmin,
            vmax=vmax,
            cmap=cfg.gridded_transition_matrix.plotting.cmap,
            colorbar_label=colorbar_label,
            title_fontsize=cfg.plotting.title_fontsize,
            colorbar_fontsize=cfg.plotting.colorbar_fontsize,
            colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
            axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
        )
