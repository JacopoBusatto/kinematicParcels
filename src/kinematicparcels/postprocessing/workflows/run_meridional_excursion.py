from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analyses import compute_meridional_excursion
from ..config.models import (
    MeridionalExcursionVariablePlotConfig,
    PostprocessConfig,
)
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_grid_table, save_table
from ..plotting import plot_grid_map, plot_point_map
from .base_products import get_trajectory_table


_ANCHOR_COLUMNS = {
    "initial_position": ("lon0", "lat0"),
    "southmost_point": ("lon_at_lat_min", "lat_min"),
    "northmost_point": ("lon_at_lat_max", "lat_max"),
}


def _has_finite_values(ds, var_name: str) -> bool:
    return bool(np.isfinite(ds[var_name].values).any())


def _plot_variable_configs(cfg) -> dict[str, MeridionalExcursionVariablePlotConfig]:
    configured = cfg.meridional_excursion.plotting.variables
    if configured:
        return configured

    return {
        variable: MeridionalExcursionVariablePlotConfig(
            over=cfg.meridional_excursion.gridding.over
        )
        for variable in cfg.meridional_excursion.gridding.variables
    }


def run_meridional_excursion(cfg: PostprocessConfig, context: dict) -> None:
    """
    Meridional-excursion workflow.
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

    print("Computing meridional excursion diagnostic")
    result = compute_meridional_excursion(
        df,
        grid=grid,
        cfg=cfg.meridional_excursion,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.meridional_excursion.output.save_table:
        table_path = outdir / f"meridional_excursion_table.{cfg.exports.table_format}"
        print("Saving meridional excursion table:", table_path)
        save_table(result.table, table_path, format=cfg.exports.table_format)

    if cfg.meridional_excursion.output.save_grid_table:
        grid_table_path = outdir / f"meridional_excursion_grid_table.{cfg.exports.table_format}"
        print("Saving meridional excursion grid table:", grid_table_path)
        save_grid_table(
            result.grid_table,
            grid_table_path,
            format=cfg.exports.table_format,
        )

    if cfg.meridional_excursion.output.save_netcdf:
        nc_path = outdir / "meridional_excursion.nc"
        print("Saving meridional excursion dataset:", nc_path)
        save_dataset_netcdf(result.dataset, nc_path)

    if not cfg.meridional_excursion.output.save_figures or not cfg.meridional_excursion.plotting.enabled:
        return

    plot_types = set(cfg.meridional_excursion.plotting.type)
    variable_configs = _plot_variable_configs(cfg)
    merge = cfg.meridional_excursion.gridding.merge

    for variable, plot_cfg in variable_configs.items():
        anchors = plot_cfg.over or cfg.meridional_excursion.gridding.over

        for anchor in anchors:
            if anchor not in _ANCHOR_COLUMNS:
                raise ValueError(f"Unsupported meridional excursion plotting anchor: {anchor!r}")

            if "gridded" in plot_types:
                var_name = f"{variable}_at_{anchor}_{merge}"
                if var_name not in result.dataset or not _has_finite_values(result.dataset, var_name):
                    continue
                plot_path = outdir / f"meridional_excursion_gridded_{var_name}.png"
                print("Saving meridional excursion gridded plot:", plot_path)
                plot_grid_map(
                    result.dataset,
                    var_name=var_name,
                    outpath=plot_path,
                    projection=cfg.plotting.projection,
                    title=plot_cfg.title or "",
                    vmin=plot_cfg.vmin,
                    vmax=plot_cfg.vmax,
                    cmap=plot_cfg.cmap,
                    colorbar_label=plot_cfg.cbar_label,
                    title_fontsize=cfg.plotting.title_fontsize,
                    colorbar_fontsize=cfg.plotting.colorbar_fontsize,
                    colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
                    axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
                )

            if "scatter" in plot_types:
                lon_col, lat_col = _ANCHOR_COLUMNS[anchor]
                if variable not in result.table.columns:
                    raise KeyError(f"Meridional excursion table missing plot variable: {variable!r}")
                finite_points = result.table[[lon_col, lat_col, variable]].dropna()
                if finite_points.empty:
                    continue
                plot_path = outdir / f"meridional_excursion_scatter_{variable}_at_{anchor}.png"
                print("Saving meridional excursion scatter plot:", plot_path)
                plot_point_map(
                    result.table,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    value_col=variable,
                    outpath=plot_path,
                    projection=cfg.plotting.projection,
                    title=plot_cfg.title or "",
                    vmin=plot_cfg.vmin,
                    vmax=plot_cfg.vmax,
                    cmap=plot_cfg.cmap,
                    colorbar_label=plot_cfg.cbar_label,
                    title_fontsize=cfg.plotting.title_fontsize,
                    colorbar_fontsize=cfg.plotting.colorbar_fontsize,
                    colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
                    axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
                )
