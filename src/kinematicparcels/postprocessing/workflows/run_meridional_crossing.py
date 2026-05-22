from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analyses import compute_meridional_crossing
from ..config.models import PostprocessConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_grid_table
from ..plotting import plot_grid_map
from .base_products import get_trajectory_table


def _has_finite_values(ds, var_name: str) -> bool:
    return bool(np.isfinite(ds[var_name].values).any())


def run_meridional_crossing(cfg: PostprocessConfig, context: dict) -> None:
    """
    Directional meridional crossing workflow.
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

    print("Computing meridional crossing diagnostic")
    result = compute_meridional_crossing(
        df,
        grid=grid,
        cfg=cfg.meridional_crossing,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.meridional_crossing.output.save_grid_table:
        table_path = outdir / f"meridional_crossing_table.{cfg.exports.table_format}"
        print("Saving meridional crossing table:", table_path)
        save_grid_table(result.grid_table, table_path, format=cfg.exports.table_format)

    if cfg.meridional_crossing.output.save_netcdf:
        nc_path = outdir / "meridional_crossing.nc"
        print("Saving meridional crossing dataset:", nc_path)
        save_dataset_netcdf(result.dataset, nc_path)

    if not cfg.meridional_crossing.output.save_figures or not cfg.meridional_crossing.plotting.enabled:
        return

    if cfg.meridional_crossing.plotting.show_probability:
        for direction in ("northward", "southward"):
            var_name = f"crossing_probability_{direction}"
            if not _has_finite_values(result.dataset, var_name):
                continue
            plot_path = outdir / f"meridional_crossing_probability_{direction}.png"
            print(f"Saving {direction} crossing probability plot:", plot_path)
            plot_grid_map(
                result.dataset,
                var_name=var_name,
                outpath=plot_path,
                projection=cfg.plotting.projection,
                title=f"Meridional crossing probability ({direction})",
            )

    if cfg.meridional_crossing.plotting.show_counts:
        for direction in ("northward", "southward"):
            var_name = f"crossing_count_{direction}"
            if not _has_finite_values(result.dataset, var_name):
                continue
            plot_path = outdir / f"meridional_crossing_count_{direction}.png"
            print(f"Saving {direction} crossing count plot:", plot_path)
            plot_grid_map(
                result.dataset,
                var_name=var_name,
                outpath=plot_path,
                projection=cfg.plotting.projection,
                title=f"Meridional crossing count ({direction})",
            )