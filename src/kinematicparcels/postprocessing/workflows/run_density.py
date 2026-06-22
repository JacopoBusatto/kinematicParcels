from __future__ import annotations

from pathlib import Path

from ..analyses import compute_time_density
from ..animations import animate_density
from ..config.models import PostprocessConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_grid_table
from .base_products import get_trajectory_table
from .snapshots import save_gridded_snapshots


def run_density(cfg: PostprocessConfig, context: dict) -> None:
    """
    Density workflow.
    """
    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    if "grid" not in context:
        print("Building grid")
        grid = build_grid_from_config(
            cfg,
            df,
            lon_col=cfg.density.lon_col,
            lat_col=cfg.density.lat_col,
            time_col=cfg.density.time_col,
        )
        context["grid"] = grid

    grid = context["grid"]

    if cfg.density.group_member is not None and "group_member" in df.columns:
        df = df.loc[df["group_member"] == cfg.density.group_member].copy()

    print("Computing density")

    density_table, density_ds = compute_time_density(
        df,
        grid=grid,
        lon_col=cfg.density.lon_col,
        lat_col=cfg.density.lat_col,
        time_col=cfg.density.time_col,
        normalize_active=cfg.density.normalize_active,
        normalize_total=cfg.density.normalize_total,
        fill_ever_active_empty_with_zero=cfg.density.fill_ever_active_empty_with_zero
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid_table_path = outdir / f"density_table.{cfg.exports.table_format}"
    density_nc_path = outdir / "density.nc"

    print("Saving density table:", grid_table_path)
    save_grid_table(
        density_table,
        grid_table_path,
        format=cfg.exports.table_format,
    )

    print("Saving density dataset:", density_nc_path)
    save_dataset_netcdf(density_ds, density_nc_path)

    if cfg.density.plot_snaps:
        save_gridded_snapshots(
            density_ds,
            var_name=cfg.density.animation_var,
            timestep_snaps=cfg.density.timestep_snaps,
            config_name="density",
            outdir=outdir,
            filename_prefix="density",
            title_prefix=cfg.density.animation_var,
            projection=cfg.plotting.projection,
            vmin=cfg.density.animation_vmin,
            vmax=cfg.density.animation_vmax,
            min_mask_value=cfg.density.min_mask_value,
            colorbar_label=cfg.density.animation_label,
            title_fontsize=cfg.plotting.title_fontsize,
            colorbar_fontsize=cfg.plotting.colorbar_fontsize,
            colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
            axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
        )

    if cfg.density.animate:
        gif_path = outdir / "density.gif"
        print("Saving density animation:", gif_path)

        animate_density(
            density_ds,
            var_name=cfg.density.animation_var,
            outpath=gif_path,
            projection=cfg.plotting.projection,
            fps=cfg.density.animation_fps,
            every_n=cfg.density.animation_every_n,
            title=cfg.density.animation_var,
            colorbar_label=cfg.density.animation_label,
            vmin=cfg.density.animation_vmin,
            vmax=cfg.density.animation_vmax,
            min_mask_value=cfg.density.min_mask_value,
            show_time_bar=cfg.density.show_time_bar,
        )
