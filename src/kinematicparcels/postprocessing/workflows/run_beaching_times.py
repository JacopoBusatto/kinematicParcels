from __future__ import annotations

from pathlib import Path

from ..analyses import compute_beaching_times
from ..config.models import PostprocessConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_grid_table
from ..plotting import plot_grid_map
from .base_products import get_particle_summary


def run_beaching_times(cfg: PostprocessConfig, context: dict) -> None:
    """
    Beaching-times workflow.
    """
    print("Getting particle summary")
    summary = get_particle_summary(cfg, context)

    print("Building release grid from particle summary")
    grid = build_grid_from_config(
        cfg,
        summary,
        lon_col=cfg.beaching_times.lon_col,
        lat_col=cfg.beaching_times.lat_col,
        time_col="time0",
    )

    print("Computing beaching times")
    grid_table, ds = compute_beaching_times(
        summary,
        grid=grid,
        lon_col=cfg.beaching_times.lon_col,
        lat_col=cfg.beaching_times.lat_col,
        value_col=cfg.beaching_times.value_col,
        agg=cfg.beaching_times.statistic,
        output_col="beaching_time_seconds",
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    table_path = outdir / f"beaching_times_table.{cfg.exports.table_format}"
    nc_path = outdir / "beaching_times.nc"

    print("Saving beaching times table:", table_path)
    save_grid_table(
        grid_table,
        table_path,
        format=cfg.exports.table_format,
    )

    print("Saving beaching times dataset:", nc_path)
    save_dataset_netcdf(ds, nc_path)

    if cfg.beaching_times.plot:
        plot_path = outdir / "beaching_times.png"
        print("Saving beaching times plot:", plot_path)
        ds["beaching_time_days"] = ds["beaching_time_seconds"] / 86400.
        plot_grid_map(
            ds,
            var_name="beaching_time_days",
            outpath=plot_path,
            title="Beaching time",
            projection=cfg.plotting.projection,
        )