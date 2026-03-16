from __future__ import annotations

from pathlib import Path

from ..config.models import PostprocessConfig
from ..core import build_particle_summary, build_release_grid_from_summary
from ..io import (
    load_trajectory_table,
    save_dataset_netcdf,
    save_grid_table,
    save_particle_summary,
)
from ..analyses import compute_beaching_times
from ..plotting import plot_grid_map

def run_beaching_times(cfg: PostprocessConfig, context: dict) -> None:
    """
    Beaching-times workflow.
    """
    if "trajectory_table" not in context:
        print("Loading trajectory table")

        df = load_trajectory_table(
            cfg.dataset.input_path,
            truncate_stagnant=cfg.cleaning.truncate_stagnant,
            stagnant_tol=cfg.cleaning.stagnant_tol,
            stagnant_min_consecutive=cfg.cleaning.stagnant_min_consecutive,
        )
        context["trajectory_table"] = df

    if "particle_summary" not in context:
        print("Building particle summary")
        summary = build_particle_summary(context["trajectory_table"])
        context["particle_summary"] = summary
    else:
        summary = context["particle_summary"]

    print("Building release grid from particle summary")
    grid = build_release_grid_from_summary(
        summary,
        lon_col=cfg.beaching_times.lon_col,
        lat_col=cfg.beaching_times.lat_col,
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

    if cfg.exports.save_particle_summary:
        summary_path = outdir / f"particle_summary.{cfg.exports.table_format}"
        print("Saving particle summary:", summary_path)
        save_particle_summary(
            summary,
            summary_path,
            format=cfg.exports.table_format,
        )

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
        plot_grid_map(
            ds,
            var_name="beaching_time_seconds",
            outpath=plot_path,
            title="Beaching time",
            projection=cfg.plotting.projection,
        )