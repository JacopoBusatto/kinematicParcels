from __future__ import annotations

from pathlib import Path

from ..analyses import (
    build_region_manager,
    classify_start_end_regions,
    compute_start_end_region_maps,
)
from ..config.models import PostprocessConfig
from ..core import build_particle_summary, build_release_grid_from_summary
from ..io import (
    load_trajectory_table,
    save_dataset_netcdf,
    save_grid_table,
    save_particle_summary,
)
from ..plotting import plot_discrete_grid_map


def run_start_end_regions(cfg: PostprocessConfig, context: dict) -> None:
    """
    Start/end region workflow.
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

    print("Building region manager")
    region_manager = build_region_manager(
        region_labels=cfg.start_end_regions.region_labels,
    )

    print("Classifying start/end regions")
    classified_summary = classify_start_end_regions(
        summary,
        region_manager=region_manager,
        how_many=cfg.start_end_regions.how_many,
        priority_level=cfg.start_end_regions.priority_level,
        priority_mode=cfg.start_end_regions.priority_mode,
        input_lon_mode=cfg.start_end_regions.input_lon_mode,
    )

    print("Building release grid from classified summary")
    grid = build_release_grid_from_summary(
        classified_summary,
        lon_col="lon0",
        lat_col="lat0",
    )

    print("Computing start/end region maps")
    start_grid_table, start_ds, end_grid_table, end_ds = compute_start_end_region_maps(
        classified_summary,
        grid=grid,
        lon_col="lon0",
        lat_col="lat0",
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.exports.save_particle_summary:
        summary_path = outdir / f"particle_summary_with_regions.{cfg.exports.table_format}"
        print("Saving classified particle summary:", summary_path)
        save_particle_summary(
            classified_summary,
            summary_path,
            format=cfg.exports.table_format,
        )

    start_table_path = outdir / f"start_regions_table.{cfg.exports.table_format}"
    end_table_path = outdir / f"end_regions_table.{cfg.exports.table_format}"

    start_nc_path = outdir / "start_regions.nc"
    end_nc_path = outdir / "end_regions.nc"

    print("Saving start region table:", start_table_path)
    save_grid_table(
        start_grid_table,
        start_table_path,
        format=cfg.exports.table_format,
    )

    print("Saving end region table:", end_table_path)
    save_grid_table(
        end_grid_table,
        end_table_path,
        format=cfg.exports.table_format,
    )

    print("Saving start region dataset:", start_nc_path)
    save_dataset_netcdf(start_ds, start_nc_path)

    print("Saving end region dataset:", end_nc_path)
    save_dataset_netcdf(end_ds, end_nc_path)

    if cfg.start_end_regions.plot:
        start_plot_path = outdir / "start_regions.png"
        end_plot_path = outdir / "end_regions.png"

        print("Saving start region plot:", start_plot_path)
        plot_discrete_grid_map(
            start_ds,
            var_name="start_numericLabel",
            outpath=start_plot_path,
            projection=cfg.plotting.projection,
            title="Start regions",
        )

        print("Saving end region plot:", end_plot_path)
        plot_discrete_grid_map(
            end_ds,
            var_name="end_numericLabel",
            outpath=end_plot_path,
            projection=cfg.plotting.projection,
            title="End regions",
        )