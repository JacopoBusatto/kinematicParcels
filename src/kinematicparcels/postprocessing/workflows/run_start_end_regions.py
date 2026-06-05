from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analyses import (
    build_region_manager,
    classify_start_end_regions,
    compute_start_end_region_maps,
)
from ..animations import animate_trajectories
from ..config.models import PostprocessConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_grid_table
from ..plotting import plot_connectivity_map, plot_discrete_grid_map, plot_trajectories_map
from .base_products import get_particle_summary, get_trajectory_table


def _is_grid_mode(cfg: PostprocessConfig) -> bool:
    """Return True when grid-based outputs (maps, NetCDF) are meaningful."""
    return cfg.release.mode == "region_grid" and not cfg.release.continuous


def _build_segment_df(classified_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a synthetic two-point trajectory DataFrame (start → end) from the
    classified particle summary.  Used when ``connectivity_segments=True`` to
    avoid loading the full trajectory table.
    """
    required = ["trajectory", "lon0", "lat0", "lonf", "latf"]
    missing = [c for c in required if c not in classified_summary.columns]
    if missing:
        raise KeyError(f"classified_summary missing columns for segment build: {missing}")

    carry_cols = ["trajectory", "lon0", "lat0"]
    end_cols = ["trajectory", "lonf", "latf"]
    if "group_member" in classified_summary.columns:
        carry_cols = ["trajectory", "group_member", "lon0", "lat0"]
        end_cols = ["trajectory", "group_member", "lonf", "latf"]

    starts = (
        classified_summary[carry_cols]
        .copy()
        .assign(obs=0)
        .rename(columns={"lon0": "lon", "lat0": "lat"})
    )
    ends = (
        classified_summary[end_cols]
        .copy()
        .assign(obs=1)
        .rename(columns={"lonf": "lon", "latf": "lat"})
    )
    sort_cols = ["trajectory"] + (["group_member"] if "group_member" in classified_summary.columns else []) + ["obs"]
    return (
        pd.concat([starts, ends], ignore_index=True)
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )


def _prepare_region_trajectory_inputs(
    traj_df: pd.DataFrame,
    classified_summary: pd.DataFrame,
    *,
    color_by: str,
    max_group_member: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    traj_plot = traj_df.copy()
    summary_plot = classified_summary.copy()

    if max_group_member is not None:
        if "group_member" in traj_plot.columns:
            traj_plot = traj_plot.loc[traj_plot["group_member"] <= max_group_member].copy()
            remaining_members = sorted(pd.unique(traj_plot["group_member"].dropna()).tolist())
            if any(member > max_group_member for member in remaining_members):
                raise ValueError(
                    f"Trajectory plotting filter failed: found group_member values {remaining_members} "
                    f"after applying max_group_member={max_group_member}."
                )
        else:
            remaining_members = []

        if "group_member" in summary_plot.columns:
            summary_plot = summary_plot.loc[summary_plot["group_member"] <= max_group_member].copy()
    else:
        remaining_members = (
            sorted(pd.unique(traj_plot["group_member"].dropna()).tolist())
            if "group_member" in traj_plot.columns
            else []
        )

    if traj_plot.empty:
        raise ValueError("No trajectories remain after applying the group-member plotting filter.")

    if color_by in summary_plot.columns:
        label_source = summary_plot[color_by]
    elif color_by in traj_plot.columns:
        label_source = traj_plot[color_by]
    else:
        raise KeyError(f"Trajectory coloring field '{color_by}' not found before plotting.")

    unique_labels = sorted(pd.Series(label_source).dropna().astype(str).unique().tolist())
    if not unique_labels:
        raise ValueError(f"No non-null labels found for trajectory coloring field '{color_by}'.")

    members_msg = remaining_members if remaining_members else "not available"
    print(f"Trajectory plot pre-check: group_members={members_msg}; {color_by} labels={unique_labels}")
    return traj_plot, summary_plot, unique_labels


def run_start_end_regions(cfg: PostprocessConfig, context: dict) -> None:
    """
    Start/end region workflow.
    """
    print("Getting particle summary")
    summary = get_particle_summary(cfg, context)

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

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    classified_path = outdir / f"particle_summary_with_regions.{cfg.exports.table_format}"
    print("Saving classified particle summary:", classified_path)
    save_grid_table(
        classified_summary,
        classified_path,
        format=cfg.exports.table_format,
    )

    # Grid-based outputs are only meaningful for region_grid + non-continuous releases.
    if _is_grid_mode(cfg):
        print("Building release grid from classified summary")
        if "grid" not in context:
            context["grid"] = build_grid_from_config(
                cfg,
                classified_summary,
                lon_col="lon0",
                lat_col="lat0",
                time_col="time0",
            )
        grid = context["grid"]

        print("Computing start/end region maps")
        start_grid_table, start_ds, end_grid_table, end_ds = compute_start_end_region_maps(
            classified_summary,
            grid=grid,
            lon_col="lon0",
            lat_col="lat0",
        )

        start_table_path = outdir / f"start_regions_table.{cfg.exports.table_format}"
        end_table_path = outdir / f"end_regions_table.{cfg.exports.table_format}"
        start_nc_path = outdir / "start_regions.nc"
        end_nc_path = outdir / "end_regions.nc"

        print("Saving start region table:", start_table_path)
        save_grid_table(start_grid_table, start_table_path, format=cfg.exports.table_format)

        print("Saving end region table:", end_table_path)
        save_grid_table(end_grid_table, end_table_path, format=cfg.exports.table_format)

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
    else:
        print(
            f"Skipping grid outputs (release.mode={cfg.release.mode!r}, "
            f"release.continuous={cfg.release.continuous})."
        )

    # Connectivity outputs: available for all release modes.
    if cfg.start_end_regions.plot_connectivity or cfg.start_end_regions.animate_connectivity:
        color_by = cfg.start_end_regions.connectivity_color_by
        max_group_member = (
            cfg.start_end_regions.connectivity_max_group_member
            if cfg.start_end_regions.connectivity_max_group_member is not None
            else cfg.trajectories.max_group_member
        )
        traj_alpha = (
            cfg.start_end_regions.connectivity_alpha
            if cfg.start_end_regions.connectivity_alpha is not None
            else cfg.trajectories.alpha
        )

        if cfg.start_end_regions.plot_connectivity:
            if cfg.start_end_regions.connectivity_segments:
                print("Building start→end segments from classified summary")
                traj_source = _build_segment_df(classified_summary)
            else:
                print("Loading full trajectory table for connectivity plot")
                traj_source = get_trajectory_table(cfg, context)

            traj_plot_df, summary_plot_df, _ = _prepare_region_trajectory_inputs(
                traj_source,
                classified_summary,
                color_by=color_by,
                max_group_member=max_group_member,
            )

            traj_plot_path = outdir / "connectivity_start.png"
            print("Saving region-coloured trajectories plot:", traj_plot_path)
            plot_trajectories_map(
                traj_plot_df,
                outpath=traj_plot_path,
                projection=cfg.plotting.projection,
                title=cfg.start_end_regions.connectivity_title,
                show_start=cfg.start_end_regions.connectivity_show_start,
                show_end=cfg.start_end_regions.connectivity_show_end,
                alpha=traj_alpha,
                max_group_member=max_group_member,
                summary_df=summary_plot_df,
                color_by=color_by,
                colorbar_label=cfg.start_end_regions.connectivity_label,
            )

            connectivity_plot_path = outdir / "connectivity_end.png"
            print("Saving connectivity map:", connectivity_plot_path)
            plot_connectivity_map(
                traj_plot_df,
                summary_df=summary_plot_df,
                outpath=connectivity_plot_path,
                projection=cfg.plotting.projection,
                title="Connectivity map",
                alpha=traj_alpha,
                max_group_member=max_group_member,
            )

        if cfg.start_end_regions.animate_connectivity:
            # Animation always requires the full trajectory table.
            print("Loading full trajectory table for connectivity animation")
            full_traj_df = get_trajectory_table(cfg, context)

            traj_plot_df_anim, summary_plot_df_anim, _ = _prepare_region_trajectory_inputs(
                full_traj_df,
                classified_summary,
                color_by=color_by,
                max_group_member=max_group_member,
            )

            traj_gif_path = outdir / "connectivity_start.gif"
            print("Saving region-coloured trajectories animation:", traj_gif_path)
            fps = (
                cfg.start_end_regions.connectivity_animation_fps
                if cfg.start_end_regions.connectivity_animation_fps is not None
                else cfg.trajectories.animation_fps
            )
            trail = (
                cfg.start_end_regions.connectivity_trail
                if cfg.start_end_regions.connectivity_trail is not None
                else cfg.trajectories.trail
            )
            trail_steps = (
                cfg.start_end_regions.connectivity_trail_steps
                if cfg.start_end_regions.connectivity_trail_steps is not None
                else cfg.trajectories.trail_steps
            )
            animate_trajectories(
                traj_plot_df_anim,
                outpath=traj_gif_path,
                projection=cfg.plotting.projection,
                fps=fps,
                title=cfg.start_end_regions.connectivity_title,
                color_by=color_by,
                colorbar_label=cfg.start_end_regions.connectivity_label,
                vmin=cfg.trajectories.animation_vmin,
                vmax=cfg.trajectories.animation_vmax,
                cmap_name=cfg.trajectories.animation_cmap,
                cmap_mode=cfg.trajectories.animation_cmap_mode,
                show_time_bar=cfg.trajectories.show_time_bar,
                trail=trail,
                trail_steps=trail_steps,
                show_tracer=(
                    cfg.start_end_regions.connectivity_animation_show_tracer
                    if cfg.start_end_regions.connectivity_animation_show_tracer is not None
                    else True
                ),
                summary_df=summary_plot_df_anim,
                max_group_member=max_group_member,
            )