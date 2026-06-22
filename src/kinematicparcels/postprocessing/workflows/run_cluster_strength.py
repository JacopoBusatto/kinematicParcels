from __future__ import annotations

from pathlib import Path

from ..analyses import compute_cluster_strength
from ..animations import animate_density
from ..config.models import ClusterStrengthConfig, PostprocessConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf
from .base_products import get_trajectory_table
from .snapshots import resolve_snapshot_indices, save_gridded_snapshots


def _require_cluster_strength_config(cfg: PostprocessConfig) -> ClusterStrengthConfig:
    if cfg.cluster_strength is None:
        raise ValueError(
            "analysis.types includes 'cluster_strength' but the cluster_strength section is missing."
        )
    return cfg.cluster_strength


def _resolve_snapshot_indices(
    timestep_snaps: int | tuple[int, ...] | None,
    *,
    n_times: int,
) -> tuple[int, ...]:
    return resolve_snapshot_indices(
        timestep_snaps,
        n_times=n_times,
        config_name="cluster_strength",
    )


def _save_snapshots(
    ds,
    *,
    cluster_cfg: ClusterStrengthConfig,
    outdir: Path,
    projection: str,
    title_fontsize: int | None,
    colorbar_fontsize: int | None,
    colorbar_tick_fontsize: int | None,
    axis_tick_fontsize: int | None,
) -> None:
    save_gridded_snapshots(
        ds,
        var_name="cluster_strength",
        timestep_snaps=cluster_cfg.timestep_snaps,
        config_name="cluster_strength",
        outdir=outdir,
        filename_prefix="cluster_strength",
        title_prefix="Cluster strength",
        projection=projection,
        vmin=cluster_cfg.vmin,
        vmax=cluster_cfg.vmax,
        min_mask_value=cluster_cfg.min_mask_value,
        cmap=cluster_cfg.cmap,
        colorbar_label="cluster_strength",
        title_fontsize=title_fontsize,
        colorbar_fontsize=colorbar_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        axis_tick_fontsize=axis_tick_fontsize,
    )


def run_cluster_strength(cfg: PostprocessConfig, context: dict) -> None:
    """
    Cluster-strength workflow.
    """
    cluster_cfg = _require_cluster_strength_config(cfg)

    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    if "grid" not in context:
        print("Building grid")
        grid = build_grid_from_config(
            cfg,
            df,
            lon_col="lon",
            lat_col="lat",
            time_col="time",
        )
        context["grid"] = grid

    grid = context["grid"]

    print("Computing cluster strength")
    cluster_ds = compute_cluster_strength(
        df,
        grid=grid,
        scale_km=cluster_cfg.scale_km,
        distance=cluster_cfg.distance,
        cutoff_factor=cluster_cfg.cutoff_factor,
        mask=cluster_cfg.mask,
        lon_col="lon",
        lat_col="lat",
        time_col="time",
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    nc_path = outdir / "cluster_strength.nc"

    print("Saving cluster strength dataset:", nc_path)
    save_dataset_netcdf(cluster_ds, nc_path)

    if cluster_cfg.plot_snaps:
        _save_snapshots(
            cluster_ds,
            cluster_cfg=cluster_cfg,
            outdir=outdir,
            projection=cfg.plotting.projection,
            title_fontsize=cfg.plotting.title_fontsize,
            colorbar_fontsize=cfg.plotting.colorbar_fontsize,
            colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
            axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
        )

    if cluster_cfg.animate:
        gif_path = outdir / "cluster_strength.gif"
        print("Saving cluster strength animation:", gif_path)
        animate_density(
            cluster_ds,
            var_name="cluster_strength",
            outpath=gif_path,
            projection=cfg.plotting.projection,
            fps=cluster_cfg.animation_fps,
            every_n=cluster_cfg.animation_every_n,
            title="cluster_strength",
            colorbar_label="cluster_strength",
            vmin=cluster_cfg.vmin,
            vmax=cluster_cfg.vmax,
            min_mask_value=cluster_cfg.min_mask_value,
            cmap_name=cluster_cfg.cmap,
            show_time_bar=True,
        )
