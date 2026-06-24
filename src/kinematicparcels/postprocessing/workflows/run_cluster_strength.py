from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ..analyses import compute_cluster_strength
from ..animations import animate_density
from ..config.models import (
    ClusterStrengthAnimationConfig,
    ClusterStrengthConfig,
    ClusterStrengthSnapshotsConfig,
    PostprocessConfig,
)
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf
from ..plotting import plot_grid_map
from ..plotting.masking import mask_values_below
from .base_products import get_trajectory_table
from .snapshots import format_time_for_filename


def _require_cluster_strength_config(cfg: PostprocessConfig) -> ClusterStrengthConfig:
    if cfg.cluster_strength is None:
        raise ValueError(
            "analysis.types includes 'cluster_strength' but the cluster_strength section is missing."
        )
    return cfg.cluster_strength


def _as_tuple(value: float | tuple[float, ...] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return (float(value),)


def _match_age_value(
    available_age_days,
    *,
    requested_age_days: float,
    tolerance_days: float | None,
) -> float:
    available = np.asarray(available_age_days, dtype=float)
    if available.size == 0:
        raise ValueError("cluster_strength dataset has no age_days values.")

    distances = np.abs(available - float(requested_age_days))
    nearest_idx = int(np.argmin(distances))
    nearest = float(available[nearest_idx])

    if tolerance_days is None:
        if not np.isclose(nearest, float(requested_age_days), rtol=0.0, atol=1.0e-9):
            raise ValueError(
                "cluster_strength requested fixed_age_days "
                f"{requested_age_days} is not present in age_days. "
                "Set age_tolerance_days to allow nearest-age matching."
            )
        return nearest

    if float(distances[nearest_idx]) > tolerance_days:
        raise ValueError(
            "cluster_strength requested fixed_age_days "
            f"{requested_age_days} has no age_days value within "
            f"{tolerance_days} days."
        )
    return nearest


def _dataset_for_release_animation(ds: xr.Dataset, release_time) -> xr.Dataset:
    release_timestamp = pd.Timestamp(release_time)
    release_ds = ds.sel(release_time=release_time).rename({"age_days": "time"})
    frame_times = [
        release_timestamp + pd.Timedelta(days=float(age_days))
        for age_days in ds["age_days"].values
    ]
    return release_ds.assign_coords(time=np.array(frame_times, dtype="datetime64[ns]"))


def _dataset_for_fixed_age_animation(ds: xr.Dataset, age_days: float) -> xr.Dataset:
    fixed_ds = ds.sel(age_days=age_days).rename({"release_time": "time"})
    return fixed_ds.assign_coords(time=ds["release_time"].values)


def _apply_plot_mask(ds: xr.Dataset, *, min_mask_value: float | None) -> xr.Dataset:
    if min_mask_value is None:
        return ds
    out = ds.copy()
    out["cluster_strength"] = mask_values_below(out["cluster_strength"], min_mask_value)
    return out


def _warn_if_release_flag_disagrees(cfg: PostprocessConfig, ds: xr.Dataset) -> None:
    n_release_times = int(ds.sizes.get("release_time", 0))
    if cfg.release.continuous and n_release_times <= 1:
        warnings.warn(
            "release.continuous is true, but cluster_strength detected only one release_time.",
            RuntimeWarning,
            stacklevel=2,
        )
    if (not cfg.release.continuous) and n_release_times > 1:
        warnings.warn(
            "release.continuous is false, but cluster_strength detected multiple release_time values; "
            "all detected releases will be written.",
            RuntimeWarning,
            stacklevel=2,
        )


def _save_release_age_snapshots(
    ds: xr.Dataset,
    *,
    snapshot_cfg: ClusterStrengthSnapshotsConfig,
    outdir: Path,
    projection: str,
    title_fontsize: int | None,
    colorbar_fontsize: int | None,
    colorbar_tick_fontsize: int | None,
    axis_tick_fontsize: int | None,
) -> None:
    requested_ages = _as_tuple(snapshot_cfg.fixed_age_days)
    if not requested_ages:
        warnings.warn(
            "cluster_strength.snapshots.enabled is true, but fixed_age_days is null; "
            "no snapshots will be saved.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    for requested_age in requested_ages:
        age_value = _match_age_value(
            ds["age_days"].values,
            requested_age_days=requested_age,
            tolerance_days=snapshot_cfg.age_tolerance_days,
        )
        age_label = f"{age_value:.6g}".replace("-", "minus_").replace(".", "p")

        for release_time in ds["release_time"].values:
            frame_ds = ds.sel(release_time=release_time, age_days=age_value)
            frame_ds = _apply_plot_mask(frame_ds, min_mask_value=snapshot_cfg.min_mask_value)
            if not np.isfinite(frame_ds["cluster_strength"].values).any():
                warnings.warn(
                    "Skipping empty cluster_strength snapshot for "
                    f"release_time={pd.Timestamp(release_time)} age_days={age_value}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            timestamp = format_time_for_filename(release_time)
            plot_path = outdir / f"cluster_strength_release_{timestamp}_age_{age_label}.png"
            print("Saving cluster strength snapshot:", plot_path)
            plot_grid_map(
                frame_ds,
                var_name="cluster_strength",
                outpath=plot_path,
                projection=projection,
                title=f"Cluster strength {timestamp} age={age_value:.6g} days",
                vmin=snapshot_cfg.vmin,
                vmax=snapshot_cfg.vmax,
                cmap=snapshot_cfg.cmap,
                colorbar_label="cluster_strength",
                title_fontsize=title_fontsize,
                colorbar_fontsize=colorbar_fontsize,
                colorbar_tick_fontsize=colorbar_tick_fontsize,
                axis_tick_fontsize=axis_tick_fontsize,
            )


def _save_release_age_animations(
    ds: xr.Dataset,
    *,
    animation_cfg: ClusterStrengthAnimationConfig,
    outdir: Path,
    projection: str,
) -> None:
    if not animation_cfg.enabled:
        return

    requested_ages = _as_tuple(animation_cfg.fixed_age_days)

    if requested_ages:
        for requested_age in requested_ages:
            age_value = _match_age_value(
                ds["age_days"].values,
                requested_age_days=requested_age,
                tolerance_days=animation_cfg.age_tolerance_days,
            )
            age_label = f"{age_value:.6g}".replace("-", "minus_").replace(".", "p")
            gif_ds = _dataset_for_fixed_age_animation(ds, age_value)
            gif_path = outdir / f"cluster_strength_age_{age_label}.gif"
            print("Saving cluster strength fixed-age animation:", gif_path)
            animate_density(
                gif_ds,
                var_name="cluster_strength",
                outpath=gif_path,
                projection=projection,
                fps=animation_cfg.fps,
                every_n=animation_cfg.every_n,
                title=f"cluster_strength age={age_value:.6g} days",
                colorbar_label="cluster_strength",
                vmin=animation_cfg.vmin,
                vmax=animation_cfg.vmax,
                min_mask_value=animation_cfg.min_mask_value,
                cmap_name=animation_cfg.cmap,
                show_time_bar=True,
            )
        return

    if not animation_cfg.every_release:
        warnings.warn(
            "cluster_strength.animation.enabled is true, but every_release is false "
            "and fixed_age_days is null; no animation will be saved.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    for release_time in ds["release_time"].values:
        gif_ds = _dataset_for_release_animation(ds, release_time)
        if not np.isfinite(gif_ds["cluster_strength"].values).any():
            warnings.warn(
                "Skipping empty cluster_strength animation for "
                f"release_time={pd.Timestamp(release_time)}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        timestamp = format_time_for_filename(release_time)
        gif_path = outdir / f"cluster_strength_release_{timestamp}.gif"
        print("Saving cluster strength release animation:", gif_path)
        animate_density(
            gif_ds,
            var_name="cluster_strength",
            outpath=gif_path,
            projection=projection,
            fps=animation_cfg.fps,
            every_n=animation_cfg.every_n,
            title=f"cluster_strength release={timestamp}",
            colorbar_label="cluster_strength",
            vmin=animation_cfg.vmin,
            vmax=animation_cfg.vmax,
            min_mask_value=animation_cfg.min_mask_value,
            cmap_name=animation_cfg.cmap,
            show_time_bar=True,
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
        max_group_member=cluster_cfg.max_group_member,
        lon_col="lon",
        lat_col="lat",
        time_col="time",
    )
    _warn_if_release_flag_disagrees(cfg, cluster_ds)

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    nc_path = outdir / "cluster_strength.nc"

    print("Saving cluster strength dataset:", nc_path)
    save_dataset_netcdf(cluster_ds, nc_path)

    if cluster_cfg.snapshots.enabled:
        _save_release_age_snapshots(
            cluster_ds,
            snapshot_cfg=cluster_cfg.snapshots,
            outdir=outdir,
            projection=cfg.plotting.projection,
            title_fontsize=cfg.plotting.title_fontsize,
            colorbar_fontsize=cfg.plotting.colorbar_fontsize,
            colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
            axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
        )

    if cluster_cfg.animation.enabled:
        _save_release_age_animations(
            cluster_ds,
            animation_cfg=cluster_cfg.animation,
            outdir=outdir,
            projection=cfg.plotting.projection,
        )
