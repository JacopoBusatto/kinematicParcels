from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..analyses import compute_exponent_maps
from ..config.models import ExponentMapPlotConfig, PostprocessConfig
from ..core import build_grid_from_config, build_release_grid_from_summary
from ..io import save_dataset_netcdf
from ..plotting import plot_exponent_map
from .base_products import get_particle_summary, get_trajectory_table


def _require_exponent_maps_config(cfg: PostprocessConfig):
    if cfg.exponent_maps is None:
        raise ValueError("analysis.types includes 'exponent_maps' but the exponent_maps section is missing.")
    return cfg.exponent_maps


def _validate_grouped_regular_release(summary_df: pd.DataFrame) -> pd.DataFrame:
    required = ["group_id", "group_member", "group_size", "time0", "lon0", "lat0"]
    missing = [col for col in required if col not in summary_df.columns]
    if missing:
        raise KeyError(f"Exponent maps require grouped particle summary columns: {missing}")

    centers = summary_df.loc[summary_df["group_member"] == 1, required].copy()
    if centers.empty:
        raise ValueError("Exponent maps require grouped outputs with group_member=1 release centers.")

    if int(centers["group_size"].max()) <= 1:
        raise ValueError("Exponent maps require grouped outputs with group_size > 1.")

    if centers.duplicated(subset=["time0", "lon0", "lat0"]).any():
        raise ValueError("Exponent maps require unique release centers per release time.")

    return centers.sort_values(["time0", "lat0", "lon0"]).reset_index(drop=True)


def _normalize_time_key(value):
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(value)
    return value


def _points_to_dataset(
    points: pd.DataFrame,
    *,
    grid,
    value_col: str,
    scale_col: str,
    scale_dim: str,
    var_name: str,
    attrs: dict[str, object],
) -> xr.Dataset:
    finite_points = points.loc[np.isfinite(points[value_col])].copy()
    if finite_points.empty:
        raise ValueError(f"Exponent-map variable '{value_col}' has no finite values to grid.")

    binned = grid.assign_bins(finite_points, lon_col="lon0", lat_col="lat0", drop_outside=False)
    if binned[["lon_bin", "lat_bin"]].isna().any().any():
        raise ValueError("Exponent-map release centers fall outside the inferred regular grid.")

    if binned.duplicated(subset=["time0", scale_col, "lon_bin", "lat_bin"]).any():
        raise ValueError("Exponent-map grid mapping produced duplicate values for the same time/scale/cell.")

    time_values = pd.Index(binned["time0"].dropna().unique()).sort_values()
    scale_values = np.sort(binned[scale_col].dropna().unique().astype(float))

    time_lookup = {_normalize_time_key(value): idx for idx, value in enumerate(time_values)}
    scale_lookup = {float(value): idx for idx, value in enumerate(scale_values)}

    data = np.full((len(time_values), len(scale_values), grid.nlat, grid.nlon), np.nan, dtype=float)

    for row in binned.itertuples(index=False):
        time_idx = time_lookup[_normalize_time_key(row.time0)]
        scale_idx = scale_lookup[float(getattr(row, scale_col))]
        data[time_idx, scale_idx, int(row.lat_bin), int(row.lon_bin)] = float(getattr(row, value_col))

    ds = xr.Dataset(
        data_vars={var_name: (("time", scale_dim, "lat", "lon"), data)},
        coords={
            "time": time_values.to_numpy(),
            scale_dim: scale_values,
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs={
            "grid_type": "regular_lonlat",
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "dlon": grid.dlon,
            "dlat": grid.dlat,
            **attrs,
        },
    )
    ds[var_name].attrs["units"] = "days^-1"
    ds[var_name].attrs["long_name"] = var_name.upper()
    return ds


def _apply_plot_mask(da: xr.DataArray, plot_cfg: ExponentMapPlotConfig) -> xr.DataArray:
    if plot_cfg.min_mask_value is None:
        return da
    return da.where(np.abs(da) >= plot_cfg.min_mask_value)


def _save_exponent_plots(
    ds: xr.Dataset,
    *,
    var_name: str,
    scale_dim: str,
    outdir: Path,
    prefix: str,
    plot_cfg: ExponentMapPlotConfig,
    projection: str,
) -> None:
    if not plot_cfg.enabled:
        return

    scale_values = ds[scale_dim].values
    if plot_cfg.average_on_time:
        averaged = _apply_plot_mask(ds[var_name].mean(dim="time", skipna=True), plot_cfg)
        for scale_value in scale_values:
            scale_da = averaged.sel({scale_dim: scale_value})
            plot_path = outdir / f"{prefix}_scale_{float(scale_value):.6g}_mean_time.png"
            plot_exponent_map(
                scale_da,
                outpath=plot_path,
                projection=projection,
                title=f"{prefix.upper()} scale={float(scale_value):.6g}",
                vmin=plot_cfg.vmin,
                vmax=plot_cfg.vmax,
                cmap=plot_cfg.cmap,
                log_scale=plot_cfg.log_scale,
            )
        return

    for scale_value in scale_values:
        for time_value in ds["time"].values:
            scale_da = _apply_plot_mask(
                ds[var_name].sel({scale_dim: scale_value, "time": time_value}),
                plot_cfg,
            )
            timestamp = pd.Timestamp(time_value).strftime("%Y%m%dT%H%M%S")
            plot_path = outdir / f"{prefix}_scale_{float(scale_value):.6g}_{timestamp}.png"
            plot_exponent_map(
                scale_da,
                outpath=plot_path,
                projection=projection,
                title=f"{prefix.upper()} scale={float(scale_value):.6g} time={timestamp}",
                vmin=plot_cfg.vmin,
                vmax=plot_cfg.vmax,
                cmap=plot_cfg.cmap,
                log_scale=plot_cfg.log_scale,
            )


def run_exponent_maps(cfg: PostprocessConfig, context: dict) -> None:
    exponent_cfg = _require_exponent_maps_config(cfg)

    print("Getting trajectory table")
    df = get_trajectory_table(cfg, context)

    print("Getting particle summary")
    summary_df = get_particle_summary(cfg, context)

    centers = _validate_grouped_regular_release(summary_df) if exponent_cfg.require_grouped_regular_grid else summary_df

    print("Building release grid")
    if exponent_cfg.infer_grid_from_start:
        grid = build_release_grid_from_summary(
            centers,
            lon_col="lon0",
            lat_col="lat0",
            time_col="time0",
        )
    else:
        grid = build_grid_from_config(
            cfg,
            centers,
            lon_col="lon0",
            lat_col="lat0",
            time_col="time0",
        )
    dlon_inferred = grid.dlon
    dlat_inferred = grid.dlat
    print(f"Inferred release spacing: dlon={dlon_inferred:.6f}, dlat={dlat_inferred:.6f}")

    print("Computing exponent maps")
    result = compute_exponent_maps(
        df,
        meridional_only=(exponent_cfg.distance == "meridional"),
        fsle_scales_km=exponent_cfg.fsle.scales_km if exponent_cfg.fsle.enabled else (),
        fsle_mask_zeros=exponent_cfg.fsle.mask_zeros,
        ftle_scales_days=exponent_cfg.ftle.scales_days if exponent_cfg.ftle.enabled else (),
        ftle_sampling_mode=exponent_cfg.ftle.sampling_mode,
        ftle_mask_short_windows=exponent_cfg.ftle.mask_short_windows,
        ftle_mask_zeros=exponent_cfg.ftle.mask_zeros,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    common_attrs = {
        "distance": exponent_cfg.distance,
        "simulation_direction": result.simulation_direction,
    }

    if exponent_cfg.fsle.enabled:
        fsle_ds = _points_to_dataset(
            result.fsle_points,
            grid=grid,
            value_col="fsle",
            scale_col="scale_km",
            scale_dim="scale_km",
            var_name="fsle",
            attrs={**common_attrs, "diagnostic": "fsle"},
        )
        fsle_path = outdir / f"fsle_map_{exponent_cfg.distance}.nc"
        print("Saving FSLE map dataset:", fsle_path)
        save_dataset_netcdf(fsle_ds, fsle_path)
        _save_exponent_plots(
            fsle_ds,
            var_name="fsle",
            scale_dim="scale_km",
            outdir=outdir,
            prefix="fsle_map",
            plot_cfg=exponent_cfg.fsle.plot,
            projection=cfg.plotting.projection,
        )

    if exponent_cfg.ftle.enabled:
        ftle_ds = _points_to_dataset(
            result.ftle_points,
            grid=grid,
            value_col="ftle",
            scale_col="scale_days",
            scale_dim="scale_days",
            var_name="ftle",
            attrs={
                **common_attrs,
                "diagnostic": "ftle",
                "sampling_mode": exponent_cfg.ftle.sampling_mode,
            },
        )
        ftle_path = outdir / f"ftle_map_{exponent_cfg.distance}.nc"
        print("Saving FTLE map dataset:", ftle_path)
        save_dataset_netcdf(ftle_ds, ftle_path)
        _save_exponent_plots(
            ftle_ds,
            var_name="ftle",
            scale_dim="scale_days",
            outdir=outdir,
            prefix="ftle_map",
            plot_cfg=exponent_cfg.ftle.plot,
            projection=cfg.plotting.projection,
        )
