from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analyses import compute_sampled_map
from ..config.models import PostprocessConfig, SampledMapPlotConfig
from ..core import build_grid_from_config
from ..io import save_dataset_netcdf, save_grid_table
from ..plotting import plot_grid_map
from .base_products import (
    OBSERVATION_VARIABLE_METADATA_CONTEXT_KEY,
    get_trajectory_table,
)


def resolve_sampled_map_plot_limits(
    values: np.ndarray,
    *,
    plot_cfg: SampledMapPlotConfig,
    signed: bool = False,
) -> tuple[float | None, float | None]:
    """Resolve explicit and percentile limits for one sampled-map figure."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return plot_cfg.vmin, plot_cfg.vmax

    if plot_cfg.percentile_limits is None:
        automatic_min = float(np.min(finite))
        automatic_max = float(np.max(finite))
    else:
        lower, upper = plot_cfg.percentile_limits
        automatic_min, automatic_max = (
            float(value) for value in np.percentile(finite, [lower, upper])
        )

    if signed and plot_cfg.vmin is None and plot_cfg.vmax is None:
        magnitude = max(abs(automatic_min), abs(automatic_max))
        if magnitude == 0.0:
            magnitude = 1.0e-12
        return -magnitude, magnitude

    vmin = plot_cfg.vmin if plot_cfg.vmin is not None else automatic_min
    vmax = plot_cfg.vmax if plot_cfg.vmax is not None else automatic_max
    if vmin == vmax:
        delta = max(abs(vmin) * 1.0e-9, 1.0e-12)
        if plot_cfg.vmin is None:
            vmin -= delta
        if plot_cfg.vmax is None:
            vmax += delta
    return vmin, vmax


def _colorbar_label(ds, var_name: str) -> str:
    attrs = ds[var_name].attrs
    source = str(attrs.get("source_variable", var_name))
    suffix = var_name.removeprefix(f"{source}_")
    product_labels = {
        "mean": "mean",
        "std": "standard deviation",
        "smoothed_mean": "smoothed mean",
        "zonal_gradient": "zonal gradient",
        "meridional_gradient": "meridional gradient",
        "gradient_magnitude": "gradient magnitude",
    }
    label = f"{source} {product_labels.get(suffix, suffix.replace('_', ' '))}"
    units = attrs.get("units")
    return f"{label} [{units}]" if units else label


def run_sampled_map(cfg: PostprocessConfig, context: dict) -> None:
    """Run the generic sampled-observation map workflow."""
    if cfg.sampled_map is None:
        raise ValueError(
            "analysis.types includes 'sampled_map' but the sampled_map section "
            "is missing."
        )

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

    print("Computing sampled observation maps")
    result = compute_sampled_map(
        df,
        grid=grid,
        cfg=cfg.sampled_map,
        variable_metadata=context.get(OBSERVATION_VARIABLE_METADATA_CONTEXT_KEY),
    )
    context["sampled_map"] = result

    for variable in cfg.sampled_map.variables:
        point_count = int(result.dataset[f"{variable}_point_count"].sum().item())
        trajectory_cells = int(
            np.count_nonzero(
                result.dataset[f"{variable}_trajectory_count"].values
            )
        )
        supported_cells = int(
            np.count_nonzero(np.isfinite(result.dataset[f"{variable}_mean"].values))
        )
        print(
            f"  {variable}: {point_count} valid points, "
            f"{trajectory_cells} occupied cells, {supported_cells} supported cells"
        )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.sampled_map.output.save_table:
        table_path = outdir / f"sampled_map_table.{cfg.exports.table_format}"
        print("Saving sampled-map table:", table_path)
        save_grid_table(
            result.table,
            table_path,
            format=cfg.exports.table_format,
        )

    if cfg.sampled_map.output.save_netcdf:
        netcdf_path = outdir / "sampled_map.nc"
        print("Saving sampled-map dataset:", netcdf_path)
        save_dataset_netcdf(result.dataset, netcdf_path)

    if not cfg.sampled_map.output.save_figures:
        return

    for variable, variable_cfg in cfg.sampled_map.variables.items():
        products = (
            ("mean", variable_cfg.plotting.mean, False),
            ("std", variable_cfg.plotting.std, False),
            ("smoothed_mean", variable_cfg.plotting.smoothed_mean, False),
            ("zonal_gradient", variable_cfg.plotting.zonal_gradient, True),
            (
                "meridional_gradient",
                variable_cfg.plotting.meridional_gradient,
                True,
            ),
            (
                "gradient_magnitude",
                variable_cfg.plotting.gradient_magnitude,
                False,
            ),
        )
        for product, plot_cfg, signed in products:
            if not plot_cfg.enabled:
                continue
            var_name = f"{variable}_{product}"
            if var_name not in result.dataset:
                continue
            values = result.dataset[var_name].values
            if not np.isfinite(values).any():
                print(f"Skipping empty sampled-map plot: {var_name}")
                continue
            vmin, vmax = resolve_sampled_map_plot_limits(
                values,
                plot_cfg=plot_cfg,
                signed=signed,
            )
            plot_path = outdir / f"sampled_map_{variable}_{product}.png"
            print("Saving sampled-map figure:", plot_path)
            plot_grid_map(
                result.dataset,
                var_name=var_name,
                outpath=plot_path,
                projection=cfg.plotting.projection,
                title=str(result.dataset[var_name].attrs.get("long_name", "")),
                vmin=vmin,
                vmax=vmax,
                cmap=plot_cfg.cmap,
                colorbar_label=(
                    plot_cfg.colorbar_label
                    or _colorbar_label(result.dataset, var_name)
                ),
                title_fontsize=cfg.plotting.title_fontsize,
                colorbar_fontsize=cfg.plotting.colorbar_fontsize,
                colorbar_tick_fontsize=cfg.plotting.colorbar_tick_fontsize,
                axis_tick_fontsize=cfg.plotting.axis_tick_fontsize,
            )
