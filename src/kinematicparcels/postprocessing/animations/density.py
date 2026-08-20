from __future__ import annotations

from pathlib import Path
import tempfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..plotting.projections import get_projection
from ..plotting.masking import mask_values_below
from ..plotting.colorbar import infer_colorbar_extend
from .utils import (
    add_numeric_progress_bar,
    add_time_progress_bar,
    build_animation_colormap,
    get_fixed_color_limits,
)


def animate_density(
    ds: xr.Dataset,
    *,
    var_name: str,
    outpath: str | Path,
    projection: str = "PlateCarree",
    fps: int = 8,
    every_n: int = 1,
    title: str = "",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    min_mask_value: float | None = None,
    cmap_name: str = "viridis",
    show_time_bar: bool = True,
    figsize: tuple[float, float] = (12, 8),
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
    frame_dim: str = "time",
    frame_label: str | None = None,
    frame_units: str = "",
) -> Path:
    """
    Animate a time-dependent gridded density variable and save it as a GIF.

    Parameters
    ----------
    ds
        xarray Dataset containing dimensions: time, lat, lon
    var_name
        Variable to animate.
    outpath
        Output GIF path.
    projection
        Cartopy projection name.
    fps
        Frames per second for the GIF.
    title
        Figure title.
    colorbar_label
        Label for the colorbar. If None, var_name is used.
    vmin, vmax
        Fixed color limits. If None, computed once from the whole dataset.
    show_time_bar
        If True, add a time progress bar.
    """
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    plot_ds = ds
    if min_mask_value is not None:
        plot_ds = ds.copy()
        plot_ds[var_name] = mask_values_below(ds[var_name], min_mask_value)

    da = plot_ds[var_name]

    required_dims = {frame_dim, "lat", "lon"}
    if not required_dims.issubset(set(da.dims)):
        raise ValueError(
            f"Variable '{var_name}' must have dimensions including {required_dims}."
        )

    if fps <= 0:
        raise ValueError("fps must be > 0.")
    if every_n < 1:
        raise ValueError("every_n must be >= 1.")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    all_frame_values = ds[frame_dim].values
    frame_indices = list(range(0, len(all_frame_values), every_n))
    frame_values = all_frame_values[frame_indices]
    if len(frame_values) == 0:
        raise ValueError("Dataset contains no time steps to animate.")

    vmin, vmax = get_fixed_color_limits(
        plot_ds,
        var_name=var_name,
        vmin=vmin,
        vmax=vmax,
    )
    colorbar_extend = infer_colorbar_extend(
        da.isel({frame_dim: frame_indices}).values,
        vmin=vmin,
        vmax=vmax,
    )

    cmap, norm = build_animation_colormap(
        cmap_name=cmap_name,
        under_color="magenta",
        over_color="red",
        vmin=vmin,
        vmax=vmax,
    )

    proj = get_projection(projection)
    colorbar_label = colorbar_label or var_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_paths: list[Path] = []

        for it, (ds_idx, frame_value) in enumerate(zip(frame_indices, frame_values)):
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=proj)

            if add_land:
                land = cfeature.NaturalEarthFeature(
                    "physical",
                    "land",
                    "10m",
                    edgecolor="none",
                    facecolor=cfeature.COLORS["land"],
                )
                ax.add_feature(land, zorder=0)

            if add_coastlines:
                ax.coastlines(resolution="10m", linewidth=0.8)

            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
                gl.top_labels = False
                gl.right_labels = False

            frame = da.isel({frame_dim: ds_idx})

            mesh = ax.pcolormesh(
                ds["lon"].values,
                ds["lat"].values,
                frame.values,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap=cmap,
                norm=norm,
            )

            cbar = plt.colorbar(
                mesh,
                ax=ax,
                shrink=0.9,
                pad=0.03,
                extend=colorbar_extend,
            )
            cbar.set_label(colorbar_label)

            if title:
                ax.set_title(title)
            else:
                ax.set_title(var_name)

            if show_time_bar:
                if np.issubdtype(np.asarray(frame_values).dtype, np.datetime64):
                    add_time_progress_bar(
                        fig,
                        current_time=frame_value,
                        time_min=frame_values[0],
                        time_max=frame_values[-1],
                    )
                else:
                    add_numeric_progress_bar(
                        fig,
                        current_value=float(frame_value),
                        value_min=float(frame_values[0]),
                        value_max=float(frame_values[-1]),
                        label=frame_label or frame_dim,
                        units=frame_units,
                    )
            else:
                if np.issubdtype(np.asarray(frame_values).dtype, np.datetime64):
                    frame_text = str(np.datetime_as_string(frame_value, unit="m"))
                else:
                    suffix = f" {frame_units}" if frame_units else ""
                    frame_text = (
                        f"{frame_label or frame_dim}={float(frame_value):.6g}{suffix}"
                    )
                ax.text(
                    0.5,
                    -0.06,
                    frame_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                )

            frame_path = tmpdir / f"frame_{it:05d}.png"

            fig.subplots_adjust(
                left=0.06,
                right=0.92,
                bottom=0.14 if show_time_bar else 0.08,
                top=0.93,
            )

            plt.savefig(frame_path, dpi=150)
            plt.close(fig)

            frame_paths.append(frame_path)

        images = [imageio.imread(fp) for fp in frame_paths]
        imageio.mimsave(outpath, images, duration=1000.0 / float(fps))

    return outpath
