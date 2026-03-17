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
from .utils import (
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
    title: str = "",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_time_bar: bool = True,
    figsize: tuple[float, float] = (12, 8),
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
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

    da = ds[var_name]

    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(set(da.dims)):
        raise ValueError(
            f"Variable '{var_name}' must have dimensions including {required_dims}."
        )

    if fps <= 0:
        raise ValueError("fps must be > 0.")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    times = ds["time"].values
    if len(times) == 0:
        raise ValueError("Dataset contains no time steps to animate.")

    vmin, vmax = get_fixed_color_limits(
        ds,
        var_name=var_name,
        vmin=vmin,
        vmax=vmax,
    )

    cmap, norm = build_animation_colormap(
        cmap_name="viridis",
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

        for it, time_value in enumerate(times):
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

            frame = da.isel(time=it)

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
                extend="both",
            )
            cbar.set_label(colorbar_label)

            if title:
                ax.set_title(title)
            else:
                ax.set_title(var_name)

            if show_time_bar:
                add_time_progress_bar(
                    fig,
                    current_time=time_value,
                    time_min=times[0],
                    time_max=times[-1],
                )
            else:
                ax.text(
                    0.5,
                    -0.06,
                    str(np.datetime_as_string(time_value, unit="m")),
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
        imageio.mimsave(outpath, images, fps=fps)

    return outpath