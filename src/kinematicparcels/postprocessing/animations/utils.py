from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import xarray as xr


def get_fixed_color_limits(
    ds: xr.Dataset,
    *,
    var_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    """
    Return fixed color limits for an animated variable.

    If vmin/vmax are provided, they are used directly.
    Otherwise they are computed once on the whole dataset.
    """
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    data = ds[var_name].values

    if vmin is None:
        finite = np.isfinite(data)
        if not finite.any():
            raise ValueError(f"Variable '{var_name}' contains no finite values.")
        vmin = float(np.nanmin(data))

    if vmax is None:
        finite = np.isfinite(data)
        if not finite.any():
            raise ValueError(f"Variable '{var_name}' contains no finite values.")
        vmax = float(np.nanmax(data))

    if vmax < vmin:
        raise ValueError("vmax must be greater than or equal to vmin.")

    return vmin, vmax


def add_time_progress_bar(
    fig: plt.Figure,
    *,
    current_time,
    time_min,
    time_max,
    rect: tuple[float, float, float, float] = (0.12, 0.06, 0.76, 0.035),
    facecolor: str = "0.85",
    progress_color: str = "0.35",
    fontsize: int = 8,
) -> plt.Axes:
    """
    Add a time progress bar to a figure.

    Parameters
    ----------
    fig
        Matplotlib figure.
    current_time
        Current frame time.
    time_min, time_max
        Full animation time range.
    rect
        Axes rectangle in figure coordinates: (left, bottom, width, height)
    """
    current_time = pd.to_datetime(current_time)
    time_min = pd.to_datetime(time_min)
    time_max = pd.to_datetime(time_max)

    if time_max <= time_min:
        raise ValueError("time_max must be greater than time_min.")

    ax = fig.add_axes(rect)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(0, 1)

    # background bar
    bg = patches.Rectangle(
        (mdates.date2num(time_min), 0.2),
        width=mdates.date2num(time_max) - mdates.date2num(time_min),
        height=0.6,
        facecolor=facecolor,
        edgecolor="none",
        transform=ax.transData,
        zorder=1,
    )
    ax.add_patch(bg)

    # progress bar
    progress_width = mdates.date2num(current_time) - mdates.date2num(time_min)
    progress = patches.Rectangle(
        (mdates.date2num(time_min), 0.2),
        width=max(progress_width, 0.0),
        height=0.6,
        facecolor=progress_color,
        edgecolor="none",
        transform=ax.transData,
        zorder=2,
    )
    ax.add_patch(progress)

    # current time marker
    ax.axvline(current_time, ymin=0.1, ymax=0.9, linewidth=1.2, zorder=3)

    # labels
    ax.text(
        time_min,
        -0.15,
        pd.to_datetime(time_min).strftime("%Y-%m-%d"),
        ha="left",
        va="top",
        fontsize=fontsize,
    )
    ax.text(
        time_max,
        -0.15,
        pd.to_datetime(time_max).strftime("%Y-%m-%d"),
        ha="right",
        va="top",
        fontsize=fontsize,
    )
    ax.text(
        current_time,
        1.05,
        pd.to_datetime(current_time).strftime("%Y-%m-%d %H:%M"),
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def build_animation_colormap(
    *,
    cmap_name: str = "viridis",
    under_color: str = "magenta",
    over_color: str = "red",
    vmin: float,
    vmax: float,
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """
    Build a colormap and normalization for animations with fixed limits.

    Values below vmin are shown with the 'under' color.
    Values above vmax are shown with the 'over' color.
    """
    if vmax < vmin:
        raise ValueError("vmax must be greater than or equal to vmin.")

    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_under(under_color)
    cmap.set_over(over_color)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    return cmap, norm