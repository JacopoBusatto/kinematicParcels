from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .colorbar import infer_colorbar_extend


def _centers_to_edges(values: np.ndarray, *, fallback_width: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Heatmap coordinates must be a non-empty one-dimensional array.")
    if values.size == 1:
        half_width = 0.5 * fallback_width
        return np.array([values[0] - half_width, values[0] + half_width])

    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _mask_values_at_or_below(
    values: np.ndarray,
    *,
    min_mask_value: float | None,
) -> np.ndarray:
    masked_values = np.asarray(values, dtype=float).copy()
    if min_mask_value is not None:
        masked_values[
            np.isfinite(masked_values) & (masked_values <= min_mask_value)
        ] = np.nan
    return masked_values


def plot_alive_latitude_fraction(
    ds: xr.Dataset,
    *,
    outpath: str | Path,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    min_mask_value: float | None = None,
    as_percent: bool = True,
    masked_color: str = "lightgray",
    title_fontsize: int | None = None,
    colorbar_fontsize: int | None = None,
    colorbar_tick_fontsize: int | None = None,
    axis_tick_fontsize: int | None = None,
) -> None:
    """Plot an alive-tracer fraction heatmap over time/age and latitude."""
    if "alive_tracer_fraction" not in ds:
        raise KeyError("Dataset is missing 'alive_tracer_fraction'.")

    axis_name = str(ds.attrs.get("time_axis", "age"))
    dimension = "time" if axis_name == "time" else "age_days"
    if dimension not in ds.dims or "latitude_bin" not in ds.dims:
        raise ValueError(
            "alive_tracer_fraction must use time/age_days and latitude_bin dimensions."
        )
    if ds.sizes[dimension] == 0 or ds.sizes["latitude_bin"] == 0:
        raise ValueError("Cannot plot an empty alive latitude fraction dataset.")

    values = np.asarray(
        ds["alive_tracer_fraction"]
        .transpose(dimension, "latitude_bin")
        .values,
        dtype=float,
    )
    masked_values = _mask_values_at_or_below(
        values, min_mask_value=min_mask_value
    )
    plot_values = masked_values * 100.0 if as_percent else masked_values
    plot_vmin = None if vmin is None else vmin * (100.0 if as_percent else 1.0)
    plot_vmax = None if vmax is None else vmax * (100.0 if as_percent else 1.0)

    resample_days = ds.attrs.get("resample_days")
    fallback_days = (
        float(resample_days)
        if isinstance(resample_days, (int, float, np.integer, np.floating))
        else 1.0
    )
    if dimension == "time":
        centers = mdates.date2num(
            pd.DatetimeIndex(ds[dimension].values).to_pydatetime()
        )
        x_edges = _centers_to_edges(centers, fallback_width=fallback_days)
    else:
        centers = np.asarray(ds[dimension].values, dtype=float)
        x_edges = _centers_to_edges(centers, fallback_width=fallback_days)

    lat_edges = np.concatenate(
        (
            [float(ds["lat_lower"].values[0])],
            np.asarray(ds["lat_upper"].values, dtype=float),
        )
    )

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(masked_color)
    colorbar_extend = infer_colorbar_extend(
        plot_values, vmin=plot_vmin, vmax=plot_vmax
    )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(
        x_edges,
        lat_edges,
        np.ma.masked_invalid(plot_values.T),
        shading="flat",
        cmap=colormap,
        vmin=plot_vmin,
        vmax=plot_vmax,
    )

    if dimension == "time":
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Time")
        title = "Alive tracer latitude fraction over time"
    else:
        ax.set_xlabel("Signed age since release [days]")
        title = "Alive tracer latitude fraction by age"

    ax.set_ylabel("Latitude [degrees north]")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(lat_edges[0]), float(lat_edges[-1]))
    if axis_tick_fontsize is not None:
        ax.tick_params(labelsize=axis_tick_fontsize)
    if title_fontsize != 0:
        ax.set_title(title, fontsize=title_fontsize)

    colorbar = fig.colorbar(mesh, ax=ax, extend=colorbar_extend)
    colorbar.set_label(
        "Fraction of alive tracers [%]"
        if as_percent
        else "Fraction of alive tracers",
        fontsize=colorbar_fontsize,
    )
    if colorbar_tick_fontsize is not None:
        colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
