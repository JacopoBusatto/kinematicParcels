from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from .projections import get_projection


def plot_grid_map(
    ds: xr.Dataset,
    *,
    var_name: str,
    outpath: str | Path,
    projection: str = "PlateCarree",
    title: str = "",
    figsize: tuple[float, float] = (12, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
) -> None:
    """
    Plot a 2D gridded variable from an xarray.Dataset.

    Expected dimensions:
    - lat
    - lon
    """
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    da = ds[var_name]

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"Variable '{var_name}' must have dimensions ('lat', 'lon') or include both lat and lon."
        )

    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError("vmin must be less than or equal to vmax.")

    values = da.values
    if vmin is not None or vmax is not None:
        clip_min = vmin if vmin is not None else -np.inf
        clip_max = vmax if vmax is not None else np.inf
        values = np.clip(values, clip_min, clip_max)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    proj = get_projection(projection)

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

    mesh = ax.pcolormesh(
        ds["lon"].values,
        ds["lat"].values,
        values,
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label(var_name)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(var_name)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_discrete_grid_map(
    ds: xr.Dataset,
    *,
    var_name: str,
    outpath: str | Path,
    projection: str = "PlateCarree",
    title: str = "",
    figsize: tuple[float, float] = (12, 8),
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
) -> None:
    """
    Plot a 2D gridded discrete variable from an xarray.Dataset.

    Intended for categorical / integer-coded maps such as region labels.
    """
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    da = ds[var_name]

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"Variable '{var_name}' must have dimensions ('lat', 'lon') or include both lat and lon."
        )

    values = da.values
    valid_values = values[np.isfinite(values)]

    if valid_values.size == 0:
        raise ValueError(f"Variable '{var_name}' contains no finite values to plot.")

    categories = np.unique(valid_values.astype(int))
    ncat = len(categories)

    cmap = plt.get_cmap("tab20", ncat)
    bounds = np.arange(ncat + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    remapped = np.full(values.shape, np.nan, dtype=float)

    valid = np.isfinite(values)
    values_int = np.full(values.shape, -1, dtype=int)
    values_int[valid] = values[valid].astype(int)

    for idx, cat in enumerate(categories):
        remapped[values_int == cat] = idx

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    proj = get_projection(projection)

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

    mesh = ax.pcolormesh(
        ds["lon"].values,
        ds["lat"].values,
        remapped,
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
        ticks=np.arange(ncat),
    )
    cbar.ax.set_yticklabels([str(v) for v in categories])
    cbar.set_label(var_name)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(var_name)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)