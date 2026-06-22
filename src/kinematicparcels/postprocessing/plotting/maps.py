from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from .colorbar import infer_colorbar_extend
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
    cmap: str | None = None,
    colorbar_label: str | None = None,
    title_fontsize: int | None = None,
    colorbar_fontsize: int | None = None,
    colorbar_tick_fontsize: int | None = None,
    axis_tick_fontsize: int | None = None,
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

    raw_values = da.values
    colorbar_extend = infer_colorbar_extend(raw_values, vmin=vmin, vmax=vmax)
    values = raw_values
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
        if axis_tick_fontsize is not None:
            gl.xlabel_style = {"size": axis_tick_fontsize}
            gl.ylabel_style = {"size": axis_tick_fontsize}

    mesh = ax.pcolormesh(
        ds["lon"].values,
        ds["lat"].values,
        values,
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        shrink=0.9,
        pad=0.03,
        extend=colorbar_extend,
    )
    cbar.set_label(colorbar_label or var_name, fontsize=colorbar_fontsize)
    if colorbar_tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    if axis_tick_fontsize is not None:
        ax.tick_params(labelsize=axis_tick_fontsize)

    if title_fontsize != 0:
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(var_name, fontsize=title_fontsize)

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
    cmap_name: str | None = None,
    colorbar_label_mode: str = "numeric",
    category_label_map: dict[int, dict[str, str]] | None = None,
    show_labels: bool = False,
    axis_tick_fontsize: int | None = None,
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

    cmap = plt.get_cmap(cmap_name or "tab20", ncat)
    bounds = np.arange(ncat + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    remapped = np.full(values.shape, np.nan, dtype=float)

    valid = np.isfinite(values)
    values_int = np.full(values.shape, -1, dtype=int)
    values_int[valid] = values[valid].astype(int)

    for idx, cat in enumerate(categories):
        remapped[values_int == cat] = idx

    if colorbar_label_mode not in {"numeric", "region_label", "region_name"}:
        raise ValueError(
            "colorbar_label_mode must be one of: 'numeric', 'region_label', 'region_name'."
        )

    def _display_label(cat: int) -> str:
        if colorbar_label_mode == "numeric":
            return str(cat)
        if category_label_map is None:
            return str(cat)
        meta = category_label_map.get(int(cat), {})
        if colorbar_label_mode == "region_label":
            return str(meta.get("label", cat))
        return str(meta.get("name", cat))

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
        if axis_tick_fontsize is not None:
            gl.xlabel_style = {"size": axis_tick_fontsize}
            gl.ylabel_style = {"size": axis_tick_fontsize}

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
    cbar.ax.set_yticklabels([_display_label(int(v)) for v in categories])
    cbar.set_label(var_name)

    if axis_tick_fontsize is not None:
        ax.tick_params(labelsize=axis_tick_fontsize)

    if show_labels:
        lon_vals = ds["lon"].values
        lat_vals = ds["lat"].values

        # Draw one annotation per category using the median cell center.
        for cat in categories:
            valid_mask = np.isfinite(values)
            values_int_safe = np.full(values.shape, -1, dtype=int)
            values_int_safe[valid_mask] = values[valid_mask].astype(int)
            mask = valid_mask & (values_int_safe == int(cat))
            if not np.any(mask):
                continue

            jj, ii = np.where(mask)
            label_lon = float(np.median(lon_vals[ii]))
            label_lat = float(np.median(lat_vals[jj]))

            ax.text(
                label_lon,
                label_lat,
                _display_label(int(cat)),
                transform=ccrs.PlateCarree(),
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                alpha=0.9,
                zorder=6,
            )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(var_name)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
