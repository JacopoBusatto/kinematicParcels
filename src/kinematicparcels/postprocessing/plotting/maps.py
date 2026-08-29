from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from .colorbar import colorbar_extend_from_limits
from .projections import get_projection


def _prepare_log_scaled_grid_values(
    values: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
) -> tuple[np.ma.MaskedArray, np.ndarray, mcolors.LogNorm]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    if not np.any(positive):
        raise ValueError("Log-scale grid maps require at least one positive value.")

    positive_values = values[positive]
    norm_vmin = float(vmin) if vmin is not None else float(np.min(positive_values))
    norm_vmax = float(vmax) if vmax is not None else float(np.max(positive_values))
    if norm_vmin <= 0.0 or norm_vmax <= 0.0:
        raise ValueError("Log-scale grid-map limits must be strictly positive.")
    if norm_vmin == norm_vmax:
        if vmin is None:
            norm_vmin = norm_vmin / 10.0
        elif vmax is None:
            norm_vmax = norm_vmax * 10.0
    if norm_vmin >= norm_vmax:
        raise ValueError(
            "Resolved log-scale grid-map vmin must be less than vmax."
        )

    plot_values = np.ma.masked_where(~positive, values)
    zero_mask = finite & (values == 0.0)
    return plot_values, zero_mask, mcolors.LogNorm(vmin=norm_vmin, vmax=norm_vmax)


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
    log_scale: bool = False,
    zero_color: str = "lightgray",
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
    colorbar_extend = colorbar_extend_from_limits(vmin=vmin, vmax=vmax)
    values = raw_values
    zero_mask = np.zeros(np.shape(raw_values), dtype=bool)
    cmap_for_plot = cmap
    if colorbar_extend != "neither":
        cmap_for_plot = plt.get_cmap(cmap).copy() if cmap is not None else plt.get_cmap().copy()
        if colorbar_extend in {"min", "both"}:
            cmap_for_plot.set_under("magenta")
        if colorbar_extend in {"max", "both"}:
            cmap_for_plot.set_over("red")
    norm = None
    if log_scale:
        values, zero_mask, norm = _prepare_log_scaled_grid_values(
            raw_values,
            vmin=vmin,
            vmax=vmax,
        )
    elif vmin is not None or vmax is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)

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
        cmap=cmap_for_plot,
        norm=norm,
    )

    if log_scale and np.any(zero_mask):
        zero_values = np.ma.masked_where(~zero_mask, np.ones_like(raw_values, dtype=float))
        ax.pcolormesh(
            ds["lon"].values,
            ds["lat"].values,
            zero_values,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=mcolors.ListedColormap([zero_color]),
            vmin=0.0,
            vmax=1.0,
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


def plot_point_map(
    df,
    *,
    lon_col: str,
    lat_col: str,
    value_col: str,
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
    Plot point values on a map using exact lon/lat columns.
    """
    required = [lon_col, lat_col, value_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError("vmin must be less than or equal to vmax.")

    work = df[required].dropna()
    if work.empty:
        raise ValueError(f"No finite point values available for '{value_col}'.")

    values = work[value_col].to_numpy(dtype=float)
    colorbar_extend = colorbar_extend_from_limits(vmin=vmin, vmax=vmax)
    cmap_for_plot = cmap
    if colorbar_extend != "neither":
        cmap_for_plot = plt.get_cmap(cmap).copy() if cmap is not None else plt.get_cmap().copy()
        if colorbar_extend in {"min", "both"}:
            cmap_for_plot.set_under("magenta")
        if colorbar_extend in {"max", "both"}:
            cmap_for_plot.set_over("red")

    norm = None
    if vmin is not None or vmax is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)

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

    scatter = ax.scatter(
        work[lon_col].to_numpy(dtype=float),
        work[lat_col].to_numpy(dtype=float),
        c=values,
        s=18,
        cmap=cmap_for_plot,
        norm=norm,
        transform=ccrs.PlateCarree(),
        linewidths=0.0,
        alpha=0.85,
        zorder=4,
    )

    cbar = plt.colorbar(
        scatter,
        ax=ax,
        shrink=0.9,
        pad=0.03,
        extend=colorbar_extend,
    )
    cbar.set_label(colorbar_label or value_col, fontsize=colorbar_fontsize)
    if colorbar_tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    if axis_tick_fontsize is not None:
        ax.tick_params(labelsize=axis_tick_fontsize)

    if title_fontsize != 0:
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(value_col, fontsize=title_fontsize)

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
