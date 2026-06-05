from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .projections import get_projection


def plot_exponent_map(
    da: xr.DataArray,
    *,
    outpath: str | Path,
    projection: str = "PlateCarree",
    title: str = "",
    figsize: tuple[float, float] = (12, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    log_scale: bool = False,
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
) -> None:
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError("Exponent-map plots require a DataArray with lat and lon dimensions.")

    values = da.values.astype(float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Exponent-map plots require at least one finite value.")

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

    mesh_kwargs: dict[str, object] = {
        "transform": ccrs.PlateCarree(),
        "shading": "auto",
        "cmap": cmap,
    }
    if log_scale:
        positive_values = finite_values[finite_values > 0]
        if positive_values.size == 0:
            raise ValueError("Log-scale exponent-map plots require strictly positive values.")
        norm_vmin = vmin if vmin is not None else float(np.nanmin(positive_values))
        norm_vmax = vmax if vmax is not None else float(np.nanmax(positive_values))
        mesh_kwargs["norm"] = mcolors.LogNorm(vmin=norm_vmin, vmax=norm_vmax)
    else:
        mesh_kwargs["vmin"] = vmin
        mesh_kwargs["vmax"] = vmax

    mesh = ax.pcolormesh(
        da["lon"].values,
        da["lat"].values,
        values,
        **mesh_kwargs,
    )

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label(da.name or "value")

    ax.set_title(title or (da.name or "value"))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)