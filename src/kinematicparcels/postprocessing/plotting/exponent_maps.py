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
        nonzero_values = finite_values[finite_values != 0]
        if nonzero_values.size == 0:
            raise ValueError("Log-scale exponent-map plots require at least one finite non-zero value.")

        has_negative = bool(np.any(nonzero_values < 0))
        has_positive = bool(np.any(nonzero_values > 0))

        if has_negative and has_positive and cmap == "viridis":
            mesh_kwargs["cmap"] = "RdBu_r"

        if has_negative:
            abs_nonzero = np.abs(nonzero_values)
            auto_vmax = float(np.nanmax(abs_nonzero))
            auto_vmin = float(np.nanmin(abs_nonzero))

            if vmax is None and vmin is None:
                norm_vmin = -auto_vmax if has_positive else float(np.nanmin(nonzero_values))
                norm_vmax = auto_vmax if has_positive else float(np.nanmax(nonzero_values))
            else:
                norm_vmin = float(vmin) if vmin is not None else (-auto_vmax if has_positive else float(np.nanmin(nonzero_values)))
                norm_vmax = float(vmax) if vmax is not None else (auto_vmax if has_positive else float(np.nanmax(nonzero_values)))

            linthresh = max(auto_vmin, auto_vmax * 1.0e-6)
            mesh_kwargs["norm"] = mcolors.SymLogNorm(
                linthresh=linthresh,
                vmin=norm_vmin,
                vmax=norm_vmax,
                base=10,
            )
        else:
            positive_values = nonzero_values
            norm_vmin = float(vmin) if vmin is not None else float(np.nanmin(positive_values))
            norm_vmax = float(vmax) if vmax is not None else float(np.nanmax(positive_values))
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