from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_trajectories_map(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Trajectories",
    show_start: bool = True,
    show_end: bool = True,
    figsize: tuple[float, float] = (12, 8),
    linewidth: float = 0.8,
    alpha: float = 0.7,
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
) -> None:
    """
    Plot trajectories on a simple geographic map.

    Parameters
    ----------
    df
        Clean trajectory table with at least:
        trajectory, obs, lon, lat
    outpath
        Output figure path.
    title
        Figure title.
    show_start
        If True, plot the first point of each trajectory.
    show_end
        If True, plot the last point of each trajectory.
    figsize
        Figure size in inches.
    linewidth
        Line width for trajectories.
    alpha
        Line transparency.
    add_land
        If True, add land feature.
    add_coastlines
        If True, add coastlines.
    add_gridlines
        If True, add lat/lon gridlines.
    """
    required = ["trajectory", "obs", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input dataframe missing required columns for plotting: {missing}"
        )

    if df.empty:
        raise ValueError("Input dataframe is empty. Nothing to plot.")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["trajectory", "obs"]).reset_index(drop=True)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    if add_land:
        # ax.add_feature(cfeature.LAND, zorder=0)
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

    for _, g in df.groupby("trajectory", sort=False):
        ax.plot(
            g["lon"].to_numpy(),
            g["lat"].to_numpy(),
            transform=ccrs.PlateCarree(),
            linewidth=linewidth,
            alpha=alpha,
        )

        if show_start:
            first = g.iloc[0]
            ax.scatter(
                first["lon"],
                first["lat"],
                transform=ccrs.PlateCarree(),
                s=10,
                marker="o",
                zorder=3,
            )

        if show_end:
            last = g.iloc[-1]
            ax.scatter(
                last["lon"],
                last["lat"],
                transform=ccrs.PlateCarree(),
                s=12,
                marker="x",
                zorder=3,
            )

    lon_min = df["lon"].min()
    lon_max = df["lon"].max()
    lat_min = df["lat"].min()
    lat_max = df["lat"].max()

    lon_pad = max(0.5, 0.05 * (lon_max - lon_min if lon_max > lon_min else 1.0))
    lat_pad = max(0.5, 0.05 * (lat_max - lat_min if lat_max > lat_min else 1.0))

    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=ccrs.PlateCarree(),
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)