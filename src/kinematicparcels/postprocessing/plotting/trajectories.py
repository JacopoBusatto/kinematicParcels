from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .projections import get_projection



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
    projection: str = "PlateCarree",
    max_group_member: int | None = None,
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
    max_group_member
        If set and group_member column exists, plot only members <= max_group_member.
        If None, plot all available members.
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

    # =========================================================================
    # GROUPED TRAJECTORIES: Filter by max_group_member if present
    # =========================================================================
    has_group_member = "group_member" in df.columns
    if has_group_member and max_group_member is not None:
        # Filter to keep only group members 1..max_group_member
        df = df[df["group_member"] <= max_group_member].copy()
        if df.empty:
            raise ValueError(
                f"No trajectories found with group_member <= {max_group_member}"
            )

    fig = plt.figure(figsize=figsize)
    proj = get_projection(projection)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)

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

    # =========================================================================
    # COLORING: Use group_member for color if available, else monochrome
    # =========================================================================
    if has_group_member:
        # Color by group_member: aqua, coldfusio, distinct colors
        group_members = sorted(df["group_member"].unique())
        n_members = len(group_members)
        cmap = plt.cm.get_cmap("tab10" if n_members <= 10 else "hsv")
        member_to_color = {
            m: cmap((i / (n_members - 1)) if n_members > 1 else 0)
            for i, m in enumerate(group_members)
        }

        for traj_id, g in df.groupby("trajectory", sort=False):
            for member in group_members:
                g_member = g[g["group_member"] == member]
                if len(g_member) > 0:
                    ax.plot(
                        g_member["lon"].to_numpy(),
                        g_member["lat"].to_numpy(),
                        transform=ccrs.PlateCarree(),
                        color=member_to_color[member],
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
    else:
        # Standard mode: all trajectories one color
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

    lon_pad = min(0.5, 0.05 * (lon_max - lon_min if lon_max > lon_min else 1.0))
    lat_pad = min(0.5, 0.05 * (lat_max - lat_min if lat_max > lat_min else 1.0))

    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=ccrs.PlateCarree(),
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)