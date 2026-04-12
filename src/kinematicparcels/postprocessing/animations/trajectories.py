from __future__ import annotations

from pathlib import Path
import tempfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
import xarray as xr  # not required but harmless if already in env

from ..plotting.projections import get_projection
from .utils import (
    add_time_progress_bar,
    build_animation_colormap,
)


def animate_trajectories(
    trajectory_df: pd.DataFrame,
    *,
    outpath: str | Path,
    projection: str = "PlateCarree",
    fps: int = 8,
    title: str = "Trajectories",
    color_by: str = "lat0",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_time_bar: bool = True,
    trail: bool = True,
    trail_steps: int | None = None,
    figsize: tuple[float, float] = (12, 8),
    add_land: bool = True,
    add_coastlines: bool = True,
    add_gridlines: bool = True,
    summary_df: pd.DataFrame | None = None,
    max_group_member: int | None = None,
) -> Path:
    """
    Animate trajectories as moving particle positions on a map.

    Color logic:
    - if color_by is in summary_df, color is fixed per trajectory
    - elif color_by is in trajectory_df, color is taken from the current frame
    
    Parameters
    ----------
    max_group_member
        If set and group_member column exists, animate only members <= max_group_member.
        If None, animate all available members.
    """
    required = ["trajectory", "obs", "time", "lon", "lat"]
    missing = [c for c in required if c not in trajectory_df.columns]
    if missing:
        raise KeyError(f"Trajectory dataframe missing required columns: {missing}")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = trajectory_df.copy()
    df["time"] = pd.to_datetime(df["time"])
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

    times = np.sort(df["time"].dropna().unique())
    if len(times) == 0:
        raise ValueError("No time steps available for trajectory animation.")

    source = None
    color_values_summary = None

    if summary_df is not None and color_by in summary_df.columns:
        source = "summary"
        if "trajectory" not in summary_df.columns:
            raise KeyError("summary_df must contain 'trajectory' column.")
        color_values_summary = (
            summary_df[["trajectory", color_by]]
            .drop_duplicates(subset=["trajectory"])
            .set_index("trajectory")[color_by]
        )

    elif color_by in df.columns:
        source = "trajectory"

    else:
        raise KeyError(
            f"animation_color_by='{color_by}' not found in summary_df or trajectory_df."
        )

    if source == "summary":
        all_values = color_values_summary.to_numpy(dtype=float)
    else:
        all_values = df[color_by].to_numpy(dtype=float)

    finite = np.isfinite(all_values)
    if not finite.any():
        raise ValueError(f"No finite values found for color variable '{color_by}'.")

    if vmin is None:
        vmin = float(np.nanmin(all_values))
    if vmax is None:
        vmax = float(np.nanmax(all_values))
    if vmax < vmin:
        raise ValueError("vmax must be greater than or equal to vmin.")

    cmap, norm = build_animation_colormap(
        cmap_name="viridis",
        under_color="magenta",
        over_color="red",
        vmin=vmin,
        vmax=vmax,
    )

    proj = get_projection(projection)
    colorbar_label = colorbar_label or color_by

    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())
    lon_pad = max(0.5, 0.05 * (lon_max - lon_min if lon_max > lon_min else 1.0))
    lat_pad = max(0.5, 0.05 * (lat_max - lat_min if lat_max > lat_min else 1.0))

    grouped = {tid: g.copy() for tid, g in df.groupby("trajectory", sort=False)}

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

            xs = []
            ys = []
            cs = []

            for tid, g in grouped.items():
                g_now = g.loc[g["time"] == time_value]
                if g_now.empty:
                    continue

                row = g_now.iloc[0]

                if trail:
                    g_past = g.loc[g["time"] <= time_value]
                    if trail_steps is not None:
                        g_past = g_past.tail(trail_steps)

                    if len(g_past) >= 2:
                        ax.plot(
                            g_past["lon"].to_numpy(),
                            g_past["lat"].to_numpy(),
                            transform=ccrs.PlateCarree(),
                            linewidth=0.8,
                            alpha=0.35,
                            color="0.4",
                            zorder=2,
                        )

                xs.append(float(row["lon"]))
                ys.append(float(row["lat"]))

                if source == "summary":
                    cs.append(float(color_values_summary.loc[tid]))
                else:
                    cs.append(float(row[color_by]))

            if len(xs) > 0:
                sc = ax.scatter(
                    xs,
                    ys,
                    c=cs,
                    cmap=cmap,
                    norm=norm,
                    s=18,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )

                cbar = plt.colorbar(
                    ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax,
                    shrink=0.9,
                    pad=0.03,
                    extend="both",
                )
                cbar.set_label(colorbar_label)

            ax.set_extent(
                [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
                crs=ccrs.PlateCarree(),
            )

            ax.set_title(title)

            if show_time_bar:
                add_time_progress_bar(
                    fig,
                    current_time=time_value,
                    time_min=times[0],
                    time_max=times[-1],
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