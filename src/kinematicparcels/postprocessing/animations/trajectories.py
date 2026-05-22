from __future__ import annotations

from pathlib import Path
import tempfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
import xarray as xr  # not required but harmless if already in env

from ..plotting.projections import get_projection
from ..plotting.trajectories import (
    _normalize_key_columns,
    _normalize_key_value,
    _split_longitude_wrapped_path,
)
from .utils import (
    add_time_progress_bar,
    build_animation_colormap,
)


def _resolve_trail_color(*, show_tracer: bool, color_code, cmap, norm):
    if show_tracer:
        return "0.4"
    if pd.isna(color_code):
        return (0.5, 0.5, 0.5, 1.0)
    return cmap(norm(color_code))


def _snap_times_to_grid(df: pd.DataFrame, *, time_col: str = "time") -> pd.DataFrame:
    """
    Snap off-grid timestamps to the nearest detected regular time grid.

    Auto-detects the outputdt as the most common gap between consecutive unique
    timestamps. Timestamps that deviate from the regular grid (particle-deletion
    events written by Parcels at sub-outputdt precision) are rounded to the nearest
    grid point. Timestamps already on the grid are unchanged.
    """
    ts = pd.to_datetime(df[time_col].dropna().unique())
    if len(ts) < 2:
        return df

    sorted_ts = np.sort(ts)
    gaps_ns = np.diff(sorted_ts.astype(np.int64))
    gaps_ns = gaps_ns[gaps_ns > 0]
    if len(gaps_ns) == 0:
        return df

    # Use the most common gap as the outputdt (majority of obs are on-grid)
    unique_gaps, counts = np.unique(gaps_ns, return_counts=True)
    outputdt_ns = int(unique_gaps[np.argmax(counts)])
    if outputdt_ns <= 0:
        return df

    # Use the most frequent timestamp as the phase origin of the regular grid
    counts_by_ts = df[time_col].dropna().value_counts()
    origin_ns = int(pd.Timestamp(counts_by_ts.index[0]).value)

    not_null = df[time_col].notna()
    t_ns = df.loc[not_null, time_col].values.astype(np.int64)

    offset_ns = t_ns - origin_ns
    rounded_offset_ns = (np.round(offset_ns / outputdt_ns) * outputdt_ns).astype(np.int64)

    n_changed = int(np.sum(offset_ns != rounded_offset_ns))
    if n_changed == 0:
        return df

    df = df.copy()
    df.loc[not_null, time_col] = pd.to_datetime(origin_ns + rounded_offset_ns)
    return df


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
    cmap_name: str | None = None,
    cmap_mode: str = "auto",  # "auto" | "categorical" | "numeric"
    show_time_bar: bool = True,
    trail: bool = True,
    trail_steps: int | None = None,
    every_n: int = 1,
    show_tracer: bool = True,
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

    Supports both numeric and categorical variables.
    """
    required = ["trajectory", "obs", "time", "lon", "lat"]
    missing = [c for c in required if c not in trajectory_df.columns]
    if missing:
        raise KeyError(f"Trajectory dataframe missing required columns: {missing}")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = trajectory_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = _snap_times_to_grid(df)  # snap sub-outputdt deletion timestamps to nearest grid point

    has_group_member = "group_member" in df.columns
    group_cols = ["trajectory"] + (["group_member"] if has_group_member else [])
    df = _normalize_key_columns(df, group_cols)
    df = df.sort_values(group_cols + ["obs"]).reset_index(drop=True)
    if has_group_member and max_group_member is not None:
        df = df[df["group_member"] <= max_group_member].copy()
        if df.empty:
            raise ValueError(
                f"No trajectories found with group_member <= {max_group_member}"
            )

    if every_n < 1:
        raise ValueError("every_n must be >= 1.")
    times = np.sort(df["time"].dropna().unique())[::every_n]
    if len(times) == 0:
        raise ValueError("No time steps available for trajectory animation.")

    if summary_df is not None and color_by in summary_df.columns:
        if "trajectory" not in summary_df.columns:
            raise KeyError("summary_df must contain 'trajectory' column.")
        lookup_cols = group_cols if all(c in summary_df.columns for c in group_cols) else ["trajectory"]
        source = "summary"
        summary_norm = _normalize_key_columns(summary_df, lookup_cols)
        color_lookup = (
            summary_norm[lookup_cols + [color_by]]
            .drop_duplicates(subset=lookup_cols)
            .set_index(lookup_cols)[color_by]
        )
    elif color_by in df.columns:
        source = "trajectory"
        lookup_cols = group_cols
        color_lookup = None
    else:
        raise KeyError(
            f"animation_color_by='{color_by}' not found in summary_df or trajectory_df."
        )

    if source == "summary":
        raw_values = pd.Series(color_lookup.to_numpy(), dtype=object)
    else:
        raw_values = df[color_by]

    non_null = raw_values.dropna()
    if non_null.empty:
        raise ValueError(f"No valid values found for color variable '{color_by}'.")

    numeric = pd.to_numeric(non_null, errors="coerce")
    categorical_mode = not numeric.notna().all()

    # Allow the caller to override the auto-detected mode
    if cmap_mode == "categorical":
        categorical_mode = True
    elif cmap_mode == "numeric":
        categorical_mode = False

    colorbar_extend = "neither"  # overridden in numeric branch

    if categorical_mode:
        categories = [str(v) for v in pd.unique(non_null.astype(str))]
        if cmap_name is not None:
            base_cmap = plt.get_cmap(cmap_name)
        else:
            base_cmap = plt.get_cmap(
                "tab10" if len(categories) <= 10 else "tab20" if len(categories) <= 20 else "hsv"
            )
        if hasattr(base_cmap, "colors") and len(getattr(base_cmap, "colors", [])) >= len(categories):
            colors = [base_cmap.colors[i] for i in range(len(categories))]
        else:
            colors = [
                base_cmap((i / max(len(categories) - 1, 1)) if len(categories) > 1 else 0)
                for i in range(len(categories))
            ]
        cmap = mcolors.ListedColormap(colors, name="trajectory_categories")
        norm = mcolors.BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))
        category_to_code = {cat: i for i, cat in enumerate(categories)}
    else:
        all_values = pd.to_numeric(raw_values, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(all_values)
        if not finite.any():
            raise ValueError(f"No finite values found for color variable '{color_by}'.")

        data_min = float(np.nanmin(all_values[finite]))
        data_max = float(np.nanmax(all_values[finite]))

        # Save user intent before filling defaults (needed for smart extend)
        user_vmin = vmin
        user_vmax = vmax

        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
        if vmax < vmin:
            raise ValueError("vmax must be greater than or equal to vmin.")
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        # Extend the colorbar only where the user-supplied bound clips real data
        extend_lower = (user_vmin is not None) and (user_vmin > data_min)
        extend_upper = (user_vmax is not None) and (user_vmax < data_max)
        if extend_lower and extend_upper:
            colorbar_extend = "both"
        elif extend_lower:
            colorbar_extend = "min"
        elif extend_upper:
            colorbar_extend = "max"
        else:
            colorbar_extend = "neither"

        actual_cmap_name = cmap_name if cmap_name is not None else "viridis"
        cmap, norm = build_animation_colormap(
            cmap_name=actual_cmap_name,
            under_color="magenta",
            over_color="red",
            vmin=vmin,
            vmax=vmax,
        )
        category_to_code = None
        categories = None

    proj = get_projection(projection)
    colorbar_label = colorbar_label or color_by

    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())
    lon_pad = max(0.5, 0.05 * (lon_max - lon_min if lon_max > lon_min else 1.0))
    lat_pad = max(0.5, 0.05 * (lat_max - lat_min if lat_max > lat_min else 1.0))

    grouped = {tid: g.copy() for tid, g in df.groupby(group_cols, sort=False)}

    # Detect backward simulation: obs increases while time decreases
    is_backward = False
    _sample = next(iter(grouped.values())).sort_values("obs")
    if len(_sample) >= 2:
        is_backward = bool(_sample["time"].iloc[0] > _sample["time"].iloc[1])

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

                xs.append(float(row["lon"]))
                ys.append(float(row["lat"]))

                if source == "summary":
                    lookup_key = tid if isinstance(tid, tuple) else (_normalize_key_value(tid),)
                    if len(lookup_cols) == 1:
                        lookup_key = lookup_key[0]

                    if lookup_key in color_lookup.index:
                        raw_value = color_lookup.loc[lookup_key]
                        if isinstance(raw_value, pd.Series):
                            raw_value = raw_value.iloc[0]
                    else:
                        raw_value = np.nan
                else:
                    raw_value = row[color_by]

                if categorical_mode:
                    color_code = category_to_code.get(str(raw_value), np.nan) if pd.notna(raw_value) else np.nan
                else:
                    color_code = float(raw_value) if pd.notna(raw_value) else np.nan
                cs.append(color_code)

                if trail:
                    if is_backward:
                        g_past = g.loc[g["time"] >= time_value]
                    else:
                        g_past = g.loc[g["time"] <= time_value]
                    if trail_steps is not None:
                        g_past = g_past.tail(trail_steps)

                    if len(g_past) >= 2:
                        plot_segments = _split_longitude_wrapped_path(
                            g_past["lon"].to_numpy(),
                            g_past["lat"].to_numpy(),
                        )
                        for plot_lon, plot_lat in plot_segments:
                            if len(plot_lon) < 2:
                                continue
                            ax.plot(
                                plot_lon,
                                plot_lat,
                                transform=ccrs.PlateCarree(),
                                linewidth=0.8,
                                alpha=0.35,
                                color=_resolve_trail_color(
                                    show_tracer=show_tracer,
                                    color_code=color_code,
                                    cmap=cmap,
                                    norm=norm,
                                ),
                                zorder=2,
                            )

            if len(xs) > 0:
                if show_tracer:
                    ax.scatter(
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
                    extend=colorbar_extend if not categorical_mode else "neither",
                )
                cbar.set_label(colorbar_label)
                if categorical_mode and categories is not None:
                    cbar.set_ticks(np.arange(len(categories)))
                    cbar.set_ticklabels(categories)

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
        imageio.mimsave(outpath, images, duration=1000.0 / float(fps))

    return outpath