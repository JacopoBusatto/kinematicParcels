from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from .projections import get_projection


def _normalize_key_value(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return _normalize_key_value(value.item() if value.ndim == 0 else value.reshape(-1)[0])
        return tuple(_normalize_key_value(v) for v in value.tolist())

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _normalize_key_value(value[0])
        return tuple(_normalize_key_value(v) for v in value)

    return value


def _normalize_key_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(_normalize_key_value)
    return out


def _split_longitude_wrapped_path(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    max_lon_step: float = 180.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split a path into contiguous line segments across periodic longitude wraps.

    A jump such as 179 -> -179 is a normal periodic wrap in the data, but when
    plotted as a single polyline it becomes a long segment across the figure.
    Returning explicit segments avoids passing NaN separators into Cartopy /
    Shapely, which emits warnings when constructing projected line strings.
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)

    if lon_arr.ndim != 1 or lat_arr.ndim != 1:
        raise ValueError("lon and lat must be 1D arrays.")
    if lon_arr.shape != lat_arr.shape:
        raise ValueError("lon and lat must have the same shape.")
    if lon_arr.size < 2:
        return [(lon_arr.copy(), lat_arr.copy())]

    valid = np.isfinite(lon_arr[:-1]) & np.isfinite(lon_arr[1:])
    jump_idx = np.flatnonzero(valid & (np.abs(np.diff(lon_arr)) > max_lon_step))
    if jump_idx.size == 0:
        return [(lon_arr.copy(), lat_arr.copy())]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0

    for idx in jump_idx:
        stop = idx + 1
        segments.append((lon_arr[start:stop].copy(), lat_arr[start:stop].copy()))
        start = stop

    segments.append((lon_arr[start:].copy(), lat_arr[start:].copy()))
    return segments


def _resolve_color_lookup(
    df: pd.DataFrame,
    *,
    color_by: str | None,
    summary_df: pd.DataFrame | None,
    key_cols: list[str],
) -> tuple[pd.Series | None, list[str]]:
    if not color_by:
        return None, ["trajectory"]

    if summary_df is not None and color_by in summary_df.columns:
        if "trajectory" not in summary_df.columns:
            raise KeyError("summary_df must contain 'trajectory' column.")
        lookup_cols = key_cols if all(c in summary_df.columns for c in key_cols) else ["trajectory"]
        summary_norm = _normalize_key_columns(summary_df, lookup_cols)
        lookup = (
            summary_norm[lookup_cols + [color_by]]
            .drop_duplicates(subset=lookup_cols)
            .set_index(lookup_cols)[color_by]
        )
        return lookup, lookup_cols

    if color_by in df.columns:
        df_norm = _normalize_key_columns(df, key_cols)
        lookup = (
            df_norm[key_cols + [color_by]]
            .drop_duplicates(subset=key_cols)
            .set_index(key_cols)[color_by]
        )
        return lookup, key_cols

    raise KeyError(f"color_by='{color_by}' not found in summary_df or trajectory_df.")


def _build_colorizer(values: pd.Series, *, cmap_name: str | None = None, cmap_mode: str = "auto") -> dict:
    non_null = values.dropna()
    if non_null.empty:
        raise ValueError("No valid values found for requested coloring variable.")

    numeric = pd.to_numeric(non_null, errors="coerce")
    auto_categorical = not numeric.notna().all()

    if cmap_mode == "categorical":
        categorical = True
    elif cmap_mode == "numeric":
        categorical = False
    else:
        categorical = auto_categorical

    if not categorical:
        full_numeric = pd.to_numeric(values, errors="coerce")
        vmin = float(full_numeric.min())
        vmax = float(full_numeric.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        cmap = plt.get_cmap(cmap_name if cmap_name is not None else "viridis")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def _to_color_numeric(value):
            if pd.isna(value):
                return (0.5, 0.5, 0.5, 1.0)
            return cmap(norm(float(value)))

        return {
            "kind": "numeric",
            "to_color": _to_color_numeric,
            "cmap": cmap,
            "norm": norm,
        }

    categories = [str(v) for v in pd.unique(non_null.astype(str))]
    if cmap_name is not None:
        base_cmap = plt.get_cmap(cmap_name)
    else:
        base_cmap = plt.get_cmap(
            "tab10" if len(categories) <= 10 else "tab20" if len(categories) <= 20 else "hsv"
        )
    if hasattr(base_cmap, "colors") and len(getattr(base_cmap, "colors", [])) >= len(categories):
        palette = [base_cmap.colors[i] for i in range(len(categories))]
    else:
        denom = max(len(categories) - 1, 1)
        palette = [base_cmap((i / denom) if len(categories) > 1 else 0) for i in range(len(categories))]
    category_colors = {cat: palette[i] for i, cat in enumerate(categories)}
    listed_cmap = mcolors.ListedColormap(palette, name="plot_categories")
    cat_norm = mcolors.BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))
    category_to_code = {cat: i for i, cat in enumerate(categories)}

    def _to_color_categorical(value):
        if pd.isna(value):
            return (0.5, 0.5, 0.5, 1.0)
        return category_colors.get(str(value), (0.5, 0.5, 0.5, 1.0))

    return {
        "kind": "categorical",
        "to_color": _to_color_categorical,
        "categories": category_colors,
        "cmap": listed_cmap,
        "norm": cat_norm,
        "category_to_code": category_to_code,
    }


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
    summary_df: pd.DataFrame | None = None,
    color_by: str | None = None,
    colorbar_label: str | None = None,
    cmap_name: str | None = None,
    cmap_mode: str = "auto",
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
    summary_df
        Optional one-row-per-trajectory summary used to resolve trajectory colors.
    color_by
        Optional column from summary_df or df used to color the trajectories.
        Supports both numeric and categorical values.
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

    color_lookup, lookup_cols = _resolve_color_lookup(
        df,
        color_by=color_by,
        summary_df=summary_df,
        key_cols=group_cols,
    )
    colorizer = _build_colorizer(color_lookup, cmap_name=cmap_name, cmap_mode=cmap_mode) if color_lookup is not None else None

    fig = plt.figure(figsize=figsize)
    proj = get_projection(projection)
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

    if colorizer is None and has_group_member:
        group_members = sorted(df["group_member"].unique())
        n_members = len(group_members)
        cmap = plt.get_cmap("tab10" if n_members <= 10 else "hsv")
        member_to_color = {
            m: cmap((i / (n_members - 1)) if n_members > 1 else 0)
            for i, m in enumerate(group_members)
        }
    else:
        member_to_color = None

    for _, g in df.groupby(group_cols, sort=False):
        if colorizer is not None:
            assert color_lookup is not None
            lookup_key = tuple(_normalize_key_value(g[col].iloc[0]) for col in lookup_cols)
            if len(lookup_cols) == 1:
                lookup_key = lookup_key[0]

            if lookup_key in color_lookup.index:
                color_value = color_lookup.loc[lookup_key]
                if isinstance(color_value, pd.Series):
                    color_value = color_value.iloc[0]
            else:
                color_value = np.nan

            color = colorizer["to_color"](color_value)
        elif member_to_color is not None:
            color = member_to_color[g["group_member"].iloc[0]]
        else:
            color = None

        plot_segments = _split_longitude_wrapped_path(
            g["lon"].to_numpy(),
            g["lat"].to_numpy(),
        )

        for plot_lon, plot_lat in plot_segments:
            if len(plot_lon) < 2:
                continue
            ax.plot(
                plot_lon,
                plot_lat,
                transform=ccrs.PlateCarree(),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

        if show_start:
            first = g.iloc[0]
            ax.scatter(
                first["lon"],
                first["lat"],
                transform=ccrs.PlateCarree(),
                s=16 if colorizer is not None else 10,
                marker="o",
                color=color,
                edgecolors="black" if color is not None else None,
                linewidths=0.4 if color is not None else 0.0,
                alpha=alpha,
                zorder=4,
            )

        if show_end:
            last = g.iloc[-1]
            ax.scatter(
                last["lon"],
                last["lat"],
                transform=ccrs.PlateCarree(),
                s=22 if color is not None else 12,
                marker="x",
                color=color,
                linewidths=1.0 if color is not None else 0.8,
                alpha=alpha,
                zorder=5,
            )

    if colorizer is not None:
        if colorizer["kind"] == "numeric":
            cbar = plt.colorbar(
                ScalarMappable(norm=colorizer["norm"], cmap=colorizer["cmap"]),
                ax=ax,
                shrink=0.9,
                pad=0.03,
            )
            cbar.set_label(colorbar_label or color_by or "value")
        else:
            cbar = plt.colorbar(
                ScalarMappable(norm=colorizer["norm"], cmap=colorizer["cmap"]),
                ax=ax,
                shrink=0.9,
                pad=0.03,
            )
            cbar.set_label(colorbar_label or color_by or "category")
            categories = list(colorizer["categories"].keys())
            cbar.set_ticks(np.arange(len(categories)))
            cbar.set_ticklabels(categories)

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


def plot_connectivity_map(
    traj_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outpath: str | Path,
    *,
    start_color_by: str = "start_region",
    end_color_by: str = "end_region",
    title: str = "Connectivity map",
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
    Plot a dual-coloured connectivity map.

    Each trajectory is drawn as a line coloured by its *end* region, with the
    start position marked by a circle coloured by the *start* region and the
    end position marked by a cross coloured by the *end* region.

    Both start and end regions share the same colour palette so the same
    region label always maps to the same colour regardless of whether it is
    the origin or the destination.

    Parameters
    ----------
    traj_df
        Trajectory table with at least: trajectory, obs, lon, lat.
        May be a full trajectory table or a two-point segment DataFrame
        (first row = start, last row = end).
    summary_df
        One-row-per-trajectory summary that must contain ``start_color_by``
        and ``end_color_by`` columns as well as a ``trajectory`` column.
    outpath
        Output figure path.
    start_color_by
        Column in ``summary_df`` used to colour the start marker.
    end_color_by
        Column in ``summary_df`` used to colour the line and end marker.
    title
        Figure title.
    max_group_member
        If set and ``group_member`` column exists, plot only members <=
        this value.
    """
    required_traj = ["trajectory", "obs", "lon", "lat"]
    missing = [c for c in required_traj if c not in traj_df.columns]
    if missing:
        raise KeyError(f"traj_df missing required columns: {missing}")

    for col in (start_color_by, end_color_by, "trajectory"):
        if col not in summary_df.columns:
            raise KeyError(f"summary_df missing required column: '{col}'")

    if traj_df.empty:
        raise ValueError("traj_df is empty. Nothing to plot.")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    has_group_member = "group_member" in traj_df.columns
    group_cols = ["trajectory"] + (["group_member"] if has_group_member else [])
    traj_df = _normalize_key_columns(traj_df, group_cols)
    traj_df = traj_df.sort_values(group_cols + ["obs"]).reset_index(drop=True)
    if has_group_member and max_group_member is not None:
        traj_df = traj_df[traj_df["group_member"] <= max_group_member].copy()
        if traj_df.empty:
            raise ValueError(f"No trajectories found with group_member <= {max_group_member}")

    # Build a single colour palette covering all region values so that the
    # same label always gets the same colour in both start and end positions.
    all_start = summary_df[start_color_by].dropna().astype(str)
    all_end = summary_df[end_color_by].dropna().astype(str)
    all_labels = pd.Series(sorted(set(all_start) | set(all_end)))
    colorizer = _build_colorizer(all_labels)

    # Build per-trajectory lookups indexed by trajectory id.
    summary_norm = _normalize_key_columns(summary_df, ["trajectory"])
    start_lookup = summary_norm.set_index("trajectory")[start_color_by]
    end_lookup = summary_norm.set_index("trajectory")[end_color_by]

    fig = plt.figure(figsize=figsize)
    proj = get_projection(projection)
    ax = plt.axes(projection=proj)

    if add_land:
        land = cfeature.NaturalEarthFeature(
            "physical", "land", "10m",
            edgecolor="none", facecolor=cfeature.COLORS["land"],
        )
        ax.add_feature(land, zorder=0)

    if add_coastlines:
        ax.coastlines(resolution="10m", linewidth=0.8)

    if add_gridlines:
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False

    for _, g in traj_df.groupby(group_cols, sort=False):
        traj_id = _normalize_key_value(g["trajectory"].iloc[0])

        raw_start_val = start_lookup.get(traj_id, np.nan)
        raw_end_val = end_lookup.get(traj_id, np.nan)
        if isinstance(raw_start_val, pd.Series):
            raw_start_val = raw_start_val.iloc[0]
        if isinstance(raw_end_val, pd.Series):
            raw_end_val = raw_end_val.iloc[0]

        start_color = colorizer["to_color"](raw_start_val)
        end_color = colorizer["to_color"](raw_end_val)

        plot_segments = _split_longitude_wrapped_path(
            g["lon"].to_numpy(),
            g["lat"].to_numpy(),
        )

        for plot_lon, plot_lat in plot_segments:
            if len(plot_lon) < 2:
                continue
            ax.plot(
                plot_lon,
                plot_lat,
                transform=ccrs.PlateCarree(),
                color=end_color,
                linewidth=linewidth,
                alpha=alpha,
            )

        first = g.iloc[0]
        ax.scatter(
            first["lon"], first["lat"],
            transform=ccrs.PlateCarree(),
            s=18,
            marker="o",
            color=start_color,
            edgecolors="black",
            linewidths=0.4,
            alpha=alpha,
            zorder=4,
        )

        last = g.iloc[-1]
        ax.scatter(
            last["lon"], last["lat"],
            transform=ccrs.PlateCarree(),
            s=24,
            marker="x",
            color=end_color,
            linewidths=1.2,
            alpha=alpha,
            zorder=5,
        )

    # Build legend: one entry per region (shared palette), plus marker-type
    # entries so the reader can distinguish start vs end positions.
    if colorizer["kind"] == "categorical":
        region_handles = [
            Line2D([0], [0], color=color, lw=2, marker="o", markersize=6, label=label)
            for label, color in colorizer["categories"].items()
        ]
        type_handles = [
            Line2D([0], [0], color="grey", lw=0, marker="o", markersize=6, label="start (○)"),
            Line2D([0], [0], color="grey", lw=1.0, marker="x", markersize=7, label="end (×) + line"),
        ]
        ax.legend(
            handles=region_handles + type_handles,
            title="region",
            loc="best",
            fontsize="small",
        )
    else:
        from matplotlib.cm import ScalarMappable
        cbar = plt.colorbar(
            ScalarMappable(norm=colorizer["norm"], cmap=colorizer["cmap"]),
            ax=ax, shrink=0.9, pad=0.03,
        )
        cbar.set_label(f"{end_color_by}")

    lon_min = traj_df["lon"].min()
    lon_max = traj_df["lon"].max()
    lat_min = traj_df["lat"].min()
    lat_max = traj_df["lat"].max()
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