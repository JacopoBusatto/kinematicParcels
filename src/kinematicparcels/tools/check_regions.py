from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kinematicparcels.regions.core import ALL_REGIONS, RegionManager  # noqa: E402


def add_region_columns(
    df,
    lon_col="lonf",
    lat_col="latf",
    labels=None,
    how_many="priority_max",
    priority_level=None,
    priority_mode="exact",
):
    selected = ALL_REGIONS if labels is None else [r for r in ALL_REGIONS if r.label in labels]
    regions = RegionManager(selected)
    priority_by_label = {r.label: r.priority for r in selected}
    out = df.copy()
    found = [
        regions.find_regions(
            row[lon_col],
            row[lat_col],
            howMany=how_many,
            priority_level=priority_level,
            priority_mode=priority_mode,
        )
        for _, row in out.iterrows()
    ]
    found = [r[0] if isinstance(r, list) and r else r for r in found]
    out["checked_region"] = [None if r is None else r["label"] for r in found]
    out["checked_numericLabel"] = [None if r is None else r["numericLabel"] for r in found]
    out["checked_priority"] = out["checked_region"].map(priority_by_label)
    return out


def plot_missing_region_points(
    df,
    *,
    lon_col="lonf",
    lat_col="latf",
    region_col="checked_region",
    output_path="missing_region_points.png",
    coastline_resolution="10m",
    pad_fraction=0.08,
    figsize=(10, 8),
    show=False,
):
    missing = df[df[region_col].isna()].dropna(subset=[lon_col, lat_col])
    if missing.empty:
        print("No missing-region rows with valid lon/lat to plot.")
        return None, None

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": data_crs})

    ax.add_feature(cfeature.LAND.with_scale(coastline_resolution), facecolor="0.90")
    ax.add_feature(cfeature.OCEAN.with_scale(coastline_resolution), facecolor="0.96")
    ax.add_feature(cfeature.COASTLINE.with_scale(coastline_resolution), linewidth=0.8)

    ax.scatter(
        missing[lon_col],
        missing[lat_col],
        s=18,
        c="tab:red",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.25,
        transform=data_crs,
        zorder=4,
        label=f"NaN region ({len(missing)})",
    )

    lon_min = missing[lon_col].min()
    lon_max = missing[lon_col].max()
    lat_min = missing[lat_col].min()
    lat_max = missing[lat_col].max()
    dx = lon_max - lon_min
    dy = lat_max - lat_min
    xpad = pad_fraction * dx if dx > 0 else 0.5
    ypad = pad_fraction * dy if dy > 0 else 0.5
    ax.set_extent(
        [lon_min - xpad, lon_max + xpad, lat_min - ypad, lat_max + ypad],
        crs=data_crs,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("Rows with NaN checked_region")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved missing-region map: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check final positions against region definitions."
    )
    parser.add_argument("path", type=Path, help="Input parquet table.")
    parser.add_argument(
        "--missing-map",
        type=Path,
        default=Path("missing_region_points.png"),
        help=(
            "Path for a Cartopy scatter map of rows with NaN checked_region. "
            "Use --no-missing-map to skip it."
        ),
    )
    parser.add_argument(
        "--no-missing-map",
        dest="missing_map",
        action="store_const",
        const=None,
        help="Skip plotting rows with NaN checked_region.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the missing-region plot interactively after saving it.",
    )
    parser.add_argument(
        "--coastline-resolution",
        choices=["10m", "50m", "110m"],
        default="10m",
        help="Natural Earth coastline resolution for the missing-region map.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    df = pd.read_parquet(args.path)
    cols = ["lonf", "latf"] + (["end_region"] if "end_region" in df.columns else [])
    out = add_region_columns(df[cols], priority_level=6, priority_mode="exact")
    print(out[out["checked_region"].isna()].head(50))
    print(out["checked_region"].value_counts(dropna=False))
    if args.missing_map is not None:
        plot_missing_region_points(
            out,
            output_path=args.missing_map,
            coastline_resolution=args.coastline_resolution,
            show=args.show,
        )
