"""
Test script to plot a polygonal region with grid points.

Usage:
    python test_polygon_region.py <dlon_dlat> [region_label ...]

Examples:
    python test_polygon_region.py 0.25
    python test_polygon_region.py 0.5 "sic"
    python test_polygon_region.py 2 ACC oSO
"""

import argparse
import matplotlib.pyplot as plt

from kinematicparcels.utilities.geographicalRegions import (
    get_region_by_label,
    RegionManager,
    plot_regions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a polygonal region with grid points overlay"
    )
    parser.add_argument(
        "spatial_resolution",
        type=float,
        help="Spatial resolution (dlon=dlat in degrees)",
    )
    parser.add_argument(
        "region_labels",
        nargs="*",
        default=["sic"],
        help="Region label(s) (default: sic)",
    )
    parser.add_argument(
        "--use-cartopy",
        action="store_true",
        default=True,
        help="Use Cartopy for plotting (default: True)",
    )
    parser.add_argument(
        "--no-cartopy",
        dest="use_cartopy",
        action="store_false",
        help="Disable Cartopy",
    )
    parser.add_argument(
        "--coastlines",
        action="store_true",
        default=True,
        help="Add coastlines (default: True)",
    )
    parser.add_argument(
        "--no-coastlines",
        dest="coastlines",
        action="store_false",
        help="Disable coastlines",
    )
    parser.add_argument(
        "--coastline-resolution",
        type=str,
        default="10m",
        choices=["10m", "50m", "110m"],
        help="Coastline resolution (default: 10m)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size (default: 12 10)",
    )

    args = parser.parse_args()

    # Get regions
    regions = [get_region_by_label(label) for label in args.region_labels]
    region_summary = ", ".join(
        f"{region.name} (label: {region.label})" for region in regions
    )
    print(f"Regions: {region_summary}")
    print(f"Spatial resolution: dlon={args.spatial_resolution}, dlat={args.spatial_resolution}")

    # Create region manager
    region_manager = RegionManager(regions)

    title_labels = ", ".join(region.label for region in regions)

    # Plot
    fig, ax = plot_regions(
        region_manager,
        labels=True,
        show_bounds=False,
        show_polygons=True,
        show_bbox=True,
        show_points=True,
        point_dlon=args.spatial_resolution,
        point_dlat=args.spatial_resolution,
        point_size=15,
        point_color="red",
        ax=None,
        figsize=tuple(args.figsize),
        title=(
            f"Polygonal Regions: {title_labels} "
            f"(resolution: {args.spatial_resolution}°)"
        ),
        linewidth=2.0,
        polygon_linestyle="-",
        use_cartopy=args.use_cartopy,
        add_coastlines=args.coastlines,
        coastline_resolution=args.coastline_resolution,
    )

    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    main()
