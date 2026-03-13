import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from kinematicparcels.utilities.geographicalRegions import Region, plot_regions, RegionManager
from kinematicparcels.utilities.regions_definitions import REGIONS_DATA_RECTANGLES, REGIONS_DATA


def _get_region_cfg_by_label(label, regions_data):
    for cfg in regions_data:
        if cfg.get("label") == label:
            return cfg
    return None


def compare_rectangle_and_polygon_region(
    label,
    *,
    show_points=False,
    point_dlon=None,
    point_dlat=None,
    use_cartopy=True,
    add_coastlines=True,
    coastline_resolution="10m",
    show_bbox=False,
    figsize=(10, 8),
):
    rect_cfg = _get_region_cfg_by_label(label, REGIONS_DATA_RECTANGLES)
    poly_cfg = _get_region_cfg_by_label(label, REGIONS_DATA)

    rect_region = None if rect_cfg is None else Region(**rect_cfg)
    poly_region = None if poly_cfg is None else Region(**poly_cfg)

    if rect_region is None:
        warnings.warn(f"Rectangular region with label '{label}' not found in REGIONS_DATA_RECTANGLES")
    if poly_region is None:
        warnings.warn(f"Polygonal region with label '{label}' not found in REGIONS_DATA")

    if rect_region is None and poly_region is None:
        raise ValueError(f"No region found for label '{label}'")

    # Base figure
    base_regions = [r for r in (rect_region, poly_region) if r is not None]
    fig, ax = plot_regions(
        RegionManager(base_regions),
        labels=False,
        show_bounds=False,
        show_polygons=False,
        show_bbox=False,
        show_points=False,
        use_cartopy=use_cartopy,
        add_coastlines=add_coastlines,
        coastline_resolution=coastline_resolution,
        figsize=figsize,
        title=f"Region comparison: {label}",
    )

    # Rectangles in blue dashed
    if rect_region is not None:
        plot_regions(
            RegionManager([rect_region]),
            labels=True,
            show_bounds=True,
            show_polygons=False,
            show_bbox=show_bbox,
            show_points=show_points,
            point_dlon=point_dlon,
            point_dlat=point_dlat,
            point_size=10,
            point_color="tab:blue",
            ax=ax,
            use_cartopy=use_cartopy,
            add_coastlines=False,
            linewidth=2.0,
            bounds_linestyle="--",
            title=f"Region comparison: {label}",
        )

    # Polygons in red solid
    if poly_region is not None:
        plot_regions(
            RegionManager([poly_region]),
            labels=True,
            show_bounds=False,
            show_polygons=True,
            show_bbox=show_bbox,
            show_points=show_points,
            point_dlon=point_dlon,
            point_dlat=point_dlat,
            point_size=10,
            point_color="tab:red",
            ax=ax,
            use_cartopy=use_cartopy,
            add_coastlines=False,
            linewidth=2.2,
            polygon_linestyle="-",
            title=f"Region comparison: {label}",
        )

    legend_items = []
    if rect_region is not None:
        legend_items.append(Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=2, label="Rectangles"))
    if poly_region is not None:
        legend_items.append(Line2D([0], [0], color="tab:red", linestyle="-", linewidth=2, label="Polygons"))
    if show_points:
        if rect_region is not None:
            legend_items.append(Line2D([0], [0], color="tab:blue", marker="o", linestyle="None", label="Rect grid"))
        if poly_region is not None:
            legend_items.append(Line2D([0], [0], color="tab:red", marker="o", linestyle="None", label="Poly grid"))

    if legend_items:
        ax.legend(handles=legend_items, loc="best")

    plt.show()
    return fig, ax