from __future__ import annotations

from .core import RegionManager, make_regular_grid_in_region


def plot_regions(
    region_manager: RegionManager,
    *,
    labels=True,
    show_bounds=True,
    show_polygons=True,
    show_bbox=False,
    ax=None,
    figsize=(12, 8),
    title="Regions",
    linewidth=1.5,
    bounds_linestyle="--",
    polygon_linestyle="-",
    alpha_bounds=0.7,
    alpha_polygons=0.9,
    label_fontsize=9,
    pad_fraction=0.03,
    show_points=False,
    point_dlon=None,
    point_dlat=None,
    point_size=5,
    point_color="red",
    use_cartopy=True,
    add_coastlines=True,
    coastline_resolution="110m",
    add_land=False,
):
    """
    Plot the regions contained in a RegionManager.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon as MplPolygon

    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        data_crs = ccrs.PlateCarree()
    else:
        data_crs = None

    created_fig = False
    if ax is None:
        if use_cartopy:
            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={"projection": data_crs},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    global_lon_min = []
    global_lon_max = []
    global_lat_min = []
    global_lat_max = []

    if use_cartopy:
        if add_land:
            ax.add_feature(cfeature.LAND, zorder=0)
            ax.add_feature(cfeature.OCEAN, zorder=0)
        if add_coastlines:
            ax.coastlines(resolution=coastline_resolution, linewidth=0.8)

    for region in region_manager.get_regions():
        lon_min, lon_max, lat_min, lat_max = region.get_bbox()
        global_lon_min.append(lon_min)
        global_lon_max.append(lon_max)
        global_lat_min.append(lat_min)
        global_lat_max.append(lat_max)

        if show_bounds:
            for bounds in region.bounds:
                lon_min_list = bounds["lon_min"]
                lon_max_list = bounds["lon_max"]
                lat_min_list = bounds["lat_min"]
                lat_max_list = bounds["lat_max"]

                for b_lon_min, b_lon_max, b_lat_min, b_lat_max in zip(
                    lon_min_list, lon_max_list, lat_min_list, lat_max_list
                ):
                    rect = Rectangle(
                        (b_lon_min, b_lat_min),
                        b_lon_max - b_lon_min,
                        b_lat_max - b_lat_min,
                        fill=False,
                        linestyle=bounds_linestyle,
                        linewidth=linewidth,
                        alpha=alpha_bounds,
                        transform=data_crs if use_cartopy else None,
                    )
                    ax.add_patch(rect)

        if show_polygons:
            for poly in region.polygons:
                if len(poly) < 3:
                    continue

                poly_xy = poly if poly[0] == poly[-1] else list(poly) + [poly[0]]

                patch = MplPolygon(
                    poly_xy,
                    closed=True,
                    fill=False,
                    linestyle=polygon_linestyle,
                    linewidth=linewidth,
                    alpha=alpha_polygons,
                    transform=data_crs if use_cartopy else None,
                )
                ax.add_patch(patch)

        if show_bbox:
            rect = Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                fill=False,
                linestyle=":",
                linewidth=max(1.0, linewidth - 0.3),
                alpha=0.8,
                transform=data_crs if use_cartopy else None,
            )
            ax.add_patch(rect)

        if show_points:
            if point_dlon is None or point_dlat is None:
                raise ValueError("For show_points you must specify point_dlon and point_dlat")

            lons, lats = make_regular_grid_in_region(
                region,
                point_dlon,
                point_dlat,
                output_lon_mode=region.lon_mode,
            )

            ax.scatter(
                lons,
                lats,
                s=point_size,
                c=point_color,
                alpha=0.7,
                zorder=3,
                transform=data_crs if use_cartopy else None,
            )

        if labels:
            x_text = 0.5 * (lon_min + lon_max)
            y_text = 0.5 * (lat_min + lat_max)

            ax.text(
                x_text,
                y_text,
                region.label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.6),
                transform=data_crs if use_cartopy else None,
            )

    if global_lon_min:
        xmin = min(global_lon_min)
        xmax = max(global_lon_max)
        ymin = min(global_lat_min)
        ymax = max(global_lat_max)

        dx = xmax - xmin
        dy = ymax - ymin

        xpad = pad_fraction * dx if dx > 0 else 1.0
        ypad = pad_fraction * dy if dy > 0 else 1.0

        if use_cartopy:
            ax.set_extent([xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad], crs=data_crs)
        else:
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_title(title)

    if not use_cartopy:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    else:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False

    if created_fig:
        plt.tight_layout()

    return fig, ax


__all__ = [
    "plot_regions",
]
