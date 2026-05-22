from kinematicparcels.utilities.compare_region_shape import compare_rectangle_and_polygon_region
import sys

if __name__ == "__main__":
    reg=sys.argv[2] if len(sys.argv) > 2 else "sic"
    dl = float(sys.argv[1]) if len(sys.argv) > 2 else 0.25
    compare_rectangle_and_polygon_region(reg, show_points=True, point_dlon=dl, point_dlat=dl, use_cartopy=True, add_coastlines=True, coastline_resolution="10m", show_bbox=True, figsize=(12, 10))