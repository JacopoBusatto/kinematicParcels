import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

zarr_path = "output_NPstg.zarr"
title = "Parcels trajectories - NPstg"

ds = xr.open_zarr(zarr_path)

lon = ds["lon"].values
lat = ds["lat"].values

ntraj = lon.shape[0]

lon_all = lon[np.isfinite(lon)]
lat_all = lat[np.isfinite(lat)]

lonmin = float(np.min(lon_all))
lonmax = float(np.max(lon_all))
latmin = float(np.min(lat_all))
latmax = float(np.max(lat_all))

pad_lon = 5.0
pad_lat = 5.0

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

ax.set_extent(
    [lonmin - pad_lon, lonmax + pad_lon, latmin - pad_lat, latmax + pad_lat],
    crs=ccrs.PlateCarree(),
)

ax.add_feature(cfeature.LAND, zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=1)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle="--")
gl.top_labels = False
gl.right_labels = False

start_lons = []
start_lats = []
end_lons = []
end_lats = []

for i in range(ntraj):
    lo = lon[i, :]
    la = lat[i, :]

    mask = np.isfinite(lo) & np.isfinite(la)
    if mask.sum() == 0:
        continue

    lo_ok = lo[mask]
    la_ok = la[mask]

    ax.plot(
        lo_ok,
        la_ok,
        linewidth=0.6,
        alpha=0.35,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    start_lons.append(lo_ok[0])
    start_lats.append(la_ok[0])
    end_lons.append(lo_ok[-1])
    end_lats.append(la_ok[-1])

ax.scatter(
    start_lons,
    start_lats,
    s=14,
    marker="o",
    alpha=0.85,
    label="start",
    transform=ccrs.PlateCarree(),
    zorder=3,
)

ax.scatter(
    end_lons,
    end_lats,
    s=18,
    marker="x",
    alpha=0.9,
    label="end",
    transform=ccrs.PlateCarree(),
    zorder=4,
)

ax.set_title(title)
ax.legend()

outpng = "output_NPstg_cartopy.png"
plt.savefig(outpng, dpi=200, bbox_inches="tight")
print(f"Saved {outpng}")

plt.show()