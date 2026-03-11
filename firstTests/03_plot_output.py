import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ds = xr.open_zarr("output_NPstg.zarr")

lon = ds["lon"].isel(trajectory=0).values
lat = ds["lat"].isel(trajectory=0).values
time = ds["time"].isel(trajectory=0).values

mask = np.isfinite(lon) & np.isfinite(lat)

plt.figure(figsize=(7, 5))
plt.plot(lon[mask], lat[mask], "-o")
plt.scatter(lon[mask][0], lat[mask][0], s=80, label="start")
plt.scatter(lon[mask][-1], lat[mask][-1], s=80, label="end")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traiettoria della prima particella")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()