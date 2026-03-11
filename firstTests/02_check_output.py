import xarray as xr

ds = xr.open_zarr("output_run01.zarr")

print(ds)
print("\nDATA_VARS:")
print(list(ds.data_vars))
print("\nCOORDS:")
print(list(ds.coords))

print("\nlon:")
print(ds["lon"].values)

print("\nlat:")
print(ds["lat"].values)

print("\ntime:")
print(ds["time"].values)