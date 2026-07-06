import numpy as np
import pandas as pd
import xarray as xr

ds = xr.open_dataset(r"F:\ARGO\netcdf\Rtraj\core\7902380_Rtraj.nc")

juld = ds["JULD"].values
cycles = ds["CYCLE_NUMBER"].values
cycle_index = ds["CYCLE_NUMBER_INDEX"].values

pres = ds["PRES_ADJUSTED"].values
if np.isfinite(pres).sum() == 0:
    pres = ds["PRES"].values

park_start = ds["JULD_PARK_START"].values
park_end = ds["JULD_PARK_END"].values

lookup = {}

for cyc, start, end in zip(cycle_index, park_start, park_end):
    mask = (
        (cycles == cyc)
        & np.isfinite(juld)
        & np.isfinite(pres)
        & np.isfinite(start)
        & np.isfinite(end)
        & (juld >= start)
        & (juld <= end)
        & (pres >= 50)
    )
    vals = pres[mask]
    if vals.size:
        lookup[cyc] = float(np.nanpercentile(vals, 95))  # median from your config

print("lookup first 20:")
for cyc in cycle_index[:20]:
    print(cyc, lookup.get(cyc))

filled = {}
source = {}

last = None
for cyc in cycle_index:
    if cyc in lookup:
        filled[cyc] = lookup[cyc]
        source[cyc] = "park_window_pres_adjusted_p50"
        last = lookup[cyc]
    elif last is not None:
        filled[cyc] = last
        source[cyc] = "depth_ffill"

next_value = None
for cyc in reversed(cycle_index):
    if cyc in lookup:
        next_value = lookup[cyc]
    elif cyc not in filled and next_value is not None:
        filled[cyc] = next_value
        source[cyc] = "depth_bfill"

fallback = 1.0
for cyc in cycle_index:
    if cyc not in filled:
        filled[cyc] = fallback
        source[cyc] = "fallback"

print("\nfinal first 20:")
for cyc in cycle_index[:20]:
    print(cyc, filled.get(cyc), source.get(cyc))

print("\nsource counts:")
print(pd.Series([source[c] for c in cycle_index]).value_counts())