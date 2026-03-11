# experiment_step1.py
from glob import glob
from parcels import FieldSet

files = sorted(glob("fields/*.nc"))
print(f"Trovati {len(files)} file")
for f in files[:3]:
    print(" ", f)

filenames = {
    "U": files,
    "V": files,
}

variables = {
    "U": "uo",   # <-- da adattare
    "V": "vo",   # <-- da adattare
}

dimensions = {
    "U": {"lon": "longitude", "lat": "latitude", "time": "time"},
    "V": {"lon": "longitude", "lat": "latitude", "time": "time"},
}

fieldset = FieldSet.from_netcdf(
    filenames,
    variables,
    dimensions,
)

print(fieldset)