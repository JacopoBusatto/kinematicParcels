from glob import glob
from datetime import timedelta

from parcels import FieldSet, ParticleSet, ScipyParticle, AdvectionRK4

# --------------------------------------------------
# 1. File list
# --------------------------------------------------
files = sorted(glob("fields/*.nc"))
print(f"Trovati {len(files)} file")
assert len(files) == 31, f"Mi aspettavo 31 file, trovati {len(files)}"

# --------------------------------------------------
# 2. FieldSet: uso ugos / vgos (2D)
# --------------------------------------------------
filenames = {
    "U": files,
    "V": files,
}

variables = {
    "U": "uo",
    "V": "vo",
}

dimensions = {
    "U": {"lon": "longitude", "lat": "latitude", "time": "time"},
    "V": {"lon": "longitude", "lat": "latitude", "time": "time"},
}

fieldset = FieldSet.from_netcdf(
    filenames=filenames,
    variables=variables,
    dimensions=dimensions,
)

print(fieldset)

# --------------------------------------------------
# 3. ParticleSet
#    punto iniziale semplice, in oceano aperto
# --------------------------------------------------
pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=[-45.0],
    lat=[40.0],
)

print("ParticleSet creato")

# --------------------------------------------------
# 4. Output
# --------------------------------------------------
output_file = pset.ParticleFile(
    name="output_run01.zarr",
    outputdt=timedelta(hours=6),
)

# --------------------------------------------------
# 5. Esecuzione
# --------------------------------------------------
pset.execute(
    AdvectionRK4,
    runtime=timedelta(days=5),
    dt=timedelta(hours=1),
    output_file=output_file,
)

print("Run completato")