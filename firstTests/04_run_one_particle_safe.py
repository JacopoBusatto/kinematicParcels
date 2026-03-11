from glob import glob
from datetime import timedelta

from parcels import FieldSet, ParticleSet, ScipyParticle, AdvectionRK4, StatusCode


def DeleteParticle(particle, fieldset, time):
    print(f"Deleting particle at lon={particle.lon}, lat={particle.lat}, time={time}")
    particle.delete()


files = sorted(glob("./fields/*.nc"))
print(f"Trovati {len(files)} file")

filenames = {"U": files, "V": files}

variables = {"U": "ugos", "V": "vgos"}

dimensions = {
    "U": {"lon": "longitude", "lat": "latitude", "time": "time"},
    "V": {"lon": "longitude", "lat": "latitude", "time": "time"},
}

fieldset = FieldSet.from_netcdf(
    filenames=filenames,
    variables=variables,
    dimensions=dimensions,
)

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=[10.0],
    lat=[40.0],
)

output_file = pset.ParticleFile(
    name="output_run01_safe.zarr",
    outputdt=timedelta(hours=6),
)

pset.execute(
    AdvectionRK4,
    runtime=timedelta(days=5),
    dt=timedelta(hours=1),
    output_file=output_file,
    error_recovery={StatusCode.ErrorOutOfBounds: DeleteParticle},
)

print("Run completato")