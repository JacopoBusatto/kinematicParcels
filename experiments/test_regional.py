from glob import glob
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from parcels import FieldSet, ParticleSet, ScipyParticle, AdvectionRK4
from utilities.geographicalRegions import get_region_by_label, make_regular_grid_in_region
from utilities.geographicalRegions import get_region_by_label, make_regular_grid_in_region
from utilities.init_checks import (
    summarize_initial_points,
    check_initial_points_in_domain,
    filter_inside_domain,
)


files = sorted(glob("./fields/*.nc"))

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

region = get_region_by_label("NPstg")
lons, lats = make_regular_grid_in_region(region, dlon=5.0, dlat=5.0)

summarize_initial_points(lons, lats, name="raw grid")

check_initial_points_in_domain(lons, lats, fieldset)

lons, lats = filter_inside_domain(lons, lats, fieldset)

summarize_initial_points(lons, lats, name="filtered grid")

print(f"Rilascio {len(lons)} particelle in {region.label}")

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=lons,
    lat=lats,
)

output_file = pset.ParticleFile(
    name="output_NPstg.zarr",
    outputdt=timedelta(days=1),
)

pset.execute(
    AdvectionRK4,
    runtime=timedelta(days=10),
    dt=timedelta(hours=1),
    output_file=output_file,
)