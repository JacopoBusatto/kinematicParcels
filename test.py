from kinematicparcels.postprocessing.io import load_trajectory_table
from kinematicparcels.postprocessing.core import RegularGrid
from kinematicparcels.postprocessing.analyses import compute_time_density

path = r"C:\Users\Jacopo\Documents\GitHub\kinematicParcels\outputs\output_PNnf_surface.zarr"

df = load_trajectory_table(path)
t0 = df["time"].min()
df0 = df.loc[df["time"] == t0].copy()

dlon = 0.025
dlat = 0.025

lon0 = df0["lon"]
lat0 = df0["lat"]

grid = RegularGrid(
    lon_min=lon0.min() - 0.5 * dlon,
    lon_max=lon0.max() + 0.5 * dlon + 1e-12,
    lat_min=lat0.min() - 0.5 * dlat,
    lat_max=lat0.max() + 0.5 * dlat + 1e-12,
    dlon=dlon,
    dlat=dlat,
)

density_table, density_ds = compute_time_density(
    df,
    grid=grid,
    lon_col="lon",
    lat_col="lat",
    time_col="time",
    normalize_active=True,
    normalize_total=True,
)

print(density_table.head())
print(density_ds)
print(density_ds["particle_count"])
import matplotlib.pyplot as plt
density_ds["particle_count"].isel(time=-1).plot()
plt.show()



from kinematicparcels.postprocessing.io import load_trajectory_table
from kinematicparcels.postprocessing.plotting import plot_trajectories_map

path = r"C:\Users\Jacopo\Documents\GitHub\kinematicParcels\outputs\output_PNnf_surface.zarr"

df = load_trajectory_table(path)

plot_trajectories_map(
    df,
    outpath=r"C:\Users\Jacopo\Documents\GitHub\kinematicParcels\outputs\test_plot.png",
    title="PNnf surface trajectories",
)