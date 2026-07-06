from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kinematicparcels.postprocessing.io.parcels import load_trajectory_table

path = r"F:/ARGO/zarr/SO_NO_QC_rtraj_z0900_1100.zarr"
output_dir = Path(r"F:/ARGO/postprocessing/SO_NO_QC_z0900_1100/bad_segments")
plot_dir = output_dir / "suspect_platform_plots"

speed_threshold_m_s = 1.
distance_threshold_km = 1000.0
max_plots = None  # set to an integer, e.g. 50, if too many figures are produced


def split_longitude_wrapped_path(lon, lat, max_lon_step=180.0):
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)

    if lon_arr.size < 2:
        return [(lon_arr, lat_arr)]

    valid = np.isfinite(lon_arr[:-1]) & np.isfinite(lon_arr[1:])
    jump_idx = np.flatnonzero(valid & (np.abs(np.diff(lon_arr)) > max_lon_step))
    if jump_idx.size == 0:
        return [(lon_arr, lat_arr)]

    segments = []
    start = 0
    for idx in jump_idx:
        stop = idx + 1
        segments.append((lon_arr[start:stop], lat_arr[start:stop]))
        start = stop
    segments.append((lon_arr[start:], lat_arr[start:]))
    return segments


def padded_extent(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]

    if lon.size == 0 or lat.size == 0:
        return [-180.0, 180.0, -80.0, -30.0]

    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))

    lon_span = max(lon_max - lon_min, 1.0)
    lat_span = max(lat_max - lat_min, 1.0)
    lon_pad = max(2.0, min(20.0, 0.15 * lon_span))
    lat_pad = max(2.0, min(10.0, 0.20 * lat_span))

    return [
        max(-180.0, lon_min - lon_pad),
        min(180.0, lon_max + lon_pad),
        max(-90.0, lat_min - lat_pad),
        min(90.0, lat_max + lat_pad),
    ]


def safe_name(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def plot_suspect_trajectory(traj, bad_traj, outpath):
    traj = traj.sort_values("obs").copy()
    bad_traj = bad_traj.sort_values("speed_m_s", ascending=False).copy()

    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    land = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        "10m",
        edgecolor="black",
        facecolor=cfeature.COLORS["land"],
        linewidth=0.4,
    )
    ax.add_feature(land, zorder=0)
    ax.coastlines(resolution="10m", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    for lon_part, lat_part in split_longitude_wrapped_path(traj["lon"], traj["lat"]):
        if len(lon_part) < 2:
            continue
        ax.plot(
            lon_part,
            lat_part,
            color="0.25",
            linewidth=1.1,
            alpha=0.75,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    ax.scatter(
        traj["lon"],
        traj["lat"],
        s=10,
        color="0.15",
        alpha=0.45,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.scatter(
        traj["lon"].iloc[0],
        traj["lat"].iloc[0],
        s=55,
        marker="o",
        color="tab:green",
        edgecolor="black",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="start",
    )
    ax.scatter(
        traj["lon"].iloc[-1],
        traj["lat"].iloc[-1],
        s=70,
        marker="X",
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="end",
    )

    for rank, row in enumerate(bad_traj.itertuples(index=False), start=1):
        ax.plot(
            [row.prev_lon, row.lon],
            [row.prev_lat, row.lat],
            color="tab:red",
            linewidth=2.8,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.scatter(
            [row.prev_lon, row.lon],
            [row.prev_lat, row.lat],
            s=42,
            color="tab:red",
            edgecolor="black",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
        mid_lon = float(np.mean([row.prev_lon, row.lon]))
        mid_lat = float(np.mean([row.prev_lat, row.lat]))
        label = f"{rank}: {row.speed_m_s:.2f} m/s, {row.dist_km:.0f} km"
        ax.text(
            mid_lon,
            mid_lat,
            label,
            fontsize=8,
            color="tab:red",
            transform=ccrs.PlateCarree(),
            zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )

    extent = padded_extent(traj["lon"], traj["lat"])
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    platform = bad_traj["platform_code"].iloc[0]
    trajectory = bad_traj["trajectory"].iloc[0]
    title = (
        f"ARGO platform {platform} - trajectory {trajectory} - "
        f"{len(bad_traj)} suspect segment(s)"
    )
    ax.set_title(title)
    ax.legend(loc="lower left")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.close(fig)

df = load_trajectory_table(
    path,
    extra_vars=["platform_code"],
    truncate_at_first_invalid=False,  # useful for diagnostics
)

df = (
    df.dropna(subset=["time", "lon", "lat"])
      .sort_values(["trajectory", "obs"])
      .reset_index(drop=True)
)

df["time"] = pd.to_datetime(df["time"])

g = df.groupby("trajectory", sort=False)

seg = df.copy()
seg["prev_obs"] = g["obs"].shift(1)
seg["prev_time"] = g["time"].shift(1)
seg["prev_lon"] = g["lon"].shift(1)
seg["prev_lat"] = g["lat"].shift(1)

# Wrapped longitude difference: avoids false huge jumps across the dateline.
dlon_deg = ((seg["lon"] - seg["prev_lon"] + 180.0) % 360.0) - 180.0
dlat_deg = seg["lat"] - seg["prev_lat"]

R = 6371.0088  # km
lat1 = np.deg2rad(seg["prev_lat"].to_numpy(float))
lat2 = np.deg2rad(seg["lat"].to_numpy(float))
dlon = np.deg2rad(dlon_deg.to_numpy(float))
dlat = np.deg2rad(dlat_deg.to_numpy(float))

a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
seg["dist_km"] = 2.0 * R * np.arcsin(np.sqrt(a))

seg["dt_s"] = (seg["time"] - seg["prev_time"]).dt.total_seconds()
seg["dt_days"] = seg["dt_s"] / 86400.0
seg["speed_m_s"] = seg["dist_km"] * 1000.0 / seg["dt_s"]

# Useful extra flag: raw plotting jump across dateline.
seg["raw_dlon_deg"] = seg["lon"] - seg["prev_lon"]
seg["wrapped_dlon_deg"] = dlon_deg

bad = seg[
    (seg["dt_s"] > 0)
    & (
        (seg["speed_m_s"] > speed_threshold_m_s)
        | (seg["dist_km"] > distance_threshold_km)
    )
].copy()

cols = [
    "platform_code", "trajectory",
    "prev_obs", "obs",
    "prev_time", "time", "dt_days",
    "prev_lon", "prev_lat", "lon", "lat",
    "raw_dlon_deg", "wrapped_dlon_deg",
    "dist_km", "speed_m_s",
]

bad = bad[cols].sort_values("speed_m_s", ascending=False)

print(bad.head(50).to_string(index=False))
output_dir.mkdir(parents=True, exist_ok=True)
bad_csv = output_dir / "suspect_fast_segments.csv"
bad.to_csv(bad_csv, index=False)

print("n suspect segments:", len(bad))
print("n suspect platforms:", bad["platform_code"].nunique())
print(bad["platform_code"].drop_duplicates().head(50).to_list())
print(bad)
print("saved suspect segment table:", bad_csv)

if bad.empty:
    print("No suspect segments found. No trajectory plots written.")
else:
    plot_dir.mkdir(parents=True, exist_ok=True)
    bad_trajectories = (
        bad[["trajectory", "platform_code"]]
        .drop_duplicates()
        .sort_values(["platform_code", "trajectory"])
        .reset_index(drop=True)
    )
    if max_plots is not None:
        bad_trajectories = bad_trajectories.head(max_plots)

    for row in bad_trajectories.itertuples(index=False):
        traj = df[df["trajectory"].eq(row.trajectory)].copy()
        bad_traj = bad[bad["trajectory"].eq(row.trajectory)].copy()
        outpath = (
            plot_dir
            / f"platform_{safe_name(row.platform_code)}_trajectory_{safe_name(row.trajectory)}.png"
        )
        plot_suspect_trajectory(traj, bad_traj, outpath)

    print("saved trajectory plots:", plot_dir)
    print("n trajectory plots:", len(bad_trajectories))
