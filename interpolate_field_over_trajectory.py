import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# ============================================================
# 1. Interpolazione del campo di velocità sulla traiettoria
# ============================================================
def interpolate_velocity_on_trajectory(
    traj: pd.DataFrame,
    field: xr.Dataset,
    *,
    time_col: str = "time",
    lon_col: str = "lon",
    lat_col: str = "lat",
    u_name: str = "x_sea_water_velocity",
    v_name: str = "y_sea_water_velocity",
    depth_index: int = 0,
) -> pd.DataFrame:
    """
    Interpola il campo di velocità curvilineo (u,v) sui punti della traiettoria.

    Parametri
    ---------
    traj : pd.DataFrame
        DataFrame con almeno colonne time, lon, lat.
    field : xr.Dataset
        Dataset con:
          - time
          - lon_rho(xi_rho, eta_rho)
          - lat_rho(xi_rho, eta_rho)
          - u(time, depth, xi_rho, eta_rho)
          - v(time, depth, xi_rho, eta_rho)

    Restituisce
    -----------
    traj_out : pd.DataFrame
        Copia di traj con colonne aggiunte:
          - u_interp
          - v_interp
          - speed_interp
    """
    traj_out = traj.copy()
    traj_out[time_col] = pd.to_datetime(traj_out[time_col])

    # Coordinate statiche della griglia
    lon2d = field["lon_rho"].values
    lat2d = field["lat_rho"].values

    # Tempi disponibili nel campo
    field_times = pd.to_datetime(field["time"].values)

    u_list = []
    v_list = []

    for _, row in traj_out.iterrows():
        t = pd.to_datetime(row[time_col])
        x = float(row[lon_col])
        y = float(row[lat_col])

        # indice tempo più vicino
        it = np.argmin(np.abs(field_times - t))

        # selezione del campo all'istante it
        u2d = field[u_name].isel(time=it, depth=depth_index).values
        v2d = field[v_name].isel(time=it, depth=depth_index).values

        # maschera dei punti validi
        valid = (
            np.isfinite(lon2d)
            & np.isfinite(lat2d)
            & np.isfinite(u2d)
            & np.isfinite(v2d)
        )

        pts = np.column_stack((lon2d[valid], lat2d[valid]))

        # Interpolazione lineare
        u_lin = LinearNDInterpolator(pts, u2d[valid], fill_value=np.nan)
        v_lin = LinearNDInterpolator(pts, v2d[valid], fill_value=np.nan)

        u_val = float(u_lin(x, y))
        v_val = float(v_lin(x, y))

        # Fallback a nearest se il punto finisce fuori dal triangolo convesso
        if np.isnan(u_val) or np.isnan(v_val):
            u_near = NearestNDInterpolator(pts, u2d[valid])
            v_near = NearestNDInterpolator(pts, v2d[valid])
            u_val = float(u_near(x, y))
            v_val = float(v_near(x, y))

        u_list.append(u_val)
        v_list.append(v_val)

    traj_out["u_interp"] = u_list
    traj_out["v_interp"] = v_list
    traj_out["speed_interp"] = np.hypot(traj_out["u_interp"], traj_out["v_interp"])

    return traj_out


# ============================================================
# 2. Plot mappa traiettoria + vettori velocità interpolati
# ============================================================
def plot_trajectory_with_velocity_vectors(
    traj_interp: pd.DataFrame,
    field: xr.Dataset | None = None,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    u_col: str = "u_interp",
    v_col: str = "v_interp",
    time_col: str = "time",
    title: str = "Trajectory with interpolated velocity vectors",
    stride: int = 1,
    scale: float | None = None,
    figsize: tuple = (10, 10),
    show_background_speed: bool = False,
    bg_time_index: int = 0,
):
    """
    Plotta la traiettoria e i vettori velocità interpolati lungo la traiettoria.

    Parametri
    ---------
    traj_interp : pd.DataFrame
        DataFrame con lon, lat, u_interp, v_interp.
    field : xr.Dataset | None
        Se fornito, può plottare anche uno sfondo della speed del campo.
    show_background_speed : bool
        Se True e field non è None, mostra la speed del campo a bg_time_index.
    stride : int
        Disegna un vettore ogni `stride` punti.
    scale : float | None
        Parametro matplotlib quiver scale. Se None, lascia automatico.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Eventuale sfondo con speed del campo
    if show_background_speed and field is not None:
        lon2d = field["lon_rho"].values
        lat2d = field["lat_rho"].values
        u2d = field["x_sea_water_velocity"].isel(time=bg_time_index, depth=0).values
        v2d = field["y_sea_water_velocity"].isel(time=bg_time_index, depth=0).values
        spd2d = np.hypot(u2d, v2d)

        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            spd2d,
            shading="auto",
            alpha=0.6,
        )
        cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label("Speed [m/s]")

    # traiettoria
    ax.plot(
        traj_interp[lon_col],
        traj_interp[lat_col],
        "-k",
        lw=1.5,
        label="Trajectory",
        zorder=3,
    )

    # start / end
    ax.scatter(
        traj_interp[lon_col].iloc[0],
        traj_interp[lat_col].iloc[0],
        s=70,
        marker="o",
        label="Start",
        zorder=4,
    )
    ax.scatter(
        traj_interp[lon_col].iloc[-1],
        traj_interp[lat_col].iloc[-1],
        s=70,
        marker="s",
        label="End",
        zorder=4,
    )

    # vettori velocità
    sl = slice(None, None, stride)
    q = ax.quiver(
        traj_interp[lon_col].iloc[sl],
        traj_interp[lat_col].iloc[sl],
        traj_interp[u_col].iloc[sl],
        traj_interp[v_col].iloc[sl],
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.0025,
        zorder=5,
    )

    # etichette temporali opzionali
    for i, (_, row) in enumerate(traj_interp.iloc[sl].iterrows()):
        ax.text(
            row[lon_col],
            row[lat_col],
            pd.to_datetime(row[time_col]).strftime("%H:%M"),
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=6,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="ur")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Esempio di utilizzo
# ============================================================
if __name__ == "__main__":
    traj = pd.read_parquet("outputs/postprocessing/one_trajectory_test/trajectory_table.parquet")
    field = xr.open_dataset("C:/Users/Jacopo/Documents/DATI/PATAGONIA/ocean_uv_opendrift_final_V2.nc")

    traj_interp = interpolate_velocity_on_trajectory(traj, field)

    print(traj_interp[["time", "lon", "lat", "u_interp", "v_interp", "speed_interp"]])

    plot_trajectory_with_velocity_vectors(
        traj_interp,
        field=field,
        stride=1,                  # un vettore per ogni punto
        scale=20,                # automatico
        show_background_speed=True,
        bg_time_index=0,
        title="Trajectory + interpolated velocity vectors",
    )