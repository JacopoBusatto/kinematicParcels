"""Grouped-entity RK4 kernel for fixed-size groups (2..4)."""

from __future__ import annotations

import numpy as np


def velocity_ms_to_deg_per_s(u_ms: float, v_ms: float, lat_deg: float) -> tuple[float, float]:
    """Convert east/north velocities from m/s to geographic degrees per second."""
    cos_lat = np.cos(lat_deg * np.pi / 180.0)
    cos_lat = max(abs(cos_lat), 1.0e-8)
    meters_per_deg_lon = 111320.0 * cos_lat
    meters_per_deg_lat = 110540.0
    return u_ms / meters_per_deg_lon, v_ms / meters_per_deg_lat

def AdvectionRK4_Grouped(particle, fieldset, time):
    """RK4 advection where one particle stores all members of a group."""
    import numpy as np

    group_size_local = int(particle.group_size)

    lons = [particle.lon_1, particle.lon_2, particle.lon_3, particle.lon_4]
    lats = [particle.lat_1, particle.lat_2, particle.lat_3, particle.lat_4]

    center_lon = 0.0
    center_lat = 0.0
    for i in range(group_size_local):
        center_lon += lons[i]
        center_lat += lats[i]
    center_lon /= group_size_local
    center_lat /= group_size_local

    cos_lat = np.cos(center_lat * np.pi / 180.0)
    if cos_lat < 1.0e-8:
        cos_lat = 1.0e-8
    meters_per_deg_lon = 111320.0 * cos_lat
    meters_per_deg_lat = 110540.0

    x_rel = [0.0, 0.0, 0.0, 0.0]
    y_rel = [0.0, 0.0, 0.0, 0.0]
    for i in range(group_size_local):
        x_rel[i] = (lons[i] - center_lon) * meters_per_deg_lon
        y_rel[i] = (lats[i] - center_lat) * meters_per_deg_lat

    particle.center_lon = center_lon
    particle.center_lat = center_lat
    particle.lon = center_lon
    particle.lat = center_lat

    apply_lkm = hasattr(fieldset, "lkm_modes") and (fieldset.lkm_modes is not None)

    new_lons = [lons[0], lons[1], lons[2], lons[3]]
    new_lats = [lats[0], lats[1], lats[2], lats[3]]

    for i in range(group_size_local):
        lon0 = lons[i]
        lat0 = lats[i]
        xr = x_rel[i]
        yr = y_rel[i]

        u_res, v_res = fieldset.UV[time, particle.depth, lat0, lon0]
        if apply_lkm:
            u_lkm = 0.0
            v_lkm = 0.0
            for n in range(fieldset.lkm_modes.n_modes):
                k_n = fieldset.lkm_modes.wavenumbers_1m[n]
                A_n = fieldset.lkm_modes.amplitudes_ms[n]
                omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
                eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
                phi_n = fieldset.lkm_modes.phases_rad[n]

                arg_x = k_n * xr - k_n * eps_n * np.sin(omega_n * time)
                arg_y = k_n * yr - k_n * eps_n * np.sin(omega_n * time + phi_n)

                u_lkm += A_n * np.sin(arg_x) * np.cos(arg_y)
                v_lkm -= A_n * np.cos(arg_x) * np.sin(arg_y)

            cos_lat_local = np.cos(lat0 * np.pi / 180.0)
            if cos_lat_local < 1.0e-8:
                cos_lat_local = 1.0e-8
            u1 = u_res + u_lkm / (111320.0 * cos_lat_local)
            v1 = v_res + v_lkm / 110540.0
        else:
            u1 = u_res
            v1 = v_res

        lon1_mid = lon0 + u1 * particle.dt / 2.0
        lat1_mid = lat0 + v1 * particle.dt / 2.0

        t2 = time + particle.dt / 2.0
        u_res, v_res = fieldset.UV[t2, particle.depth, lat1_mid, lon1_mid]
        if apply_lkm:
            u_lkm = 0.0
            v_lkm = 0.0
            for n in range(fieldset.lkm_modes.n_modes):
                k_n = fieldset.lkm_modes.wavenumbers_1m[n]
                A_n = fieldset.lkm_modes.amplitudes_ms[n]
                omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
                eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
                phi_n = fieldset.lkm_modes.phases_rad[n]

                arg_x = k_n * xr - k_n * eps_n * np.sin(omega_n * t2)
                arg_y = k_n * yr - k_n * eps_n * np.sin(omega_n * t2 + phi_n)

                u_lkm += A_n * np.sin(arg_x) * np.cos(arg_y)
                v_lkm -= A_n * np.cos(arg_x) * np.sin(arg_y)

            cos_lat_local = np.cos(lat1_mid * np.pi / 180.0)
            if cos_lat_local < 1.0e-8:
                cos_lat_local = 1.0e-8
            u2 = u_res + u_lkm / (111320.0 * cos_lat_local)
            v2 = v_res + v_lkm / 110540.0
        else:
            u2 = u_res
            v2 = v_res

        lon2_mid = lon0 + u2 * particle.dt / 2.0
        lat2_mid = lat0 + v2 * particle.dt / 2.0

        u_res, v_res = fieldset.UV[t2, particle.depth, lat2_mid, lon2_mid]
        if apply_lkm:
            u_lkm = 0.0
            v_lkm = 0.0
            for n in range(fieldset.lkm_modes.n_modes):
                k_n = fieldset.lkm_modes.wavenumbers_1m[n]
                A_n = fieldset.lkm_modes.amplitudes_ms[n]
                omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
                eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
                phi_n = fieldset.lkm_modes.phases_rad[n]

                arg_x = k_n * xr - k_n * eps_n * np.sin(omega_n * t2)
                arg_y = k_n * yr - k_n * eps_n * np.sin(omega_n * t2 + phi_n)

                u_lkm += A_n * np.sin(arg_x) * np.cos(arg_y)
                v_lkm -= A_n * np.cos(arg_x) * np.sin(arg_y)

            cos_lat_local = np.cos(lat2_mid * np.pi / 180.0)
            if cos_lat_local < 1.0e-8:
                cos_lat_local = 1.0e-8
            u3 = u_res + u_lkm / (111320.0 * cos_lat_local)
            v3 = v_res + v_lkm / 110540.0
        else:
            u3 = u_res
            v3 = v_res

        lon3_end = lon0 + u3 * particle.dt
        lat3_end = lat0 + v3 * particle.dt

        t4 = time + particle.dt
        u_res, v_res = fieldset.UV[t4, particle.depth, lat3_end, lon3_end]
        if apply_lkm:
            u_lkm = 0.0
            v_lkm = 0.0
            for n in range(fieldset.lkm_modes.n_modes):
                k_n = fieldset.lkm_modes.wavenumbers_1m[n]
                A_n = fieldset.lkm_modes.amplitudes_ms[n]
                omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
                eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
                phi_n = fieldset.lkm_modes.phases_rad[n]

                arg_x = k_n * xr - k_n * eps_n * np.sin(omega_n * t4)
                arg_y = k_n * yr - k_n * eps_n * np.sin(omega_n * t4 + phi_n)

                u_lkm += A_n * np.sin(arg_x) * np.cos(arg_y)
                v_lkm -= A_n * np.cos(arg_x) * np.sin(arg_y)

            cos_lat_local = np.cos(lat3_end * np.pi / 180.0)
            if cos_lat_local < 1.0e-8:
                cos_lat_local = 1.0e-8
            u4 = u_res + u_lkm / (111320.0 * cos_lat_local)
            v4 = v_res + v_lkm / 110540.0
        else:
            u4 = u_res
            v4 = v_res

        new_lons[i] = lon0 + (particle.dt / 6.0) * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        new_lats[i] = lat0 + (particle.dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)

    particle.lon_1 = new_lons[0]
    particle.lat_1 = new_lats[0]
    if group_size_local > 1:
        particle.lon_2 = new_lons[1]
        particle.lat_2 = new_lats[1]
    if group_size_local > 2:
        particle.lon_3 = new_lons[2]
        particle.lat_3 = new_lats[2]
    if group_size_local > 3:
        particle.lon_4 = new_lons[3]
        particle.lat_4 = new_lats[3]

    end_center_lon = 0.0
    end_center_lat = 0.0
    for i in range(group_size_local):
        end_center_lon += new_lons[i]
        end_center_lat += new_lats[i]
    end_center_lon /= group_size_local
    end_center_lat /= group_size_local

    particle.center_lon = end_center_lon
    particle.center_lat = end_center_lat
    particle.lon = end_center_lon
    particle.lat = end_center_lat


def BoundaryHaloKill_GroupedEntity(particle, fieldset, time):
    """Delete a grouped-entity particle when any member enters the boundary halo.

    Must run BEFORE AdvectionRK4_Grouped so no out-of-bound field sampling occurs.
    All members share one Parcels particle, so particle.delete() kills the whole
    group atomically.

    Reads the same fieldset constants as BoundaryHaloKill:
        bh_lat_min, bh_lat_max, bh_lon_min, bh_lon_max, bh_periodic
    """
    group_size_local = int(particle.group_size)

    lons = [particle.lon_1, particle.lon_2, particle.lon_3, particle.lon_4]
    lats = [particle.lat_1, particle.lat_2, particle.lat_3, particle.lat_4]

    check_lon = fieldset.bh_periodic < 0.5

    for i in range(group_size_local):
        lat_i = lats[i]
        lon_i = lons[i]
        if lat_i < fieldset.bh_lat_min or lat_i > fieldset.bh_lat_max:
            particle.delete()
            return
        if check_lon:
            if lon_i < fieldset.bh_lon_min or lon_i > fieldset.bh_lon_max:
                particle.delete()
                return


def WrapLongitudePeriodic_GroupedEntity(particle, fieldset, time):
    """Wrap grouped-member longitudes back into the original zonal domain when periodic."""
    if fieldset.bh_periodic < 0.5:
        return

    west = fieldset.periodic_lon_west
    span = fieldset.periodic_lon_span
    if span <= 0.0:
        return

    while particle.lon < west:
        particle.lon += span
    while particle.lon >= west + span:
        particle.lon -= span

    for i in range(int(particle.group_size)):
        lon_name = f"lon_{i + 1}"
        lon_i = getattr(particle, lon_name)
        while lon_i < west:
            lon_i += span
        while lon_i >= west + span:
            lon_i -= span
        setattr(particle, lon_name, lon_i)


def make_grouped_rk4_lkm_kernel(group_size: int):
    """Return grouped kernel for group size 2..4."""
    if group_size < 2 or group_size > 4:
        raise ValueError(f"Unsupported group_size={group_size}. Supported: 2..4")
    return AdvectionRK4_Grouped
