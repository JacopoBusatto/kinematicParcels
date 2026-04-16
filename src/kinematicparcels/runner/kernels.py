"""
Modified Parcels kernels for LKM integration.

Provides advection kernels that include inline LKM velocity calculation.
"""

from parcels import JITParticle, ScipyParticle
import numpy as np


def AdvectionRK4_LKM(particle, fieldset, time):
    """
    RK4 advection kernel with inline LKM calculation.

    This kernel computes LKM velocities at each of the 4 RK4 stages
    and adds them to the resolved field velocities.

    Requirements:
    - Particle must have: x_rel_m, y_rel_m (relative coordinates from group center)
    - fieldset must have: lkm_modes attribute (set before execution)
    - Both are set by update_group_centers_and_relative_coords() before pset.execute()

    Parameters
    ----------
    particle : Particle
        Parcels particle with x_rel_m, y_rel_m variables
    fieldset : FieldSet
        Velocity field with UV variables and lkm_modes attribute
    time : float
        Current time (seconds)
    """
    group_size = int(particle.group_size)
    apply_lkm = (group_size >= 2) and hasattr(fieldset, 'lkm_modes') and (fieldset.lkm_modes is not None)

    # ====================================================================
    # RK4 Stage 1: at current position
    # ====================================================================
    (u_res, v_res) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    u_lkm_1 = 0.0
    v_lkm_1 = 0.0
    if apply_lkm:
        for n in range(fieldset.lkm_modes.n_modes):
            k_n = fieldset.lkm_modes.wavenumbers_1m[n]
            A_n = fieldset.lkm_modes.amplitudes_ms[n]
            omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
            eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
            phi_n = fieldset.lkm_modes.phases_rad[n]

            arg_x = k_n * particle.x_rel_m - k_n * eps_n * np.sin(omega_n * time)
            arg_y = k_n * particle.y_rel_m - k_n * eps_n * np.sin(omega_n * time + phi_n)

            u_lkm_1 += A_n * np.sin(arg_x) * np.cos(arg_y)
            v_lkm_1 -= A_n * np.cos(arg_x) * np.sin(arg_y)

        cos_lat = np.cos(particle.lat * np.pi / 180.0)
        if cos_lat < 1.0e-8:
            cos_lat = 1.0e-8
        u_lkm_1 /= 111320.0 * cos_lat
        v_lkm_1 /= 110540.0

    u1 = u_res + u_lkm_1
    v1 = v_res + v_lkm_1
    lon1 = particle.lon + u1 * particle.dt / 2.0
    lat1 = particle.lat + v1 * particle.dt / 2.0

    # ====================================================================
    # RK4 Stage 2: at half-step position
    # ====================================================================
    (u_res, v_res) = fieldset.UV[time + particle.dt / 2.0, particle.depth, lat1, lon1]

    u_lkm_2 = 0.0
    v_lkm_2 = 0.0
    if apply_lkm:
        for n in range(fieldset.lkm_modes.n_modes):
            k_n = fieldset.lkm_modes.wavenumbers_1m[n]
            A_n = fieldset.lkm_modes.amplitudes_ms[n]
            omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
            eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
            phi_n = fieldset.lkm_modes.phases_rad[n]

            time_2 = time + particle.dt / 2.0
            arg_x = k_n * particle.x_rel_m - k_n * eps_n * np.sin(omega_n * time_2)
            arg_y = k_n * particle.y_rel_m - k_n * eps_n * np.sin(omega_n * time_2 + phi_n)

            u_lkm_2 += A_n * np.sin(arg_x) * np.cos(arg_y)
            v_lkm_2 -= A_n * np.cos(arg_x) * np.sin(arg_y)

        cos_lat = np.cos(lat1 * np.pi / 180.0)
        if cos_lat < 1.0e-8:
            cos_lat = 1.0e-8
        u_lkm_2 /= 111320.0 * cos_lat
        v_lkm_2 /= 110540.0

    u2 = u_res + u_lkm_2
    v2 = v_res + v_lkm_2
    lon2 = particle.lon + u2 * particle.dt / 2.0
    lat2 = particle.lat + v2 * particle.dt / 2.0

    # ====================================================================
    # RK4 Stage 3: at other half-step position
    # ====================================================================
    (u_res, v_res) = fieldset.UV[time + particle.dt / 2.0, particle.depth, lat2, lon2]

    u_lkm_3 = 0.0
    v_lkm_3 = 0.0
    if apply_lkm:
        for n in range(fieldset.lkm_modes.n_modes):
            k_n = fieldset.lkm_modes.wavenumbers_1m[n]
            A_n = fieldset.lkm_modes.amplitudes_ms[n]
            omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
            eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
            phi_n = fieldset.lkm_modes.phases_rad[n]

            time_3 = time + particle.dt / 2.0
            arg_x = k_n * particle.x_rel_m - k_n * eps_n * np.sin(omega_n * time_3)
            arg_y = k_n * particle.y_rel_m - k_n * eps_n * np.sin(omega_n * time_3 + phi_n)

            u_lkm_3 += A_n * np.sin(arg_x) * np.cos(arg_y)
            v_lkm_3 -= A_n * np.cos(arg_x) * np.sin(arg_y)

        cos_lat = np.cos(lat2 * np.pi / 180.0)
        if cos_lat < 1.0e-8:
            cos_lat = 1.0e-8
        u_lkm_3 /= 111320.0 * cos_lat
        v_lkm_3 /= 110540.0

    u3 = u_res + u_lkm_3
    v3 = v_res + v_lkm_3
    lon3 = particle.lon + u3 * particle.dt
    lat3 = particle.lat + v3 * particle.dt

    # ====================================================================
    # RK4 Stage 4: at end position
    # ====================================================================
    (u_res, v_res) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]

    u_lkm_4 = 0.0
    v_lkm_4 = 0.0
    if apply_lkm:
        for n in range(fieldset.lkm_modes.n_modes):
            k_n = fieldset.lkm_modes.wavenumbers_1m[n]
            A_n = fieldset.lkm_modes.amplitudes_ms[n]
            omega_n = fieldset.lkm_modes.frequencies_hz[n] * 2.0 * np.pi
            eps_n = fieldset.lkm_modes.osc_amplitudes_m[n]
            phi_n = fieldset.lkm_modes.phases_rad[n]

            time_4 = time + particle.dt
            arg_x = k_n * particle.x_rel_m - k_n * eps_n * np.sin(omega_n * time_4)
            arg_y = k_n * particle.y_rel_m - k_n * eps_n * np.sin(omega_n * time_4 + phi_n)

            u_lkm_4 += A_n * np.sin(arg_x) * np.cos(arg_y)
            v_lkm_4 -= A_n * np.cos(arg_x) * np.sin(arg_y)

        cos_lat = np.cos(lat3 * np.pi / 180.0)
        if cos_lat < 1.0e-8:
            cos_lat = 1.0e-8
        u_lkm_4 /= 111320.0 * cos_lat
        v_lkm_4 /= 110540.0

    u4 = u_res + u_lkm_4
    v4 = v_res + v_lkm_4

    particle.lon += (particle.dt / 6.0) * (u1 + 2 * u2 + 2 * u3 + u4)
    particle.lat += (particle.dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)