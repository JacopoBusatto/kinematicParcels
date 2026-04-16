"""
Lagrangian Kinematic Model (LKM) implementation.

This module provides the core functionality for building LKM mode parameters
and evaluating LKM velocities at given relative coordinates and time.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LKMModes:
    """Container for all LKM mode parameters."""
    n_modes: int
    wavelengths_m: np.ndarray  # λ_n in meters
    wavenumbers_1m: np.ndarray  # k_n = 2π/λ_n in 1/m
    amplitudes_ms: np.ndarray  # A_n in m/s
    osc_amplitudes_m: np.ndarray  # ε_n = 1e-1/k_n in meters
    frequencies_hz: np.ndarray  # f_n = 1/T_n in Hz
    phases_rad: np.ndarray  # φ_n in radians


def build_lkm_modes(
    L_min_m: float,
    L_max_m: float,
    increment_factor: float,
    epsilon_tke: float,
    c0: float,
    phi_spec: Tuple[str, float | None] = ("constant", np.pi/4),
) -> LKMModes:
    """
    Build LKM mode parameters following the paper's calibration.

    Parameters
    ----------
    L_min_m : float
        Minimum wavelength in meters
    L_max_m : float
        Maximum wavelength in meters
    increment_factor : float
        Geometric spacing factor r (λ_n = L_min * r^n)
    epsilon_tke : float
        Turbulent dissipation rate proxy in m²/s³
    c0 : float
        Kolmogorov constant (order 1)
    phi_spec : tuple
        Phase specification: ("constant", value) or ("random", seed) or ("paper", None)

    Returns
    -------
    LKMModes
        Complete mode parameter set
    """
    # Build wavelength ladder: λ_n = L_min * r^n
    wavelengths = []
    n = 0
    while True:
        l_n = L_min_m * (increment_factor ** n)
        if l_n > L_max_m:
            break
        wavelengths.append(l_n)
        n += 1

    wavelengths = np.array(wavelengths)
    n_modes = len(wavelengths)

    if n_modes == 0:
        raise ValueError(f"No modes generated: L_min={L_min_m}, L_max={L_max_m}, r={increment_factor}")

    # Wavenumbers: k_n = 2π/λ_n
    wavenumbers = 2 * np.pi / wavelengths

    # Reference wavenumber for scaling
    k_0 = wavenumbers[-1]  # largest k (smallest λ)

    # Amplitudes: A_0 = c0 * (ε/k_0)^(1/3), A_n = A_0 * (k_n/k_0)^(-1/3)
    A_0 = c0 * (epsilon_tke / k_0)**(1/3)
    amplitudes = A_0 * (wavenumbers / k_0)**(-1/3)

    # Oscillation amplitudes: ε_n = 1e-1/k_n
    osc_amplitudes = 1.0e-1 / wavenumbers

    # Frequencies: T_n = 2*λ_n / A_n, f_n = 1/T_n (Hz)
    periods = 2 * wavelengths / amplitudes
    frequencies = 1.0 / periods

    # Phases
    if phi_spec[0] == "constant":
        phases = np.full(n_modes, phi_spec[1] if phi_spec[1] is not None else np.pi/4)
    elif phi_spec[0] == "random":
        seed = phi_spec[1] if phi_spec[1] is not None else 42
        np.random.seed(seed)
        phases = np.random.uniform(0, 2*np.pi, n_modes)
    elif phi_spec[0] == "paper":
        phases = np.full(n_modes, np.pi/4)  # Paper uses π/4
    else:
        raise ValueError(f"Unknown phi_spec: {phi_spec}")

    return LKMModes(
        n_modes=n_modes,
        wavelengths_m=wavelengths,
        wavenumbers_1m=wavenumbers,
        amplitudes_ms=amplitudes,
        osc_amplitudes_m=osc_amplitudes,
        frequencies_hz=frequencies,
        phases_rad=phases,
    )


def evaluate_lkm_velocity(
    x_rel_m: float,
    y_rel_m: float,
    t_sec: float,
    lkm_modes: LKMModes,
) -> Tuple[float, float]:
    """
    Evaluate LKM velocity at given relative coordinates and time.

    Implements the paper's formulas:
    u_LKM = Σ A_n sin[k_n*x - k_n*ε_n*sin(ω_n*t)] * cos[k_n*y - k_n*ε_n*sin(ω_n*t + φ_n)]
    v_LKM = -Σ A_n cos[k_n*x - k_n*ε_n*sin(ω_n*t)] * sin[k_n*y - k_n*ε_n*sin(ω_n*t + φ_n)]

    Parameters
    ----------
    x_rel_m, y_rel_m : float
        Relative coordinates in meters (from group center)
    t_sec : float
        Time in seconds
    lkm_modes : LKMModes
        Precomputed mode parameters

    Returns
    -------
    u_lkm, v_lkm : float
        LKM velocity components in m/s
    """
    u_lkm = 0.0
    v_lkm = 0.0

    for n in range(lkm_modes.n_modes):
        k_n = lkm_modes.wavenumbers_1m[n]
        A_n = lkm_modes.amplitudes_ms[n]
        eps_n = lkm_modes.osc_amplitudes_m[n]
        omega_n = 2.0 * np.pi * lkm_modes.frequencies_hz[n]
        phi_n = lkm_modes.phases_rad[n]

        # Time-dependent modulation
        sin_mod_x = np.sin(omega_n * t_sec)
        sin_mod_y = np.sin(omega_n * t_sec + phi_n)

        # Arguments of trigonometric functions
        arg_x = k_n * x_rel_m - k_n * eps_n * sin_mod_x
        arg_y = k_n * y_rel_m - k_n * eps_n * sin_mod_y

        # Add contributions to u and v
        u_lkm += A_n * np.sin(arg_x) * np.cos(arg_y)
        v_lkm -= A_n * np.cos(arg_x) * np.sin(arg_y)

    return u_lkm, v_lkm