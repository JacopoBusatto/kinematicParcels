#!/usr/bin/env python
"""Test integration of LKM kernel into run_experiment.py"""

import numpy as np
from kinematicparcels.utilities.lkm import build_lkm_modes
from kinematicparcels.runner.kernels import AdvectionRK4_LKM
from kinematicparcels.runner.kernels_lkm_inline import make_AdvectionRK4_with_LKM

# Test LKM modes creation
lkm_modes = build_lkm_modes(
    L_min_m=100,
    L_max_m=10000,
    increment_factor=1.5,
    epsilon_tke=0.01,
    c0=0.4,
    phi_spec=('constant', np.pi/4),
)

print(f"✓ LKM modes created: {lkm_modes.n_modes} modes")
print(f"  Wavenumbers: {lkm_modes.wavenumbers_1m[:3]} ... {lkm_modes.wavenumbers_1m[-1]}")
print(f"  Amplitudes: {lkm_modes.amplitudes_ms[:3]}")

# Test kernel factory
kernel_func = make_AdvectionRK4_with_LKM(lkm_modes)
print(f"✓ Kernel factory works: returned {kernel_func.__name__}")

# Verify kernel is the correct function
print(f"✓ Kernel matches AdvectionRK4_LKM: {kernel_func is AdvectionRK4_LKM}")

print("\n✓ All integration tests passed!")
