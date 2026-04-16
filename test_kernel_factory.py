"""Quick test of the new inline LKM kernel."""
import numpy as np
from src.kinematicparcels.utilities.lkm import build_lkm_modes
from src.kinematicparcels.runner.kernels_lkm_inline import make_AdvectionRK4_with_LKM

# Build test modes
lkm_modes = build_lkm_modes(
    L_min_m=100,
    L_max_m=10000,
    increment_factor=1.414,
    epsilon_tke=0.01,
    c0=1.0,
    phi_spec=("constant", np.pi/4),
)

print(f"✓ LKM modes created: {lkm_modes.n_modes} modes")

# Create the kernel via factory
kernel_func = make_AdvectionRK4_with_LKM(lkm_modes)

print(f"✓ Kernel factory created")
print(f"✓ Kernel function: {kernel_func.__name__}")

# Check it's callable
if callable(kernel_func):
    print("✓ Kernel is callable")
else:
    print("✗ Kernel is not callable!")

print("\nNew implementation ready for testing!")
