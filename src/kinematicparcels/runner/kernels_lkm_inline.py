"""
Custom RK4 kernel with inline LKM calculation.

This kernel computes LKM velocities on-the-fly during RK4 integration,
avoiding pre-computation issues and keeping group dynamics in sync.

The AdvectionRK4_LKM kernel is defined in kernels.py and accessed via fieldset.lkm_modes.
"""

from kinematicparcels.utilities.lkm import LKMModes
from kinematicparcels.runner.kernels import AdvectionRK4_LKM


def make_AdvectionRK4_with_LKM(lkm_modes: LKMModes):
    """
    Factory function that creates a custom RK4 kernel with inline LKM.

    This is a wrapper that returns the AdvectionRK4_LKM kernel from kernels.py.
    The lkm_modes object must be attached to the fieldset before pset.execute():
        fieldset.lkm_modes = lkm_modes

    Parameters
    ----------
    lkm_modes : LKMModes
        Precomputed LKM mode parameters (will be attached to fieldset)

    Returns
    -------
    callable
        AdvectionRK4_LKM kernel function
    """
    # Return the kernel directly - lkm_modes are accessed via fieldset.lkm_modes
    return AdvectionRK4_LKM
