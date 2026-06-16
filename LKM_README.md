# Lagrangian Kinematic Model (LKM) - Current Implementation

This document describes the current LKM implementation in kinematicParcels.

## Overview

LKM adds a deterministic, time-dependent, multiscale horizontal velocity contribution to the resolved field velocity.

At each advection stage, total velocity is:

- u_total = u_resolved + u_lkm
- v_total = v_resolved + v_lkm

## Current Execution Modes

There are two active execution paths.

1. Member-based path (legacy grouped particles)
- One Parcels particle per member.
- Group centers and relative coordinates are updated in synchronized chunks.
- Uses `update_group_centers_and_relative_coords()` before each execute chunk.
- Kernel: `AdvectionRK4_LKM` via `kernels_lkm_inline.py` wrapper.

2. Grouped-entity path (current default for `group.size > 1`)
- One Parcels particle stores all members of a group (`lon_1..lon_5`, `lat_1..lat_5`).
- Group center and member-relative coordinates are computed inside grouped kernel.
- Current grouped-entity geometry uses arithmetic lon/lat center plus local metric conversion (equirectangular approximation).
- Kernel: `AdvectionRK4_Grouped` from `runner/grouped_kernels.py`.
- Supported group sizes: 2..5.
- Currently scipy-only in grouped-entity mode.

## Core Files

- `src/kinematicparcels/utilities/lkm.py`
- `src/kinematicparcels/runner/kernels.py`
- `src/kinematicparcels/runner/kernels_lkm_inline.py`
- `src/kinematicparcels/runner/grouped_kernels.py`
- `src/kinematicparcels/runner/run_experiment.py`
- `src/kinematicparcels/utilities/group_dynamics.py`
- `src/kinematicparcels/utilities/geometry.py`

## LKM Parameters and Formulas

Modes are created in `build_lkm_modes()` using geometric wavelengths:

- lambda_n = L_min * r^n
- k_n = 2*pi / lambda_n
- A_n = A_0 * (k_n / k_0)^(-1/3)
- epsilon_n = 1e-1 / k_n
- T_n = 2*lambda_n / A_n
- omega_n = 2*pi / T_n

Important: the current code uses epsilon_n = 1e-1/k_n.

## YAML Configuration

```yaml
lkm:
  enabled: true
  mode: group_center_of_mass
  L_min_km: 0.1
  L_max_km: 10.0
  increment_factor: 1.414
  epsilon_tke: 1.0e-2
  c0: 1.0
  phi_mode: constant
  phi_value: 0.78539816339
  update_every_steps: 1
  apply_to_group_size_min: 2
  debug_output: false
```

Notes:
- `update_every_steps` is used by the member-based synchronized path.
- In grouped-entity path, center updates happen inside the grouped kernel each full step.

## Output Layouts

Two raw output layouts may appear depending on run mode.

1. Member-based output
- One Parcels particle is written for each member.
- Core variables are `time`, `lon`, `lat`, `z`.
- Typical grouped variables include `group_id`, `group_member`, `group_size`.
- Depending on release mode and diagnostics, raw output may also contain `circle_id`, `center_lon`, `center_lat`, `x_rel_m`, `y_rel_m`.
- In this layout, `lon` and `lat` are the actual member coordinates.

2. Grouped-entity output
- One Parcels particle is written for the whole group.
- Core variables are `time`, `lon`, `lat`, `z`.
- Typical grouped variables include `group_id`, `group_size`, `center_lon`, `center_lat` and, for circle releases, `circle_id`.
- Group-level canonical `lon`, `lat` are the group-center track.
- In current grouped-entity output, `lon` and `lat` are equal to `center_lon` and `center_lat`.
- Member coordinates are stored in `lon_1..lon_5`, `lat_1..lat_5`.
- `group_member` is not stored directly in raw output.
- Only the numbered member variables from `1` to `group_size` are meaningful.

Postprocessing expands grouped-entity output to member-wise rows so trajectory plots and `max_group_member` work as expected.

## Known Limits

- Grouped-entity mode currently supports only scipy particles.
- Grouped-entity mode currently requires `depth.enabled: false`.
- Grouped-entity kernels are fixed-size (2..5 members).

## Validation Checklist

- LKM disabled reproduces standard advection behavior.
- LKM enabled with grouped runs produces non-trivial member separation.
- Postprocessing member expansion yields expected `group_member` values.

## References

- Lacorata, G., Palatella, L., Santoleri, R. (2014), Journal of Physical Oceanography.
- Implementation notes: `references/lkm_kernel_implementation_notes.tex`
