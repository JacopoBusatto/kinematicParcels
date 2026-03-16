# Parcels Post-processing Framework

This module provides a scalable and modular framework for analysing Lagrangian particle simulations produced with **OceanParcels**.

The design is intended to support **large trajectory datasets** (10⁵–10⁶ particles) while keeping the codebase readable and extendable for future diagnostics such as:

- particle density maps
- residence time statistics
- connectivity matrices
- exit time analyses
- FSLE diagnostics

---

# Philosophy

The post-processing pipeline follows a clear sequence:

```
Parcels Zarr output
        │
        ▼
Trajectory table (canonical format)
        │
        ▼
Trajectory cleaning
        │
        ▼
Particle summary statistics
        │
        ▼
Grid-based diagnostics
        │
        ▼
Scientific analysis products
```

Each step is implemented as a **separate module**, allowing independent reuse and testing.

---

# Package structure

```
postprocessing/
│
├── config/
│   YAML configuration system
│
├── io/
│   Readers and trajectory table construction
│
├── core/
│   particle-level statistics
│
├── grid/
│   regular grid representation
│
├── analysis/
│   grid-based diagnostics (density, etc.)
│
├── plotting/
│   visualization utilities
│
└── workflows/
    high-level workflows
```

---

# 1. Trajectory Table

The first step converts the Parcels output dataset into a **canonical table format**.

```
trajectory | obs | time | lon | lat | z
```

Each row represents **one particle observation at one timestep**.

Example:

| trajectory | obs | time | lon | lat |
|------------|-----|------|-----|-----|
| 0 | 0 | t0 | -73.6 | -52.4 |
| 0 | 1 | t1 | -73.59 | -52.39 |
| 1 | 0 | t0 | -73.5 | -52.3 |

This format simplifies:

- filtering
- grouping
- grid aggregation
- statistics

Reader function:

```
load_trajectory_table()
```

---

# 2. Trajectory Cleaning

Trajectory datasets may contain artefacts:

- particles released on land
- particles becoming immobile
- invalid coordinates

Cleaning operations are applied before analysis.

## Invalid point truncation

Trajectories are truncated when invalid positions appear.

## Stagnant trajectory detection

Particles can sometimes become **numerically stuck** (for example when hitting land).

This is detected when consecutive steps satisfy:

```
|Δlon| < tol
|Δlat| < tol
```

for at least `N` consecutive timesteps.

When detected:

- the trajectory is truncated before the stagnant segment
- trajectories stagnant from the beginning can be removed

Configuration:

```yaml
cleaning:
  truncate_stagnant: true
  stagnant_tol: 1e-6
  stagnant_min_consecutive: 2
```

---

# 3. Particle Summary

The particle summary reduces each trajectory to a **single row of statistics**.

Example variables:

| variable | meaning |
|--------|--------|
| lon0 | initial longitude |
| lat0 | initial latitude |
| lonf | final longitude |
| latf | final latitude |
| lifetime_seconds | particle lifetime |

Function:

```
build_particle_summary()
```

This dataset is typically used for:

- release diagnostics
- exit statistics
- connectivity analyses

---

# 4. Regular Grid

Many diagnostics require aggregating particles on a spatial grid.

The `RegularGrid` class defines:

```
lon_min
lon_max
lat_min
lat_max
dlon
dlat
```

and internally computes:

- cell centers
- grid indices
- mapping from positions to pixels

Important:

The grid must be defined using **cell edges**, not centers, to avoid numerical artefacts.

---

# 5. Particle Density

Particle density is computed on the grid at each timestep.

Output is an **xarray Dataset**:

```
density(time, lat, lon)
```

Variables:

| variable | meaning |
|--------|--------|
| particle_count | number of particles in cell |
| density_active | normalized by active particles |
| density_total | normalized by total released |

Normalization options allow consistent comparisons between simulations.

Configuration:

```yaml
density:
  normalize_active: true
  normalize_total: true
```

---

# Configuration system

The entire pipeline can be controlled through a YAML configuration file.

Example:

```yaml
dataset:
  input_path: outputs/simulation.zarr

grid:
  lon_min: -73.6625
  lon_max: -72.4875
  lat_min: -52.4875
  lat_max: -51.4375
  dlon: 0.025
  dlat: 0.025

density:
  normalize_active: true
  normalize_total: true

cleaning:
  truncate_stagnant: true
  stagnant_tol: 1e-6
  stagnant_min_consecutive: 2
```

---

# Design goals

The framework is designed to be:

### Scalable

Handle trajectory datasets with:

```
10^5 – 10^6 particles
10^3 – 10^4 timesteps
```

### Modular

Each analysis is implemented as a standalone module.

### Reproducible

All analysis steps are controlled through YAML configuration.

### Extensible

Future modules may include:

- residence time maps
- connectivity matrices
- FSLE diagnostics
- particle encounter statistics

---

# Future work

Planned additions:

- residence time computation
- connectivity matrices
- exit-time statistics
- FSLE implementation
- multi-experiment comparison tools