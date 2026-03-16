## Parcels Post-processing Framework

This module provides a modular and scalable framework for analysing Lagrangian particle simulations produced with OceanParcels.

The system is designed to work efficiently with large trajectory datasets (10⁵–10⁶ particles) while keeping the codebase:

- modular
- reproducible
- extendable

Typical diagnostics produced include:

- particle density maps
- beaching time maps
- start/end region classification
- trajectory visualisation

Future extensions may include:

- connectivity matrices
- residence time statistics
- particle encounter metrics
- FSLE diagnostics


------------------------------------------------------------
PIPELINE OVERVIEW
------------------------------------------------------------

The post-processing workflow follows a clear sequence:

Parcels Zarr output
        │
        ▼
Trajectory table
        │
        ▼
Trajectory cleaning
        │
        ▼
Particle summary
        │
        ▼
Grid-based diagnostics
        │
        ▼
Scientific products


Each step is implemented in a separate module, allowing independent testing and reuse.


------------------------------------------------------------
RUNNING THE POST-PROCESSING
------------------------------------------------------------

The framework is executed through the CLI:

run-parcels-postprocessing postprocess.yml

Alternatively:

python -m kinematicparcels.postprocessing.runner.run_postprocessing postprocess.yml

The YAML configuration determines:

- which analyses are executed
- cleaning behaviour
- grid configuration
- plotting options
- output formats


------------------------------------------------------------
PACKAGE ARCHITECTURE
------------------------------------------------------------

postprocessing/

config/
    YAML configuration system

io/
    Readers and trajectory table construction

core/
    Particle-level statistics

grid/
    Regular grid representation

analyses/
    Scientific diagnostics (density, beaching, etc.)

plotting/
    Visualisation utilities

runner/
    CLI and analysis dispatcher

workflows/
    High-level analysis workflows


------------------------------------------------------------
DATA MODEL
------------------------------------------------------------

TRAJECTORY TABLE

The Parcels Zarr output is converted into a canonical trajectory table.

trajectory | obs | time | lon | lat

Each row represents one particle observation at one timestep.

Example:

trajectory  obs  time   lon    lat
0           0    t0     -73.6  -52.4
0           1    t1     -73.59 -52.39
1           0    t0     -73.5  -52.3

This format simplifies:

- filtering
- grouping
- grid aggregation
- statistics

Reader function:

load_trajectory_table()


------------------------------------------------------------
TRAJECTORY CLEANING
------------------------------------------------------------

Trajectory datasets may contain artefacts:

- particles released on land
- particles becoming numerically stuck
- invalid coordinates

Cleaning operations are applied before analysis.

STAGNANT TRAJECTORY DETECTION

Particles sometimes become numerically stuck when encountering land.

This is detected when consecutive steps satisfy:

|Δlon| < tol
|Δlat| < tol

for at least N consecutive timesteps.

When detected:

- the trajectory is truncated before the stagnant segment
- trajectories stagnant from the beginning can be removed

Configuration example:

cleaning:
  truncate_stagnant: true
  stagnant_tol: 1e-6
  stagnant_min_consecutive: 2


------------------------------------------------------------
PARTICLE SUMMARY
------------------------------------------------------------

The particle summary reduces each trajectory to a single row of statistics.

Example variables:

lon0
lat0
lonf
latf
lifetime_seconds

Meaning:

lon0 lat0 → initial particle position
lonf latf → final particle position
lifetime_seconds → particle lifetime

Function:

build_particle_summary()

This dataset is typically used for:

- release diagnostics
- exit statistics
- connectivity analyses


------------------------------------------------------------
REGULAR GRID SYSTEM
------------------------------------------------------------

Many diagnostics require aggregating particles on a spatial grid.

The RegularGrid class defines:

lon_min
lon_max
lat_min
lat_max
dlon
dlat

and internally computes:

- cell centers
- grid indices
- mapping from positions to pixels
- aggregation statistics

The grid must be defined using cell edges rather than centers to avoid floating point artefacts.

GRID MODES

1) from_initial_centers

The grid is automatically constructed from the release positions.

grid.mode: from_initial_centers

The grid spacing is defined by:

dlon
dlat

2) explicit_edges

The grid limits are explicitly defined in the configuration file.

grid.mode: explicit_edges


------------------------------------------------------------
IMPLEMENTED ANALYSES
------------------------------------------------------------

Analyses are selected in the YAML configuration file.

Example:

analysis:
  types:
    - density
    - beaching_times
    - start_end_regions
    - trajectories


------------------------------------------------------------
PARTICLE DENSITY
------------------------------------------------------------

Particle density is computed on the grid at each timestep.

Output dataset structure:

density(time, lat, lon)

Variables:

particle_count
density_active
density_total

Meaning:

particle_count   → number of particles in the grid cell
density_active   → normalized by active particles
density_total    → normalized by total released particles

Configuration example:

density:
  normalize_active: true
  normalize_total: true


------------------------------------------------------------
BEACHING TIMES
------------------------------------------------------------

Beaching time represents the lifetime of particles at their release location.

For each release pixel:

beaching_time = minimum particle lifetime reaching that pixel

Output dataset:

beaching_time(lat, lon)

Interpretation:

small values  → particles exit quickly
large values  → particles remain longer

This diagnostic helps identify:

- retention zones
- fast escape pathways


------------------------------------------------------------
START / END REGIONS
------------------------------------------------------------

Particles can be classified into geographical regions.

Region definitions come from:

kinematicparcels.utilities.geographicalRegions

For each particle we compute:

start_region
end_region
start_numericLabel
end_numericLabel

Maps are produced on the release grid.

Configuration example:

start_end_regions:
  region_labels: null
  how_many: priority_max
  priority_level: null
  priority_mode: exact
  input_lon_mode: "-180_180"
  plot: true

Priority rules resolve overlapping regions.


------------------------------------------------------------
TRAJECTORY MAPS
------------------------------------------------------------

Trajectory maps visualise particle tracks on geographic maps.

Features include:

- coastlines
- projection selection
- stagnation filtering

Configuration example:

trajectories:
  plot: true

Projection selection:

plotting:
  projection: PlateCarree

Other useful projections include:

SouthPolarStereo
NorthPolarStereo


------------------------------------------------------------
PLOTTING SYSTEM
------------------------------------------------------------

The plotting module provides two types of maps.

Continuous maps:

Used for:
- density
- beaching times

Discrete maps:

Used for:
- region classification

Cartopy is used for geospatial rendering.


------------------------------------------------------------
WORKFLOW EXECUTION
------------------------------------------------------------

The execution driver is:

run_postprocessing()

The runner performs:

1) load configuration
2) initialise shared context
3) run requested analyses sequentially

Pseudo-logic:

for analysis in cfg.analysis.types:
    run_analysis(...)

Shared data such as:

trajectory_table
particle_summary
grid

are reused between analyses.


------------------------------------------------------------
EXAMPLE CONFIGURATION FILE
------------------------------------------------------------

Example postprocess.yml:

dataset:
  input_path: outputs/simulation.zarr

analysis:
  types:
    - density
    - beaching_times
    - start_end_regions

cleaning:
  truncate_stagnant: true
  stagnant_tol: 1e-6
  stagnant_min_consecutive: 2

grid:
  mode: from_initial_centers
  dlon: 0.025
  dlat: 0.025

density:
  normalize_active: true
  normalize_total: true

start_end_regions:
  plot: true

plotting:
  projection: PlateCarree


------------------------------------------------------------
DESIGN GOALS
------------------------------------------------------------

SCALABILITY

Designed to handle trajectory datasets with:

10^5 – 10^6 particles
10^3 – 10^4 timesteps


MODULARITY

Each analysis is implemented as an independent module.


REPRODUCIBILITY

All analysis parameters are defined in YAML configuration files.


EXTENSIBILITY

New analyses can be added by:

1) implementing an analysis module
2) adding a workflow
3) registering it in the runner


------------------------------------------------------------
FUTURE EXTENSIONS
------------------------------------------------------------

Planned diagnostics include:

- residence time maps
- connectivity matrices
- particle encounter statistics
- FSLE diagnostics
- multi-experiment comparison tools


------------------------------------------------------------
SUMMARY
------------------------------------------------------------

The Parcels post-processing framework provides a robust and extensible environment for analysing large Lagrangian simulations.

Key characteristics:

- modular architecture
- reproducible workflows
- scalable grid aggregation
- flexible region-based diagnostics
- CLI-based execution

The system is designed to evolve into a comprehensive toolbox for Lagrangian ocean analysis.