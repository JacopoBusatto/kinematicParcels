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
```
trajectory | obs | time | lon | lat
```
Each row represents one particle observation at one timestep.

Example:

```
trajectory  obs  time   lon    lat
0           0    t0     -73.6  -52.4
0           1    t1     -73.59 -52.39
1           0    t0     -73.5  -52.3
```

This format simplifies:

- filtering
- grouping
- grid aggregation
- statistics

Reader function:

```python
load_trajectory_table()
```

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

```yaml
cleaning:
  truncate_stagnant: true
  stagnant_tol: 1e-6
  stagnant_min_consecutive: 2
```

------------------------------------------------------------
GROUPED TRAJECTORIES
------------------------------------------------------------

When simulation release uses grouped-particle mode (see README.md), the output contains metadata describing group membership:

- `group_id`: Shared index across all members of a group
- `group_member`: Member index within the group (1, 2, 3, ...)
- `group_size`: Total particles per group
- `circle_id`: Source circle index when the release mode is `circle`

The postprocessing framework provides visualization and filtering support for grouped trajectories.

## Trajectory Visualization with Groups

The trajectory plotting module automatically detects group membership in the trajectory data and provides:

**Group filtering**: Display only specific group members
- Controlled by `max_group_member` configuration parameter
- `null` = show all members with distinct colors
- `1` = show only center particles (member 1 of each group)
- `2` = show center + one ring member, etc.

**Member-based coloring**: Each group member gets a distinct color across all groups
- Enables visual tracking of dispersion patterns
- Useful for pair-dispersion studies

Configuration example:

```yaml
trajectories:
  # Plot only center particles and first ring member
  max_group_member: 2
  animate: false
```

With `max_group_member: null` (plot all members):

```yaml
trajectories:
  max_group_member: null
  animate: true
```

## Example: Pair Dispersion Visualization

For pair-dispersion experiments (group.size: 2):

- **Member 1**: Center particle (blue)
- **Member 2**: Offset particle (orange)

The distance between member 1 and member 2 trajectories visualizes dispersion growth.

Configuration:

```yaml
release:
  group:
    size: 2
    radius_km: 0.1
    placement: equal_angles

trajectories:
  max_group_member: 2
  animate: true
  title: "Pair Dispersion over Time"
```

For more details on grouped-particle setup and advanced analysis, see [GROUPED_PARTICLES.md](GROUPED_PARTICLES.md).

------------------------------------------------------------
PARTICLE SUMMARY
------------------------------------------------------------

The particle summary reduces each trajectory to a single row of statistics.

Example variables:
```
lon0
lat0
lonf
latf
lifetime_seconds
circle_id
```

Meaning:

lon0 lat0 → initial particle position
lonf latf → final particle position
lifetime_seconds → particle lifetime

Function:

```python
build_particle_summary()
```

This dataset is typically used for:

- release diagnostics
- exit statistics
- connectivity analyses

When present in the trajectory dataset, release metadata such as `circle_id`,
`group_id`, `group_member`, and `group_size` is preserved in the particle summary.


------------------------------------------------------------
REGULAR GRID SYSTEM
------------------------------------------------------------

Many diagnostics require aggregating particles on a spatial grid.

The RegularGrid class defines:

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
- aggregation statistics

The grid must be defined using cell edges rather than centers to avoid floating point artefacts.

GRID MODES

1) from_initial_centers

The grid is automatically constructed from the release positions.

```yaml
grid.mode: from_initial_centers
```

The grid spacing is defined by:

```yaml
dlon: 0.05
dlat: 0.05
```

2) explicit_edges

The grid limits are explicitly defined in the configuration file.

```yaml
grid.mode: explicit_edges
```

------------------------------------------------------------
BASE PRODUCTS AND PERSISTENCE POLICY
------------------------------------------------------------

The framework distinguishes between:

- base products
- analysis products

BASE PRODUCTS

The base products are:

- trajectory_table
- particle_summary

These datasets are considered the common input for several analyses.

PERSISTENCE RULE

Only the `summary` workflow is allowed to write the base products to disk.

This means:

- `summary` may build and save:
  - trajectory_table
  - particle_summary
- all other workflows may:
  - reuse them from shared context
  - read them from disk if already exported
  - recompute them in memory if missing
- but all other workflows must NOT save them

This rule avoids repeated overwriting of the same parquet/csv files
and keeps responsibilities clearly separated.

Typical use:

```yaml
analysis:
  types:
    - summary
```

or:

```yaml
analysis:
  types:
    - summary
    - density
    - beaching_times
    # - trajectories
    # - start_end_regions
```
------------------------------------------------------------
IMPLEMENTED ANALYSES
------------------------------------------------------------

Analyses are selected in the YAML configuration file.

Example:
```yaml
analysis:
  types:
    - summary
    - density
    - beaching_times
    - start_end_regions
    - trajectories
```

------------------------------------------------------------
SUMMARY WORKFLOW
------------------------------------------------------------

The `summary` workflow is the base workflow of the post-processing system.

It performs:

1) build trajectory_table
2) build particle_summary
3) save them to disk if requested

This workflow is the only one that persists the base datasets.

If the YAML contains:

```yaml
analysis:
  types:
    - summary
```
and:

```yaml
exports:
  save_trajectory_table: true
  save_particle_summary: true
  table_format: parquet
```
then the workflow writes:

- trajectory_table.parquet
- particle_summary.parquet

inside the configured output directory.


------------------------------------------------------------
UPDATED PACKAGE ARCHITECTURE
------------------------------------------------------------
```
postprocessing/

config/
    YAML configuration system

io/
    Readers and export utilities

core/
    Particle statistics and grid helpers

analyses/
    Scientific diagnostics
    - density
    - beaching_times
    - start_end_regions

plotting/
    Map and trajectory plotting utilities

runner/
    CLI and dispatcher

workflows/
    High-level workflows
    - run_summary
    - run_density
    - run_beaching_times
    - run_start_end_regions
    - run_trajectories
    - base_products
```

------------------------------------------------------------
IMPLEMENTED ANALYSES (UPDATED)
------------------------------------------------------------

The currently supported analysis types are:

- summary
- density
- beaching_times
- start_end_regions
- trajectories

These are selected in the YAML configuration:

```yaml
analysis:
  types:
    - summary
    - density
    - beaching_times
    - start_end_regions
    - trajectories
```

------------------------------------------------------------
SUMMARY
------------------------------------------------------------

Builds the two base products:

- trajectory_table
- particle_summary

This is the only workflow allowed to persist these datasets.

Use this analysis when you want to:

- inspect the cleaned trajectory table
- inspect the particle summary
- export the base datasets for reuse in later runs


------------------------------------------------------------
DENSITY
------------------------------------------------------------

Computes time-dependent particle density on a regular grid.

Input:
- trajectory_table

Output:
- density table
- density NetCDF

Variables include:

- particle_count
- particle_fraction_active [%]
- particle_fraction_total [%]

Options:

```yaml
density:
  lon_col: lon
  lat_col: lat
  time_col: time
  normalize_active: true
  normalize_total: true
  fill_ever_active_empty_with_zero: true

  animate: true
  animation_var: particle_fraction_total
  animation_label: "density [%]"
  animation_fps: 6
  animation_vmin: 0
  animation_vmax: 0.05
  show_time_bar: true
```

- `time_col`, `lon_col`  and `lat_col` string name of coordinates
- `normalize_active` and `normalize_total` calculate the fraction of trajectory in each pixel with respect to the active or total number
- `fill_ever_active_empty_with_zero` set to 0 empty (but explored at least once) pixels
- `animate` plot an animation
- `animatioon_var` which variable to plot
- `animation_label` colorbar label
- `animation_fps` fps of the gif
- `animation_vmin` and `animation_vmin` min and max values for the colorbar. setting them clips smaller than `vmin` and higher than `vmax` values
- `show_time_bar` draw a time progression bar

------------------------------------------------------------
BEACHING TIMES
------------------------------------------------------------

Computes the beaching time on the native release grid.

Input:
- particle_summary

Grid:
- reconstructed from initial release positions

Mapped variable:
- lifetime_seconds

Default aggregation:
- min

This gives a conservative estimate of the earliest beaching/exit time
associated with each release pixel.

Options:
```yaml
beaching_times:
  lon_col: lon0
  lat_col: lat0
  value_col: lifetime_seconds
  statistic: min
  plot: true
```

- `lon_col` and `lat_col` column names of the coordinates to use
- `value_col` name of the variable
- `statistic` in case of overlapping values, which statistics to use. Available options: mean, count, sum, min, max, median, std,
- `plot` draw a plot

------------------------------------------------------------
START / END REGIONS
------------------------------------------------------------

Classifies each particle according to the region of:

- initial position
- final position

Input:
- particle_summary

Region definitions are loaded from:

kinematicparcels.utilities.geographicalRegions

The analysis supports:

- how_many
- priority_level
- priority_mode
- input_lon_mode

Outputs include:

- classified particle summary
- start region map
- end region map
- optional plots

Options:
```yaml
start_end_regions:
  region_labels: null
  how_many: last
  priority_level: 6
  priority_mode: atleast
  input_lon_mode: "-180_180"
  plot: true
```

- `region_labels` regions to inglude, null includes all regions avalilable
- `priority_level` priority level to choose, behave together with `how_many` and `priority_mode`
- `how_many` which region to choose in case of multiple possibilities. Avaliabe: first (min), last (max), all, priority_min (all with min priority) or priority_max (all with max priority)
- `priority_mode` which priority level to choose. Available: exact, atleast, atmost
- `input_lon_mode` format of the longitude. Available: "-180_180", "0_360"
- `plot` draw a plot

------------------------------------------------------------
TRAJECTORIES
------------------------------------------------------------

Plots the cleaned trajectory set on a Cartopy map.

Input:
- trajectory_table

The analysis uses the stagnation-cleaned trajectories and is intended
as a diagnostic visualisation of the simulated particle paths.


Options:
```yaml
trajectories:
  plot: false
  animate: true
  title: "Trajectories"
  show_start: true
  show_end: false

  animation_fps: 6
  animation_color_by: lat0
  animation_vmin: -52.5
  animation_vmax: -51.3
  animation_label: "initial latitude"
  show_time_bar: true
  trail: true
  trail_steps: 20
```

- `plot` draw string plot
- `animate` draw the animation
- `title` string title of the plot
- `show_start` and `show_end` mark start and end point
- `animation_fps`
- `animation_color_by` and `animation_label` variable to plot as color and colorbar label
- `animation_color_by` can use summary metadata such as `circle_id`, `group_member`, or `group_id` when available
- `animation_vmin` and `animation_vmax` 
- `show_time_bar`
- `trail` and `trail_steps` plot a trail behind trajectories and its lenght
- `show_time_bar` draw a time progression bar

------------------------------------------------------------
YAML EXAMPLE (UPDATED)
------------------------------------------------------------
```yaml
dataset:
  input_path: outputs/simulation.zarr

analysis:
  types:
    - summary
    - density
    - beaching_times
    - start_end_regions
    - trajectories

exports:
  save_trajectory_table: true
  save_particle_summary: true
  table_format: parquet

cleaning:
  truncate_stagnant: true
  stagnant_tol: 1e-6
  stagnant_min_consecutive: 2

grid:
  mode: from_initial_centers
  lon_min: -74.0
  lon_max: -72.0
  lat_min: -53.0
  lat_max: -51.0
  dlon: 0.025
  dlat: 0.025

density:
  lon_col: lon
  lat_col: lat
  time_col: time
  normalize_active: true
  normalize_total: true
  fill_ever_active_empty_with_zero: false

beaching_times:
  lon_col: lon0
  lat_col: lat0
  value_col: lifetime_seconds
  statistic: min
  plot: true

start_end_regions:
  region_labels: null
  how_many: priority_max
  priority_level: null
  priority_mode: exact
  input_lon_mode: "-180_180"
  plot: true

trajectories:
  plot: true
  title: "Trajectories"
  show_start: true
  show_end: true

plotting:
  projection: PlateCarree
```

------------------------------------------------------------
CLI USAGE
------------------------------------------------------------

The post-processing pipeline is launched from the command line using:

```bash
run-parcels-postprocessing postprocess.yml
```

Alternative Python module execution:

```bash
python -m kinematicparcels.postprocessing.runner.run_postprocessing postprocess.yml
```

This command reads the YAML configuration, builds the required context,
and executes the requested analyses in sequence.


------------------------------------------------------------
WORKFLOW EXECUTION
------------------------------------------------------------

The execution driver is:

```python
run_postprocessing()
```

The runner performs:

1) load configuration
2) initialise shared context
3) run requested analyses sequentially

Pseudo-logic:

```python
for analysis in cfg.analysis.types:
    run_analysis(...)
```

Shared data such as:

trajectory_table
particle_summary
grid

are reused between analyses.


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


---

## POSTPROCESSING SERIES (MULTIPLE RUNS)

The framework also supports running post-processing on **multiple simulation outputs automatically**.

This is particularly useful when working with:

* time series of simulations
* ensemble runs
* parameter sweeps

---

## CONCEPT

The system uses:

1. A **template postprocess YAML**
2. A **master series YAML**

The series runner:

* generates one YAML per simulation
* links each YAML to the correct input dataset
* writes outputs into matching directories
* optionally executes all runs sequentially

---

## MASTER CONFIGURATION (SCHEDULE MODE)

```yaml
template_config: .\experiments\configs\postprocess.yml

series:
  simulation_output_root: C:/Users/Jacopo/Documents/DATI/PATAGONIA/simulation_series
  postprocess_output_root: C:/Users/Jacopo/Documents/DATI/PATAGONIA/postprocessing_series

  schedule:
    start_time: "2026-01-01 00:00"
    frequency: "1D"
    duration: "10D"
    input_subdir_format: "%Y%m%d-%H%M"

  dataset_filename: "output_PFall.zarr"
  config_filename: "postprocess.yml"
  runner_exe: "run-parcels-postprocess.exe"
```

---

## AUTOMATIC MAPPING

For each generated run:

INPUT:

```
simulation_output_root/<run>/output_PFall.zarr
```

OUTPUT:

```
postprocess_output_root/<run>/
```

CONFIG:

```
postprocess_output_root/<run>/postprocess.yml
```

---

## GENERATED STRUCTURE

```
simulation_series/
  20260101-0000/output_PFall.zarr
  20260102-0000/output_PFall.zarr

postprocessing_series/
  20260101-0000/postprocess.yml
  20260101-0000/<products>

  20260102-0000/postprocess.yml
  20260102-0000/<products>
```

---

## EXECUTION

Generate YAMLs only:

```bash
python run_postprocessing_series.py master_postprocess.yml --generate-only
```

Generate and execute:

```bash
python run_postprocessing_series.py master_postprocess.yml
```

---

## ALTERNATIVE: EXPLICIT RUN LIST

Instead of using a time schedule, runs can be specified manually:

```yaml
series:
  run_dirs:
    - "20260101-0000"
    - "20260102-0000"
    - "20260103-0000"
```

This is useful when:

* runs are irregular
* some runs failed and need reprocessing
* only a subset must be analysed

---

## DESIGN PRINCIPLES

The series system follows the same philosophy as the simulation runner:

* each run has its own YAML configuration
* no hidden logic inside the processing code
* full reproducibility of results
* clear mapping between input and output

The single-run postprocessing remains unchanged and fully compatible.

The series runner acts only as an orchestration layer.


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
