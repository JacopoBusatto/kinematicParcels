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
- transition probability matrices
- trajectory visualisation
- FSLE spectra
- FSLE and FTLE exponent maps

Future extensions may include:

- residence time statistics
- particle encounter metrics


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
    - fsle
    - meridional_crossing
    - start_end_regions
    - transition_probability
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
  Particle statistics, grid helpers, and shared region classification helpers

analyses/
    Scientific diagnostics
    - density
    - beaching_times
    - exponent_maps
  - meridional_crossing
    - start_end_regions
    - transition_probability

Shared region-selection helpers used by multiple analyses live under:

- `postprocessing/core/regions.py`

plotting/
    Map and trajectory plotting utilities

runner/
    CLI and dispatcher

workflows/
    High-level workflows
    - run_summary
    - run_density
    - run_beaching_times
  - run_exponent_maps
    - run_fsle
    - run_meridional_crossing
    - run_start_end_regions
    - run_trajectories
    - run_transition_probability
    - base_products
```

------------------------------------------------------------
IMPLEMENTED ANALYSES (UPDATED)
------------------------------------------------------------

The currently supported analysis types are:

- summary
- density
- beaching_times
- exponent_maps
- fsle
- meridional_crossing
- start_end_regions
- transition_probability
- trajectories

These are selected in the YAML configuration:

```yaml
analysis:
  types:
    - summary
    - density
    - beaching_times
    - exponent_maps
    - fsle
    - meridional_crossing
    - start_end_regions
    - transition_probability
    - trajectories
```

------------------------------------------------------------
COMMON YAML SECTIONS
------------------------------------------------------------

All analyses share the same top-level configuration structure.

`dataset`

- `input_path`: path to the input Parcels Zarr dataset
- `coordinates.trajectory`, `coordinates.obs`, `coordinates.time`, `coordinates.lon`, `coordinates.lat`: override variable and dimension names when the dataset does not use the Parcels defaults
- `coordinates.z`: depth variable name, or `null` for purely 2D datasets

`analysis`

- `types`: ordered list of analyses to execute; supported values are `summary`, `density`, `beaching_times`, `fsle`, `meridional_crossing`, `start_end_regions`, `transition_probability`, and `trajectories`
- `types`: ordered list of analyses to execute; supported values are `summary`, `density`, `beaching_times`, `exponent_maps`, `fsle`, `meridional_crossing`, `start_end_regions`, `transition_probability`, and `trajectories`

`output`

- `output_dir`: directory where all analysis products are written

`exports`

- `save_trajectory_table`: export the cleaned trajectory table when the `summary` workflow is run
- `save_particle_summary`: export the particle summary when the `summary` workflow is run
- `table_format`: tabular export format, currently `parquet` or `csv`

`cleaning`

- `truncate_stagnant`: truncate trajectories before stagnant segments
- `stagnant_tol`: tolerance used to compare consecutive longitude and latitude increments
- `stagnant_min_consecutive`: minimum number of consecutive stagnant steps before truncation is triggered

`grid`

- `mode`: `explicit_edges` or `from_initial_centers`
- `lon_min`, `lon_max`, `lat_min`, `lat_max`: domain bounds for the regular grid
- `dlon`, `dlat`: grid spacing in degrees

If the `grid` section is present, all bounds and spacings must be provided, even when `mode: from_initial_centers` is used.

`release`

- `mode`: release layout used by the simulation, such as `point_list`, `region_grid`, `circle`, or `lkm`
- `continuous`: whether the release was time-continuous; some grid-based outputs are only meaningful when this is `false`

`plotting`

- `projection`: Cartopy projection name used by map products, for example `PlateCarree`

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
  group_member: null

  animate: true
  animation_var: particle_fraction_total
  animation_label: "density [%]"
  animation_fps: 6
  animation_every_n: 1
  animation_vmin: 0
  animation_vmax: 0.05
  show_time_bar: true
```

- `time_col`, `lon_col`  and `lat_col` string name of coordinates
- `normalize_active` and `normalize_total` calculate the fraction of trajectory in each pixel with respect to the active or total number
- `fill_ever_active_empty_with_zero` set to 0 empty (but explored at least once) pixels
- `group_member` optionally keeps only one grouped-particle member before binning; use `null` to keep all members
- `animate` plot an animation
- `animation_var` which variable to plot
- `animation_label` colorbar label
- `animation_fps` fps of the gif
- `animation_every_n` use every Nth time slice in the animation
- `animation_vmin` and `animation_vmax` min and max values for the colorbar. Setting them clips values outside the chosen range
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
  plotting:
    enabled: true
    vmin: null
    vmax: null
```

- `lon_col` and `lat_col` column names of the coordinates to use
- `value_col` name of the variable
- `statistic` in case of overlapping values, which statistics to use. Available options: mean, count, sum, min, max, median, std,
- `plotting.enabled` draw a plot
- `plotting.vmin` and `plotting.vmax` optionally constrain the map color scale

For backward compatibility, the loader also accepts the legacy shorthand `plot: true`, which maps to `plotting.enabled`.

------------------------------------------------------------
FSLE
------------------------------------------------------------

Computes the first-kind finite-size Lyapunov exponent from grouped trajectories.

This analysis produces the FSLE spectrum as a function of separation scale. If you want gridded FSLE/FTLE maps on the release grid, use `exponent_maps` instead.

Input:
- trajectory_table

Requirements:
- grouped outputs only (`group_size > 1`)
- pair construction from grouped metadata

Estimator:
- overshoot-aware discrete estimator
- $\lambda(\delta) = \langle \ln(d / d_{old}) \rangle / \langle \tau \rangle$

Outputs:
- FSLE spectrum table
- optional crossing-event table
- optional FSLE spectrum plot

Pair modes:
- `center_pairs`: member 1 paired with each other member in the group
- `all_pairs`: all $N(N-1)/2$ pairs in the group

The error bars use the shell-spacing factor $\ln(\rho_{increment})$ and remain configurable through `rho_increment`.

Options:
```yaml
fsle:
  pair_mode: center_pairs
  meridional_only: false
  min_scale: 0.01
  max_scale: 5.0
  rho_increment: 1.4142135623730951
  save_crossing_events: false
  plot: true
  reference_slopes:
    - delta^-2/3
    - delta^-1
  reference_slope_anchor_scales:
    delta^-2/3: 0.5
    delta^-1: 2.0
  x_min: 0.01
  x_max: 5.0
  y_min: 1.0e-3
  y_max: 10.0
```

- `pair_mode` pair construction rule for grouped releases. Available: `center_pairs`, `all_pairs`
- `meridional_only` if `true`, compute pair separation from latitude differences only and ignore zonal separation
- `min_scale` and `max_scale` lower and upper separation thresholds in km
- `rho_increment` geometric factor between consecutive scales. Must be greater than 1
- `save_crossing_events` save the raw threshold-crossing events table in addition to the spectrum table
- `plot` save the FSLE spectrum plot
- `reference_slopes` choose which reference slopes to draw. Available: `delta^-2/3`, `delta^-1`, `delta^-2`
- `reference_slope_anchor_scales` choose where each reference slope should intersect the spectrum. For each selected slope, provide a target $\delta$ in km; the line is anchored to the nearest spectrum point at that scale
- `x_min`, `x_max`, `y_min`, `y_max` optional plot bounds for the log-log spectrum

Example config:
- `experiments/configs/postprocessing/postprocess_pairs_fsle.yml`

------------------------------------------------------------
EXPONENT MAPS
------------------------------------------------------------

Computes gridded FSLE and FTLE products on the native grouped release grid.

Input:
- trajectory_table
- particle_summary

Requirements:
- grouped outputs only (`group_size > 1`)
- member 1 must be the central trajectory of each group
- release centers must form a regular lon/lat grid at each release time

Behavior:
- FSLE uses member 1 versus the other group members, takes the first crossing of each requested target separation, then keeps the minimum valid crossing time across members for each group.
- FTLE uses member 1 versus the other group members, samples each requested target age with a configurable rule, then keeps the maximum valid member separation for each group.
- Backward simulations are detected from the obs/time ordering and the saved exponents are sign-adjusted to preserve the backward-time convergence interpretation used by the legacy scripts.

Outputs:
- one FSLE NetCDF with dimensions `time`, `scale_km`, `lat`, `lon`
- one FTLE NetCDF with dimensions `time`, `scale_days`, `lat`, `lon`
- optional PNG maps for each selected scale, either averaged on release time or one figure per release time

Options:
```yaml
exponent_maps:
  distance: geodesical
  require_grouped_regular_grid: true
  fsle:
    enable: true
    scale: [5.0, 10.0, 20.0]
    mask_zeros: false
    plot:
      enable: true
      average_on_time: true
      vmin: null
      vmax: null
      min_mask_value: null
      log_scale: false
      cmap: viridis
  ftle:
    enable: true
    scale: [5.0, 10.0]
    sampling_mode: last_before_or_at
    mask_short_windows: true
    mask_zeros: false
    plot:
      enable: true
      average_on_time: true
      vmin: null
      vmax: null
      min_mask_value: null
      log_scale: false
      cmap: viridis
```

- `distance` chooses the pair-distance metric. Available: `geodesical`, `meridional`
- `require_grouped_regular_grid` fails early if the input is not a grouped regular release map product
- `fsle.scale` target separations in km
- `fsle.mask_zeros` stores `NaN` instead of `0` when a group never reaches the requested scale
- `ftle.scale` target ages in days
- `ftle.sampling_mode` chooses how the age-window sample is selected: `last_before_or_at` uses an exact hit at the target age when available, otherwise the last observation before the target age only if the trajectory also extends beyond the target; `max_within_window` uses the maximum separation within the window
- `ftle.mask_short_windows` stores `NaN` instead of `0` when a group does not live long enough to cover the requested window
- `ftle.mask_zeros` stores `NaN` instead of `0` when the sampled separation does not exceed the initial separation
- `plot.average_on_time` averages the release-time dimension before plotting; otherwise the workflow writes one map per release time and scale
- `plot.min_mask_value` masks small absolute exponent values before plotting
- `plot.log_scale` uses a logarithmic color scale and therefore requires strictly positive plotted values

Example config:
- `experiments/configs/postprocessing/postprocess_exponent_maps.yml`

------------------------------------------------------------
MERIDIONAL CROSSING
------------------------------------------------------------

Computes directional meridional-crossing statistics on a regular grid.

Input:
- trajectory_table

Outputs:
- meridional crossing grid table
- meridional crossing NetCDF
- optional northward and southward probability plots
- optional northward and southward count plots

The workflow segments trajectories into coherent northward or southward motion,
detects crossings of grid latitudes, and aggregates the results per release grid cell.

Options:
```yaml
meridional_crossing:
  direction: both

  segmentation:
    lat_filter: rolling_mean
    filter_window: 5
    direction_threshold_deg: auto
    min_segment_duration_days: 1.5
    min_segment_displacement_deg: auto
    valid_if: duration_or_displacement

  crossing:
    crossing_latitude_reference: center
    count_once_per_segment_per_lat_bin: true

  output:
    save_netcdf: true
    save_grid_table: true
    save_figures: true

  plotting:
    enabled: true
    probability:
      enabled: true
      vmin: null
      vmax: null
    count:
      enabled: false
      vmin: null
      vmax: null
```

- `direction` selects which crossing directions are considered: `northward`, `southward`, or `both`
- `segmentation.lat_filter` smooths the latitude signal before segment detection; available values are `rolling_mean`, `rolling_median`, and `none`
- `segmentation.filter_window` sets the smoothing window length in samples
- `segmentation.direction_threshold_deg` is the minimum signed latitudinal displacement used to classify motion direction; `auto` derives it from the data
- `segmentation.min_segment_duration_days` requires each accepted segment to last at least this many days
- `segmentation.min_segment_displacement_deg` requires each accepted segment to span at least this many degrees of latitude; `auto` derives it from the data
- `segmentation.valid_if` currently supports `duration_or_displacement`
- `crossing.crossing_latitude_reference` chooses whether crossings are tested against latitude-bin centers or edges
- `crossing.count_once_per_segment_per_lat_bin` prevents multiple counts from the same segment within the same latitude bin when `true`
- `output.save_netcdf`, `output.save_grid_table`, and `output.save_figures` control which artifacts are written
- `plotting.enabled` enables figure generation globally for this analysis
- `plotting.probability.enabled` and `plotting.count.enabled` control the probability and count maps separately
- `plotting.probability.vmin`, `plotting.probability.vmax`, `plotting.count.vmin`, and `plotting.count.vmax` optionally constrain the map color scales

The preferred keys are the nested `plotting.probability.enabled` and `plotting.count.enabled`. The loader also accepts the older aliases `show_probability` and `show_counts` for backward compatibility.

------------------------------------------------------------
START / END REGIONS
------------------------------------------------------------

Classifies each particle according to the region of:

- initial position
- final position

Input:
- particle_summary

Region definitions are loaded from:

kinematicparcels.regions

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
- optional connectivity plots and animations

Options:
```yaml
start_end_regions:
  region_labels: null
  how_many: priority_max
  priority_level: null
  priority_mode: exact
  input_lon_mode: "-180_180"
  plot: false

  plot_connectivity: false
  animate_connectivity: false
  connectivity_segments: true
  connectivity_color_by: start_region
  connectivity_label: region
  connectivity_title: "Trajectories by region"
  connectivity_show_start: true
  connectivity_show_end: true
  connectivity_alpha: null
  connectivity_max_group_member: null
  connectivity_animation_fps: null
  connectivity_animation_show_tracer: null
  connectivity_trail: null
  connectivity_trail_steps: null
  discrete_cmap: null
  colorbar_label_mode: numeric
  show_region_labels: false
```

- `region_labels` regions to include; `null` includes all available regions
- `priority_level` priority level to choose, behave together with `how_many` and `priority_mode`
- `how_many` which region to choose in case of multiple possibilities. Available: first (min), last (max), all, priority_min (all with min priority), or priority_max (all with max priority)
- `priority_mode` which priority level to choose. Available: exact, atleast, atmost
- `input_lon_mode` format of the longitude. Available: "-180_180", "0_360"
- `plot` draw the start-region and end-region maps when grid outputs are meaningful
- `plot_connectivity` save connectivity PNGs coloured by the selected region label
- `animate_connectivity` save a connectivity animation using the full trajectory table
- `connectivity_segments` use straight start-to-end segments instead of full trajectories for the static connectivity plot
- `connectivity_color_by` choose which classified field colors the connectivity products, typically `start_region` or `end_region`
- `connectivity_label` colorbar label for connectivity products
- `connectivity_title` title used for the region-coloured trajectory products
- `connectivity_show_start` and `connectivity_show_end` mark start and end points in the connectivity plot
- `connectivity_alpha` optionally overrides the trajectory alpha used by connectivity plots
- `connectivity_max_group_member` optionally overrides the grouped-particle member filter for connectivity plots and animations
- `connectivity_animation_fps` optionally overrides the default trajectory animation frame rate
- `connectivity_animation_show_tracer` controls whether the moving tracer marker is drawn in the connectivity animation
- `connectivity_trail` and `connectivity_trail_steps` optionally override the trail settings used in the connectivity animation
- `discrete_cmap` optionally sets the colormap used by start/end discrete region maps (for example `Set3`, `tab20b`); `null` uses the default
- `colorbar_label_mode` controls how start/end map colorbar ticks are named: `numeric`, `region_label`, or `region_name`
- `show_region_labels` draws text labels inside cells on the start-region map; the end-region map remains unannotated

The gridded start/end maps are produced when `release.mode: region_grid`; in continuous runs, per-cell classes use modal aggregation.

------------------------------------------------------------
TRANSITION PROBABILITY
------------------------------------------------------------

Computes the time-dependent transition probability matrix between a selected
set of geographic regions.

Input:
- trajectory_table

For each retained time step, the workflow:

- classifies every particle position into one of the selected regions
- determines the region where each particle started
- excludes particles whose initial point is outside the selected regions
- counts how many particles that started in region `i` are in region `j`
- normalizes by the number of particles that started in region `i`

The exported CSV contains one row per sampled particle age and one probability
column per ordered region pair:

```text
age_days, p_<origin1>__<target1>, ..., p_<originN>__<targetN>
```

The analysis reuses the same region-selection options as `start_end_regions`
and warns when the selected regions do not all share the same priority level.

Options:
```yaml
transition_probability:
  region_labels:
    - sesc-mod
    - sesc-sir
  time_step_stride: 1
  how_many: priority_max
  priority_level: null
  priority_mode: exact
  input_lon_mode: "-180_180"
  min_life_days: 0
  trimming_age_days: null
  max_group_member: null
  filter_isolated: false
  plotting:
    enabled: false
    x_log_scale: false
    y_log_scale: false
    colormap: null
```

- `region_labels` list of regions to include in the matrix; required
- `time_step_stride` retain one point every N input time steps
- `how_many`, `priority_level`, `priority_mode`, `input_lon_mode` follow the same meaning as in `start_end_regions`
- `min_life_days` keep only particles whose total lifetime is at least this many days
- `trimming_age_days` discard samples whose particle age exceeds this value; no interpolation is performed at the cutoff
- `max_group_member` when grouped trajectories are present, include members up to this index; `null` keeps all members
- `filter_isolated` replace isolated symbolic labels when previous and next labels are equal and non-null
- `plotting.enabled` saves the original comprehensive transition plot and additional plots decomposed by starting region
- `plotting.x_log_scale` and `plotting.y_log_scale` switch the corresponding axes to log scale; non-positive values are masked on log axes
- `plotting.colormap` optionally selects the Matplotlib categorical colormap used for the starting-region colors, for example `Paired`, `tab10`, `Dark2`, or `Set1`

The normalization denominator is still the number of particles that started in each
origin region. Setting `trimming_age_days` alone does not force a constant denominator
across all exported ages; use `min_life_days = trimming_age_days` when that is desired.

Output:

- `transition_probability.csv`
- `transition_probability_plot.png` when `transition_probability.plotting.enabled` is `true`
- `transition_probability_<origin>_plot.png` when `transition_probability.plotting.enabled` is `true`

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
  plot: true
  alpha: 0.7
  plot_color_by: null
  plot_cmap: null
  plot_cmap_mode: auto

  animate: true
  title: "Trajectories"
  show_start: true
  show_end: true

  animation_fps: 6
  animation_every_n: 1
  animation_color_by: lat0
  animation_cmap: null
  animation_cmap_mode: auto
  animation_vmin: -52.5
  animation_vmax: -51.3
  animation_label: "initial latitude"
  show_time_bar: true
  trail: true
  trail_steps: 20
  max_group_member: null
```

- `plot` draw a static trajectory plot
- `animate` draw the animation
- `title` string title of the plot
- `show_start` and `show_end` mark start and end point
- `alpha` line transparency used by the static plot
- `plot_color_by` selects the summary or trajectory column used to color the static plot; `null` enables automatic grouped-member coloring when available
- `plot_cmap` explicitly chooses the static-plot colormap
- `plot_cmap_mode` controls whether the static coloring is treated as `auto`, `categorical`, or `numeric`
- `animation_fps`
- `animation_every_n` use every Nth frame in the animation
- `animation_color_by` and `animation_label` variable to plot as color and colorbar label
- `animation_color_by` can use summary metadata such as `circle_id`, `group_member`, or `group_id` when available
- `animation_cmap` explicitly chooses the animation colormap
- `animation_cmap_mode` controls whether the animation coloring is treated as `auto`, `categorical`, or `numeric`
- `animation_vmin` and `animation_vmax` optionally constrain the animation color scale
- `show_time_bar`
- `trail` and `trail_steps` plot a trail behind trajectories and its length
- `max_group_member` keeps only grouped members up to the selected index; `null` keeps all members
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
    - fsle
    - meridional_crossing
    - start_end_regions
    - transition_probability
    - trajectories

output:
  output_dir: outputs/postprocessing/example

release:
  mode: region_grid
  continuous: false

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
  group_member: null
  animate: false
  animation_every_n: 1

beaching_times:
  lon_col: lon0
  lat_col: lat0
  value_col: lifetime_seconds
  statistic: min
  plotting:
    enabled: true
    vmin: null
    vmax: null

fsle:
  pair_mode: center_pairs
  meridional_only: false
  min_scale: 0.01
  max_scale: 10.0
  rho_increment: 1.4142135623730951
  save_crossing_events: false
  plot: true

meridional_crossing:
  direction: both
  segmentation:
    lat_filter: rolling_mean
    filter_window: 5
    direction_threshold_deg: auto
    min_segment_duration_days: 1.5
    min_segment_displacement_deg: auto
    valid_if: duration_or_displacement
  crossing:
    crossing_latitude_reference: center
    count_once_per_segment_per_lat_bin: true
  output:
    save_netcdf: true
    save_grid_table: true
    save_figures: true
  plotting:
    enabled: true
    probability:
      enabled: true
      vmin: null
      vmax: null
    count:
      enabled: false
      vmin: null
      vmax: null

start_end_regions:
  region_labels: null
  how_many: priority_max
  priority_level: null
  priority_mode: exact
  input_lon_mode: "-180_180"
  plot: true
  plot_connectivity: false
  animate_connectivity: false
  connectivity_segments: true

transition_probability:
  region_labels:
    - sesc-mod
    - sesc-sir
  time_step_stride: 2
  how_many: priority_max
  priority_level: 7
  priority_mode: exact
  input_lon_mode: "-180_180"
  min_life_days: 0
  trimming_age_days: null
  max_group_member: null
  filter_isolated: true

trajectories:
  plot: true
  title: "Trajectories"
  alpha: 0.7
  show_start: true
  show_end: true
  plot_color_by: null
  plot_cmap: null
  plot_cmap_mode: auto
  animate: false
  animation_every_n: 1
  animation_color_by: lat0
  animation_cmap: null
  animation_cmap_mode: auto
  animation_vmin: null
  animation_vmax: null
  animation_label: value
  show_time_bar: true
  trail: true
  trail_steps: null
  max_group_member: null

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
template_config: .\experiments\configs\postprocessing\postprocess.yml

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
