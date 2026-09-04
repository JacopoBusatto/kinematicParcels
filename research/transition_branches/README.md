# Lagrangian transport and directional structures

This package turns a normalized sparse transition matrix into two complementary,
independent structural analyses:

```text
transition matrix
    |
    +-> displacement-weighted transport field
    |       -> transport cores
    |       -> probable transport fronts
    |
    +-> distance-free directional field
            -> directional corridors
            -> probable directional fronts
```

It does not identify named currents or classify bifurcations, merging, saddles,
or other topology. The products remain spatially and dynamically agnostic.

## Scientific contract

For every populated source cell, supplied transition probabilities must sum to
one within numerical tolerance. `transition_count` is the sampling support and
the code never renormalizes missing probability mass. Stay and moving
transitions remain distinct. Unsupported quantities remain `NaN`; missing
support is never represented as zero or interpreted as a front.

All distances, displacement vectors, component lengths, and cross-stream
samples use the selected geometry backend. `geographic` coordinates are
longitude/latitude degrees and use ellipsoidal geodesics. `cartesian`
coordinates are planar x/y values measured directly in `geometry.length_unit`;
they require no CRS. Every selection uses the configured
`statistics.min_moving_support`. The current ARGO and drifter production YAMLs
set that threshold to 2 and 3 respectively; the default is 10.

## Displacement-weighted transport pathway

The full-population finite-time transport vector is

```text
U_out_all = (1 / delta_t) sum(j != i) P_ij delta_r_ij
```

and satisfies

```text
U_out_all = P_move * U_out_move.
```

It answers: **Where is finite-time net transport strongest?** Transition
displacement length contributes directly to this vector.

### Transport cores

Core detection is a support-aware transverse-ridge test:

1. Cells with `N_out_move < statistics.min_moving_support` are excluded.
2. The selected scalar field is either raw `|U_out_all|` or its support-aware
   3 x 3 mean. Smoothing never fills an unsupported focal cell.
3. At every remaining cell, the code constructs a geometry-aware cross-stream direction
   perpendicular to the local mean transport bearing `theta_mu_out`. The
   sampling distance is
   `branches.transverse_scale_grid` times the local effective grid scale,
   defined as the geometric mean of the x/zonal and y/meridional cell sizes.
4. The scalar field is sampled on both sides with supported bilinear
   interpolation. A two-sided ridge requires the centre value to be at least as
   large as both flank values, within
   `branches.ridge_comparison_tolerance`. If exactly one side is
   observable, the boundary-aware test applies the same comparison to that
   side. A cell with neither side observable cannot be a core.
5. A ridge candidate is retained only if its scalar value is at or above the
   configured `branches.transport_percentile`, calculated over all supported
   cells with a finite selected field.
6. Retained cells are joined through eight-neighbour connectivity, including
   longitude wrapping when configured. Components are split into graph
   segments at endpoints and junctions so that fronts can be evaluated along
   locally ordered pieces.

The output labels a retained cell as `two_sided` or `one_sided`. Components,
junctions, and graph degree are neutral pixel-graph descriptions and are not
interpreted as physical current topology.

### Probable transport fronts

For each core cell in each graph-segment context, the algorithm:

1. Samples a geometry-aware transverse section extending
   `edges.half_width_grid_scales` on either side, at
   `edges.sampling_interval_grid_scales` intervals.
2. Projects every sampled transport vector onto the central
   `theta_mu_out` tangent. That tangent is fixed inside one section but may
   rotate between sections.
3. Refines the core axis to the maximum supported projected transport within
   `edges.core_refinement_grid_scales` of the original core cell.
4. Applies a contiguous rolling median; missing samples break a run rather than
   being filled. It also forms an along-segment composite from nearby sections
   when those sections are available.
5. On each observable side, tests successive valid outward samples for a
   positive drop. The next one or two later valid samples must remain lower.
   When an along-segment composite is available, the decline must also satisfy
   the configured minimum number and fraction of neighbouring sections.
6. Ranks eligible drops by persistence, absolute loss, and loss steepness,
   retaining one candidate per side and segment context. Duplicate
   segment-context results for the same physical core side are reduced by their
   median position and loss.

There is deliberately no fixed minimum loss magnitude. The result is a set of
probable front points, not a continuous front line. Each core side receives one
of:

- `probable_transport_front`;
- `observable_no_retained_front`;
- `side_not_observable`.

## Distance-free directional pathway

For moving transitions, define the unit displacement vector
`r_hat_ij = delta_r_ij / |delta_r_ij|` and moving-conditioned probability
`q_ij = P_ij / P_move`. The moving-conditioned vector is

```text
D_out_move = sum(j != i) q_ij r_hat_ij.
```

This is the vector form of the existing first circular harmonic:

```text
|D_out_move| = R1_out
arg(D_out_move) = theta1_out.
```

The full-population directional vector is

```text
D_out_all = sum(j != i) P_ij r_hat_ij
          = P_move * D_out_move,

|D_out_all| = P_move * R1_out.
```

It answers: **Where does the transition population share a strong preferred
direction, independently of displacement length?** Geographic outputs use
`D_out_all_east` and `D_out_all_north`; Cartesian outputs use `D_out_all_x` and
`D_out_all_y`. These components and `D_out_all_magnitude` are dimensionless. A
zero moving probability gives a zero full-population vector but no artificial
moving bearing; a vanishing first harmonic likewise has no artificial direction.

### Directional corridors

Corridors use absolute, bounded, scientifically interpretable thresholds rather
than a global percentile. A cell must have:

- configured moving support;
- `P_move >= directional.minimum_P_move`;
- `R1_out >= directional.minimum_R1`;
- `P_move * R1_out >= directional.minimum_strength`;
- a defined `theta1_out`.

Eligible neighbouring cells connect only when their directions agree under a
circular angular difference and their cell-to-cell connection is approximately
along the local direction at both endpoints. Components shorter than
`directional.minimum_component_cells` are discarded. Direction is evaluated
locally: a corridor may turn through 360/0 degrees and may bend substantially
over long distances as long as neighbouring changes remain compatible. No
fixed component-wide compass bearing is imposed.

Operationally, the candidate cells are placed on an eight-neighbour graph. An
edge is retained only when:

- the circular difference between the two `theta1_out` bearings does not
  exceed `directional.maximum_neighbor_direction_difference_degrees`; and
- the geometry-aware axis joining the cell centres agrees with the local directional
  axis at both endpoints within
  `directional.maximum_step_direction_mismatch_degrees`.

Connected components smaller than `directional.minimum_component_cells` are
removed. Unlike transport cores, corridor membership has no transverse-ridge
or global-percentile requirement.

There is no transverse-ridge or width requirement. A one-cell-wide or
two-cell-wide coherent band is valid; its physical width is resolution-limited
by the grid. Transverse observability is evaluated separately, so such a narrow
corridor may have two observable flanks. A coastal corridor may have only its
offshore side observable. A cell with neither supported flank can remain a
directional-corridor cell while both possible fronts are explicitly
`side_not_observable`.

### Directional fronts

Each corridor cell defines a geometry-aware transverse section using its own central
directional tangent. Within that section the tangent is fixed and

```text
D_parallel(s) = D_out_all(s) dot t_hat_0
              = P_move(s) R1(s) cos(theta1(s) - theta1_0).
```

The tangent rotates from section to section as a corridor curves. `D_parallel`
decreases when directional organization or moving probability weakens, when
nearby flow turns away, or when it reverses; opposing flow may make it negative.

The front calculation reuses the configured forward sampling, supported bilinear
interpolation, axis refinement, contiguous smoothing, outward-drop, and
persistence concepts used for transport fronts. Here the axis is refined to
the maximum supported `D_parallel`, and the along-corridor composite uses
neighbours within the configured number of graph hops rather than ordered
transport segments. An outward decrease is eligible only when subsequent
contiguous samples remain lower and, when neighbour information exists, the
decline passes the persistence thresholds. Missing samples break a profile run
and can never form a candidate drop. Scientist-facing statuses are:

- `probable_directional_front`;
- `observable_no_retained_directional_front`;
- `side_not_observable`.

Directional corridors/fronts are calculated without transport core/front input.
The two pathways may overlap, but neither seeds, constrains, or validates the
other.

## Transport/directional comparison

Every cell satisfying the configured support threshold is classified as:

```text
transport_and_directional
directional_only
transport_only
neither
```

The component comparison reports overlap fractions from each component's own
cell set and deliberately performs no one-to-one component matching. A
`directional_only` cell can represent slower but persistent motion. A
`transport_only` cell can represent strong displacement-weighted transport with
competing transition directions. Neither category is treated as a failure.

## Modules

- `statistics.py`: matrix validation, transport/circular statistics, and the
  directional-vector identities derived from the retained first harmonic.
- `cores.py` and `fronts.py`: unchanged validated transport-core/front pathway.
- `directional_corridors.py`: absolute-threshold, locally compatible curved
  corridor graph and boundary-aware observability.
- `directional_fronts.py`: local `D_parallel` sections and persistent
  directional-front outcomes.
- `comparison.py`: neutral cell/component overlap diagnostics.
- `geometry.py`: shared angles, supported interpolation, physical scales, and
  grid helpers.
- `validation.py`: optional transport-front cross-stream-gradient validation.
- `plotting.py`: seven production maps plus an optional validation map.
- `io.py`: scientific tables, reproducibility files, and optional debug tables.
- `config.py`: one production realization per YAML configuration.
- `workflow.py`: the single command-line production path.
- `_statistics_kernel.py`, `_ridge_kernel.py`, and `_edge_kernel.py`: verified
  low-level calculations retained to reproduce the validated transport method.

## Geometry modes and units

Choose exactly one coordinate system:

- `geographic` means longitude/latitude coordinates in degrees. Inverse and
  forward calculations use `pyproj.Geod` with `geometry.ellipsoid`; its metre
  results are converted to `geometry.length_unit`.
- `cartesian` means genuine planar x/y coordinates. Input coordinates, grid
  spacing, calculated distances, and `geometry.length_unit` are the same unit.
  Use this mode when the transition matrix is defined directly in Cartesian
  coordinates. No CRS is needed or accepted, and Cartesian x is not periodic
  in this version.

In both modes bearings are degrees clockwise from positive y: 0 degrees is
positive y (north geographically), 90 degrees is positive x (east
geographically), 180 degrees is negative y, and 270 degrees is negative x.
`input.timestep` is expressed in `input.time_unit`. The code divides a
displacement by that number, so a 30-day matrix uses `timestep: 30` and
`time_unit: day`, while a 2-second Cartesian matrix uses `timestep: 2` and
`time_unit: s`.

Minimal geographic configuration:

```yaml
input:
  transition_table: matrix.parquet
  matrix_id: argo_30d
  timestep: 30
  time_unit: day
output:
  root: outputs
geometry:
  coordinate_system: geographic
  length_unit: km
  ellipsoid: WGS84
grid:
  lon_min: -180
  lon_max: 180
  lat_min: -80
  lat_max: -30
  dlon: 1
  dlat: 1
  periodic_longitude: true
```

Minimal Cartesian configuration (a commented, ready-to-edit version is provided
in [`configs/cartesian_example.yaml`](configs/cartesian_example.yaml)):

```yaml
input:
  transition_table: cartesian_matrix.parquet
  matrix_id: cartesian_2s
  timestep: 2
  time_unit: s
output:
  root: outputs
  run_name: cartesian
geometry:
  coordinate_system: cartesian
  length_unit: cm
grid:
  x_min: 0
  x_max: 120
  y_min: 0
  y_max: 60
  dx: 2
  dy: 2
```

Supported length units are `mm`, `cm`, `m`, and `km`; supported time units are
`s`, `min`, `h`, and `day`. A geographic `length_unit` does not change the
degree-valued input coordinates. It selects the linear unit used after the
ellipsoidal calculation. For example, geographic coordinates with
`length_unit: km` produce distances in kilometres and rates in kilometres per
configured time unit.

## Required input Parquet schema

The input is a sparse transition table with one row per observed directed
source/destination bin pair. Its DataFrame index is ignored and no `cell_id`
column is required. Bin indices are zero-based. Internally a cell is indexed
row-major as `y_bin * nx + x_bin` (or `lat_bin * nlon + lon_bin`).

Geographic files require exactly these named fields:

| Column | dtype | Unit/meaning |
| --- | --- | --- |
| `start_lon_bin`, `start_lat_bin` | integer | Zero-based source-bin indices. |
| `end_lon_bin`, `end_lat_bin` | integer | Zero-based destination-bin indices. |
| `start_lon_center`, `start_lat_center` | floating point | Source centre in degrees. |
| `end_lon_center`, `end_lat_center` | floating point | Destination centre in degrees. |
| `transition_count` | integer | Positive observed transition count for this link. |
| `transition_probability` | floating point | Source-normalized probability in `[0, 1]`. |

Cartesian files require:

| Column | dtype | Unit/meaning |
| --- | --- | --- |
| `start_x_bin`, `start_y_bin` | integer | Zero-based source-bin indices. |
| `end_x_bin`, `end_y_bin` | integer | Zero-based destination-bin indices. |
| `start_x_center`, `start_y_center` | floating point | Source centre in `geometry.length_unit`. |
| `end_x_center`, `end_y_center` | floating point | Destination centre in `geometry.length_unit`. |
| `transition_count` | integer | Positive observed transition count for this link. |
| `transition_probability` | floating point | Source-normalized probability in `[0, 1]`. |

There is no column mapping or coordinate auto-detection. A geographic run
rejects the Cartesian names and vice versa. Extra columns are ignored. The
required bin and count columns must have an integer dtype, while centres and
probabilities must have a floating-point dtype.

Validation is fail-fast. The table must be non-empty and finite, keys must be
unique, counts must be positive, bins must lie within the configured grid, and
stored centres must equal `minimum + (bin + 0.5) * spacing` within
`validation.center_atol`. For each populated source, probabilities must sum to
one. Every link must also satisfy
`transition_probability = transition_count / source_total` within the
configured tolerances. The workflow neither regrids nor renormalizes data.

## Configuration reference

Configuration files use YAML. The examples in `configs/` override only the
values needed for each production run; any omitted optional value comes from
`config.py`. The `input`, `output`, `geometry`, and `grid` sections are
required. Relative input and output paths are resolved relative to the YAML
file, and `~` is expanded.
The output directory is
`<output.root>/<output.run_name>_<UTC timestamp>`.

Use the parameter names exactly as documented. Unknown keys are rejected so a
misspelling cannot silently change a run. The removed `input.timestep_days`,
top-level `ellipsoid`, and old unit-specific tolerance names are errors.

### Input and output

| Parameter | Default | Meaning |
| --- | --- | --- |
| `input.transition_table` | required | Parquet file containing the sparse normalized transition table. |
| `input.matrix_id` | required | Non-empty human-readable identifier recorded in the manifest; it does not select or modify data. |
| `input.timestep` | required | Positive transition lag expressed in `input.time_unit`; it must match matrix generation. |
| `input.time_unit` | required | Time unit used by `timestep` and output rates: `s`, `min`, `h`, or `day`. |
| `output.root` | required | Parent directory in which the timestamped run directory is created. |
| `output.run_name` | `lagrangian_currents` | Prefix of the timestamped run directory. |

### Geometry

| Parameter | Default | Meaning |
| --- | --- | --- |
| `geometry.coordinate_system` | required | `geographic` for degree lon/lat with geodesics, or `cartesian` for planar x/y. |
| `geometry.length_unit` | required | Linear calculation/output unit: `mm`, `cm`, `m`, or `km`. It is also the input-coordinate unit in Cartesian mode. |
| `geometry.ellipsoid` | geographic: required | Ellipsoid name passed to `pyproj.Geod`. It must be omitted for Cartesian geometry. |

Projected-Earth coordinates and CRS transformations are outside this workflow.

### Grid

The YAML grid must be the grid used to create the transition table. The
workflow does not regrid the table. It validates observed bin bounds and
recomputes the stored cell centres using the configured origin and spacing, so
an origin or resolution mismatch normally stops the run with
`bin_out_of_bounds` and/or `grid_center_mismatch`. A sparse table cannot
always reveal an incorrect outer extent or periodicity, however, so those
values must be checked by the user.

Every span must be an exact integer multiple of its positive spacing. Grid
fields have no defaults and must all be supplied.

Geographic grid:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `grid.lon_min` | required | Western grid edge in degrees. |
| `grid.lon_max` | required | Eastern grid edge in degrees. |
| `grid.lat_min` | required | Southern grid edge in degrees, at least -90. |
| `grid.lat_max` | required | Northern grid edge in degrees, at most 90. |
| `grid.dlon` | required | Positive longitudinal cell width in degrees. |
| `grid.dlat` | required | Positive latitudinal cell height in degrees. |
| `grid.periodic_longitude` | `true` | Wrap interpolation and eight-neighbour connectivity across the first/last longitude columns. Use only for a cyclic longitude domain. |

Cartesian grid (non-periodic):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `grid.x_min`, `grid.x_max` | required | Planar x edges in `geometry.length_unit`, with max greater than min. |
| `grid.y_min`, `grid.y_max` | required | Planar y edges in `geometry.length_unit`, with max greater than min. |
| `grid.dx`, `grid.dy` | required | Positive planar cell sizes in `geometry.length_unit`. |

### Transition statistics

| Parameter | Default | Meaning |
| --- | --- | --- |
| `statistics.min_moving_support` | `10` | Minimum moving transition count `N_out_move` for all core, corridor, front, comparison, and plotted selections. Lower values retain more weakly sampled cells; higher values are more conservative. |
| `statistics.angular_bins` | `36` | Number of equal bearing bins used for outgoing and incoming angular entropy; must be at least four. It does not change the first harmonic `R1`. |
| `statistics.direction_zero_tolerance` | `1e-12` | Numerical tolerance, in the configured length unit for displacement vectors, below which magnitude is treated as zero and its bearing is undefined. |
| `statistics.high_R1` | `0.8` | High first-harmonic threshold used in statistical diagnostic categories and summaries. It is not the corridor-selection threshold. |
| `statistics.low_R1` | `0.5` | Low first-harmonic threshold used in diagnostic categories; it must not exceed `high_R1`. |

### Transport-core selection

| Parameter | Default | Meaning |
| --- | --- | --- |
| `branches.transport_percentile` | `0.9` | Quantile of the selected transport field above which transverse ridge candidates are retained. Lowering it yields more cores; increasing it retains only stronger transport. Must lie strictly between zero and one. |
| `branches.ridge_field` | `raw` | Core field: `raw` uses `|U_out_all|`; `smoothed` uses a support-aware 3 x 3 mean. |
| `branches.transverse_scale_grid` | `1.0` | Distance from a cell centre to each sample used by the transverse-ridge test, in local effective grid scales. |
| `branches.interpolation_weight_tolerance` | `1e-10` | Numerical bilinear-weight cutoff. Corners with weights at or below this value are ignored; every other contributing corner must be supported. Shared by core, transport-front, corridor-observability, and directional-front sampling. |
| `branches.orientation_reliable_R1` | `0.8` | `R1_out` threshold for the reliable-orientation diagnostic. It does not filter core membership. |
| `branches.orientation_ambiguous_R1` | `0.5` | `R1_out` threshold for the ambiguous-orientation diagnostic. It does not filter core membership. |
| `branches.direction_disagreement_degrees` | `20.0` | Maximum `theta_mu_out` versus `theta1_out` disagreement used by the reliable-orientation diagnostic; not a core filter. |
| `branches.abrupt_tangent_mismatch_degrees` | `45.0` | Threshold used to flag a large mismatch between a core graph segment and the mean transport direction; diagnostic only. |
| `branches.ridge_comparison_tolerance` | `1e-12` | Numerical allowance in the configured rate unit for the centre-versus-flank ridge comparison. |
| `branches.smoothing_window_cells` | `3` | Fixed validated smoothing window. The current implementation accepts only `3`. |

### Directional-corridor selection

The three strength criteria are simultaneous. Since
`D_out_all_magnitude = P_move * R1_out`, `minimum_strength` constrains their
combined signal in addition to their separate lower bounds.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `directional.minimum_P_move` | `0.5` | Minimum probability that a transition leaves its source cell. |
| `directional.minimum_R1` | `0.8` | Minimum concentration of moving-transition bearings. |
| `directional.minimum_strength` | `0.5` | Minimum dimensionless `P_move * R1_out`. |
| `directional.maximum_neighbor_direction_difference_degrees` | `45.0` | Largest circular `theta1_out` difference allowed between connected candidate cells. |
| `directional.maximum_step_direction_mismatch_degrees` | `45.0` | Largest mismatch allowed between the geodesic cell-to-cell axis and the local directional axis at either endpoint. |
| `directional.minimum_component_cells` | `3` | Minimum number of connected cells required to retain a corridor component. |
| `directional.transverse_scale_grid` | `1.0` | Transverse distance, in local effective grid scales, used to determine whether each corridor side is observable. It does not change corridor membership, but it affects directional-front availability. |

Lower strength thresholds or larger angular tolerances produce more and
potentially less coherent corridors. Smaller angular tolerances and larger
strength thresholds are more selective.

### Shared front detection

These parameters control both probable transport fronts and probable
directional fronts unless stated otherwise. Distances expressed in grid scales
use the local effective cell scale. In geographic mode its size changes with
latitude and resolution; in Cartesian mode it is the geometric mean of `dx`
and `dy`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `edges.half_width_grid_scales` | `5` | Distance sampled on each side of a core/corridor axis; must be at least two grid scales. |
| `edges.sampling_interval_grid_scales` | `1.0` | Spacing between transverse samples. Smaller values give denser interpolation and higher cost. |
| `edges.core_refinement_grid_scales` | `1.0` | Maximum distance over which the original core/corridor axis is shifted to the local projected-field maximum; must be in `(0, 1]`. |
| `edges.robust_median_window_samples` | `3` | Positive odd window for the contiguous rolling-median profile. Missing data split the profile into separate runs. |
| `edges.composite_half_window_sections` | `2` | Context used for composite profiles: plus/minus this many ordered sections on a transport segment, or this many graph hops for a directional corridor. Zero uses only the focal section. |
| `edges.minimum_persistent_neighbor_sections` | `2` | Minimum number of composite sections that must show an outward decline when neighbour information is available. |
| `edges.minimum_persistent_fraction` | `0.5` | Minimum fraction of available composite sections that must show the decline. Larger values make front selection stricter. |
| `edges.diagnostic_low_R1` | `0.5` | Flags low directional concentration on transport-front sections; diagnostic only. |
| `edges.diagnostic_large_direction_disagreement_degrees` | `20.0` | Flags disagreement between mean-vector and first-harmonic bearings on transport-front sections; diagnostic only. |
| `edges.diagnostic_high_curvature_degrees` | `60.0` | Flags highly turning transport-core segments; diagnostic only. |
| `edges.diagnostic_strong_outer_recovery_fraction` | `0.5` | Flags a transport profile whose signal recovers outward by more than this fraction of its selected drop; it does not reject the front. |
| `edges.diagnostic_min_full_section_valid_samples` | `7` | Minimum valid samples before a section avoids the `short_available_cross_section` quality flag. |
| `edges.nearby_branch_cross_distance_scales` | `5.0` | Cross-stream search distance used to flag contamination by another transport-core component; diagnostic only. |
| `edges.nearby_branch_along_distance_scales` | `1.0` | Along-stream search distance used with the preceding contamination diagnostic. |

### Validation

Transition-table validation always runs. The first four parameters below
control that fail-fast check. The remaining parameters affect only the optional
transport-gradient comparison enabled by `run_validation: true`; this
comparison never moves, accepts, or rejects a selected front.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `validation.normalization_atol` | `1e-12` | Absolute tolerance for requiring each populated source row to sum to probability one. |
| `validation.probability_rtol` | `1e-10` | Relative tolerance for checking `transition_probability = transition_count / N_out_total`. |
| `validation.probability_atol` | `1e-12` | Absolute tolerance for the same count/probability identity. |
| `validation.center_atol` | `1e-9` | Absolute centre-coordinate tolerance: degrees geographically or `geometry.length_unit` in Cartesian mode. |
| `validation.gradient_zero_tolerance` | `1e-12` | Gradient magnitude at or below which the optional gradient orientation is considered undefined. |
| `validation.interpolation_weight_tolerance` | `1e-10` | Bilinear-weight cutoff used only when sampling optional validation gradients. |
| `validation.gradient_search_radius_grid_scales` | `1.0` | Radius around a selected transport front used to find the local maximum transverse gradient. |
| `validation.local_background_radius_grid_scales` | `2.0` | Radius around the refined core used to calculate the front gradient's local percentile. |
| `validation.duplicate_disagreement_grid_scales` | `1.0` | Candidate-position spread above which duplicate segment-context fronts for one core side are flagged as disagreeing. |
| `validation.core_gradient_ratio_epsilon` | `1e-12` | Denominator guard for the optional flank-to-core gradient ratio. |
| `validation.multiple_drop_similarity_fraction` | `0.1` | Reserved configuration field; it is currently not used by the production or validation calculations. |
| `validation.direct_sample_atol_grid_cells` | `1e-8` | Grid-coordinate tolerance for labeling a validation gradient sample as direct rather than interpolated. |

### Plotting

Plotting parameters change figures only and never change scientific
selections.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `plotting.enabled` | `true` | Create the standard figures. |
| `plotting.dpi` | `160` | Positive raster resolution used when saving figures. |
| `plotting.projection` | `SouthPolarStereo` | Geographic map projection; supported values are `SouthPolarStereo` and `PlateCarree`. Ignored in Cartesian mode. |
| `plotting.central_longitude` | `0.0` | Central longitude of the polar stereographic projection; geographic only. |
| `plotting.circular_boundary` | `true` | Clip South Polar Stereographic axes to a circular boundary; geographic only. |
| `plotting.draw_coastlines` | `true` | Draw Cartopy coastlines; geographic only. |
| `plotting.vector_stride_cells` | `5` | Plot one transport/directional arrow every this many grid cells in both dimensions. |
| `plotting.vector_reference` | `5.0` | Reference-arrow magnitude in the configured length/time rate unit. |
| `plotting.directional_vector_reference` | `0.5` | Dimensionless reference-arrow magnitude for directional-vector maps; must be in `(0, 1]`. |
| `plotting.structure_map_max_percentile` | `100.0` | Percentile used only for the upper colour limit in the structure and optional validation maps. |
| `plotting.debug_plots` | `false` | Create the optional gradient-validation map. Requires `run_validation: true`. |

### Top-level controls

| Parameter | Default | Meaning |
| --- | --- | --- |
| `write_debug_outputs` | `false` | Write candidate drops, raw sections, composites, component details, and graph-edge tables in addition to production products. |
| `run_validation` | `false` | Run and write the independent transport-gradient comparison. It does not alter front detection. |
| `analysis_version` | `4.0.0-production` | Version label written to the manifest. Normally this should track the software realization rather than be used as a tuning parameter. |

### Practical tuning order

For a new dataset:

1. Set the geometry mode, input path, timestep/unit, and every grid value to
   match matrix generation exactly.
2. Inspect the moving-count distribution and choose
   `statistics.min_moving_support` before tuning structure thresholds.
3. Tune transport coverage primarily with
   `branches.transport_percentile`; compare `raw` and `smoothed` only if
   smoothing is scientifically justified.
4. Tune directional coverage with `minimum_P_move`, `minimum_R1`, and
   `minimum_strength`, then tune local continuity with the two angular
   tolerances and `minimum_component_cells`.
5. Adjust front width and persistence only after core/corridor membership is
   satisfactory. Use debug outputs to inspect profiles and rejected candidate
   drops before changing these settings.
6. Treat parameters named `diagnostic_*` as quality-flag thresholds, not
   selection controls.

## Run

From the repository root:

```powershell
python -m research.transition_branches.workflow `
  --config research/transition_branches/configs/southern_ocean_argo.yaml
```

Cartesian runs use the same command with a Cartesian YAML. Geographic figures use
Cartopy. Cartesian figures use ordinary equal-aspect Matplotlib x/y axes whose
labels include `geometry.length_unit`; Cartopy and a CRS are not involved.

### Output naming and units

Coordinate columns follow the selected mode: geographic tables use `lon`/`lat`
names and Cartesian tables use `x`/`y` names. Physical column suffixes are
generated from the configured units:

- length: `_km`, `_cm`, `_m`, or `_mm`;
- rate: for example `_km_day` or `_cm_s`;
- length-integrated transport: for example `_km2_day` or `_cm2_s`;
- rate gradient: for example `_km_day_per_km` or `_cm_s_per_cm`.

With `coordinate_system: geographic`, `length_unit: km`, and `time_unit: day`,
the established geographic filenames and kilometre/day columns are preserved.
The resolved configuration and manifest record the coordinate system,
coordinate unit, length unit, time unit, rate unit, bearing convention, and
geometry backend.

## Production outputs

The existing products retain their original meanings:

- `cell_statistics.parquet` (extended with fundamental directional fields);
- `branch_cores.parquet`;
- `fronts.parquet`;
- `resolved_config.yaml`;
- `manifest.json`.

The independent directional/comparison products are:

- `directional_corridors.parquet`;
- `directional_fronts.parquet`;
- `structure_comparison.parquet`;
- `structure_component_comparison.parquet`.

The production figures are:

- `figures/01_transport_vectors.png`;
- `figures/02_R1.png`;
- `figures/03_R2.png`;
- `figures/04_angular_entropy.png`;
- `figures/05_cores_and_fronts.png`;
- `figures/06_directional_vectors.png`;
- `figures/07_directional_corridors_and_fronts.png`.

Figures 01 and 06 use quantitative arrow keys. In Figure 06, arrow direction is
`theta1_out`, arrow length is `P_move * R1_out`, and the key explicitly states
that the dimensionless arrows are not velocity. Figure 07 contains only results
from the directional pathway.

`plotting.structure_map_max_percentile` affects only the colorbar maximum in
Figure 05 and the optional transport-gradient validation map; it never changes
scientific selections. Directional backgrounds are naturally bounded in [0, 1].

## Optional development products

Set `write_debug_outputs: true` for transport and directional candidate-drop,
raw-section, composite, component, and graph-edge tables. Set
`run_validation: true` to write `gradient_validation.parquet`; this optional
validation evaluates already-selected transport fronts and never moves them.
With validation enabled, `plotting.debug_plots: true` also creates
`figures/debug_gradient_validation.png`.
