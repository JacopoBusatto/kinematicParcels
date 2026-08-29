# Codex task: Lagrangian transport branches and finite-time barriers from sparse transition matrices

## Scientific objective

Implement a research-specific analysis that uses the full gridded transition matrix S_ij to:

1. diagnose local outgoing transport modes;
2. connect those modes into a directed network of Lagrangian transport branches;
3. use each branch segment only as a local coordinate system;
4. scan cross-stream from the branch to determine where finite-time side-changing probability is locally minimized;
5. connect robust permeability minima into possibly disconnected Lagrangian barrier segments;
6. preserve enough intermediate information to audit every scientific decision.

The scientific focus is **front/barrier permeability**, not diffusivity.

Do NOT implement pair dispersion, relative dispersion, diffusivity tensors, or Roach-style asymptotic diffusivity in this work. Those belong to a separate study.

The method must be generic to any spatial S_ij matrix. The Southern Ocean / ACC is the primary application, but the core algorithm must not hard-code ACC geography.

## Repository design

Repository:
`JacopoBusatto/kinematicParcels`

The generic gridded transition matrix already exists in the normal postprocessing framework. Do NOT add this analysis as another generic postprocessing method.

Create a research-specific directory, e.g.

```text
research/lagrangian_barriers/
    README.md
    common.py
    validate_transition_matrix.py
    compute_transition_geometry.py
    detect_transport_modes.py
    build_branch_network.py
    diagnose_cross_branch_permeability.py
    connect_barrier_segments.py
    plot_lagrangian_barriers.py
    run_lagrangian_barrier_analysis.py
    configs/
        southern_ocean_argo.yaml
        southern_ocean_drifters.yaml
    tests/
        test_synthetic_transition_geometry.py
        test_synthetic_branches.py
        test_synthetic_barriers.py
```

A different clean split is acceptable, but keep the scientific stages modular. A single top-level driver should be able to run the complete workflow.

Do not modify the reusable transition-matrix algorithm except if a real bug is found. If a bug is found, stop and document it clearly before changing generic postprocessing behavior.

## Primary input

The immediate Argo input is a sparse transition table like:

```python
Index([
    'start_lon_bin',
    'start_lat_bin',
    'end_lon_bin',
    'end_lat_bin',
    'start_lon_center',
    'start_lat_center',
    'end_lon_center',
    'end_lat_center',
    'transition_count',
    'transition_probability',
], dtype='str')
```

Example path:

```text
F:/PLATFORMS/ARGO/postprocessing/SO_z0850_1150/
gridded_transition_matrix_dt_10d_table.parquet
```

The grid is regular 1 degree x 1 degree.

When available, also read the companion transition NetCDF for grid metadata, `n_segments_start`, directional summaries, timestep metadata, and provenance. Do not reconstruct trajectories.

For Argo, 10 days is the native time step of the trajectory positions used here. Each row of S_ij therefore describes the empirical endpoint probability one native Argo trajectory step later.

For drifters, the native trajectory sampling is different. The same algorithm must accept any transition timestep from metadata/configuration. A 10-day drifter matrix can later be used for a matched surface/depth comparison.

## Fundamental normalization

Every retained segment has exactly one valid start and one valid endpoint assigned to the transition grid.

For every populated start cell i:

```math
sum_j S_ij = 1
```

within floating-point tolerance.

Do NOT silently renormalize rows that violate this.

Output the violation and fail by default if the discrepancy exceeds a configurable tolerance.

Also verify:

```math
N_i = sum_j C_ij
S_ij = C_ij / N_i
```

within numerical tolerance.

## Existing directional summaries

The existing N/E/S/W/stay maps use four 90-degree sectors separated by the physical `dy = +/- dx` lines. Exact diagonal boundary cases are split 50/50 between adjacent sectors so that:

```math
P_N + P_E + P_S + P_W + P_stay = 1
```

These products are useful diagnostics but MUST NOT be the basis of branch detection.

The full S_ij is the basis of branch detection.

---

# Stage 0: reproducibility and preflight validation

Before doing science:

- resolve all input paths;
- calculate input file size, modification time, and preferably SHA256;
- record git commit hash and whether the working tree is dirty;
- record Python and package versions;
- record all CLI/config parameters;
- record grid spacing, grid bounds, timestep, number of sparse transitions, total transition count, populated start cells;
- validate required columns and dtypes;
- detect duplicated `(start_lon_bin,start_lat_bin,end_lon_bin,end_lat_bin)` rows;
- validate probability and count ranges;
- validate row normalization;
- validate that grid-center coordinates agree with bin indices/grid metadata;
- handle longitude periodically.

Write all preflight results to machine-readable and human-readable outputs.

---

# Stage 1: physical transition geometry

For each sparse transition i -> j, calculate a dateline-safe physical displacement:

```math
Delta r_ij = (Delta x_ij, Delta y_ij)
```

in km.

Use geodesic inverse calculations if practical, or a well-tested local tangent-plane approximation. Raw lon/lat degree differences are not acceptable for geometry.

Save for every transition:

- `dx_km`
- `dy_km`
- `distance_km`
- `bearing_deg`
- `is_stay`

Keep the original S_ij and C_ij unchanged.

For each start cell calculate:

```math
P_stay = S_ii
P_move = 1 - P_stay
```

and, for `j != i`:

```math
S_move_ij = S_ij / P_move
```

if `P_move > 0`.

---

# Stage 2: Chamberlain-style diagnostic moments

These are diagnostics/QC, not the branch definition.

For moving transitions from each cell calculate:

```math
mu_i = sum_j S_move_ij * Delta r_ij
```

and the covariance:

```math
Sigma_i =
sum_j S_move_ij
(Delta r_ij - mu_i)(Delta r_ij - mu_i)^T
```

Save:

- mean `dx`, `dy`
- mean displacement magnitude
- mean bearing
- mean transition distance
- covariance components
- major/minor eigenvalues
- variance-ellipse orientation
- ellipse axis scales
- anisotropy/eccentricity diagnostic
- angular entropy / circular concentration if useful
- `P_stay`, `P_move`
- `N_i`
- number of nonzero destination cells

Make Chamberlain-style maps of:
- mean transition vectors;
- variance ellipses;
- support;
- stay/move probability.

Do not call these diffusivity or dispersion tensors. Use terms such as `transition covariance`, `conditional displacement covariance`, and `transition variance ellipse`.

---

# Stage 3: local outgoing transport modes

The branch detector must preserve multimodality.

For each start cell i:

1. exclude stay transitions from the angular mode calculation;
2. use the conditional moving probabilities `S_move_ij`;
3. construct a circular angular probability density from transition bearings;
4. use periodic angular bins + circular smoothing or a wrapped kernel;
5. detect all statistically/significantly supported peaks;
6. do NOT force one mode per cell.

All scientific parameters must be configurable:
- angular bin count / angular resolution;
- circular smoothing bandwidth;
- minimum start count;
- minimum modal probability mass;
- minimum peak prominence;
- minimum angular separation;
- optional minimum mean displacement distance.

Assign outgoing sparse transitions to detected modes in an explicit, reproducible way, e.g. by circular angular basin or nearest valid peak.

For every `(start_cell, mode_id)` save:

- start cell indices and center;
- mode ID;
- modal probability mass among moving transitions;
- modal probability mass relative to all transitions;
- transition count represented by the mode;
- peak bearing;
- probability-weighted modal vector `(dx,dy)`;
- modal mean distance;
- angular standard deviation / width;
- peak prominence;
- number of member endpoint cells;
- number of member transition records;
- support / quality flags.

Also save a mode-membership table that maps each sparse transition row to:
- `start_cell`
- `end_cell`
- `mode_id`
- original `S_ij`
- original `C_ij`
- bearing/distance
- assignment diagnostics.

Unassigned transitions must remain visible as unassigned, not silently discarded.

---

# Stage 4: directed transport-branch graph

Represent every local mode `(i,k)` as a directed graph node.

Candidate downstream graph edges must be grounded primarily in the original S_ij links belonging to the source mode.

For a source mode `(i,k)` and endpoint cell `j` represented in that mode:
- examine the modes detected in cell `j`;
- identify downstream mode(s) compatible with continuation;
- calculate angular alignment;
- preserve genuine splitting/merging;
- do not force one outgoing graph edge.

Every graph edge must record its score components separately. At minimum:

- source node;
- target node;
- total S_ij support for the source-to-target region;
- transition count support;
- angular mismatch;
- spatial gap/distance;
- final edge score;
- every threshold/flag used to accept/reject the edge.

Do not hide the score in one opaque number.

The graph must allow:
- disconnected components;
- branch endings;
- split points;
- merge points;
- cycles if supported.

Do NOT impose one branch per longitude.
Do NOT force circumpolar closure.
Do NOT bridge unsupported gaps.

After graph pruning, decompose the graph into branch segments between junctions/gaps.

Save the full graph before and after pruning.

For the Southern Ocean application, ACC classification must be a separate post-detection annotation. The branch graph itself must not use AVISO/SEANOE fronts.

Optionally allow a user-provided geographic seed region, e.g. Drake Passage, to identify the connected branch family associated with the ACC after the graph has been built. Keep this selection fully documented.

---

# Stage 5: branch-segment geometry

For every branch segment:

- construct an ordered metric-space polyline;
- preserve raw node coordinates;
- optionally produce a gently smoothed version;
- never smooth across a graph gap or junction;
- record smoothing parameters.

At regularly spaced points along the segment calculate:

- along-branch coordinate `s_km`;
- lon/lat;
- tangent unit vector;
- normal unit vector;
- local bearing;
- curvature;
- radius of curvature when defined;
- distance to nearest other branch segment;
- branch-support diagnostics.

The normal orientation must be deterministic so that positive/negative cross-stream side does not flip randomly between neighboring points.

Do NOT construct a global parallel offset curve. Large offsets of a curved branch can self-intersect or form loops.

---

# Stage 6: local cross-stream permeability scan

At each sufficiently supported branch location q:

1. construct a local tangent/normal coordinate frame;
2. scan candidate positions along the local normal:

```math
x_q(d) = Gamma(s_q) + d * n_hat(s_q)
```

3. use a configurable offset range and spacing in km;
4. restrict the effective search range if required by:
   - local radius of curvature;
   - nearby branch segments;
   - branch self-proximity;
   - domain boundaries;
   - insufficient transition support.

Record the requested and effective search range plus every restriction flag.

For each candidate offset `d`, define a short local tangent line through `x_q(d)`.

For each nearby source cell, calculate:
- signed normal coordinate relative to that candidate line;
- along-line coordinate.

Only use source cells within a configurable local source window:
- maximum absolute normal distance from candidate line;
- maximum absolute along-line distance from q.

Use counts, not equal-weight averages of cell probabilities.

For source cells on the negative side:

```math
P_minus_to_plus =
sum cross counts from negative-side sources to positive-side endpoints
/
sum all counts from negative-side sources
```

Likewise:

```math
P_plus_to_minus
```

Also calculate a count-weighted bidirectional value:

```math
P_cross =
(C_minus_to_plus + C_plus_to_minus)
/
(N_minus + N_plus)
```

Keep both directions as primary outputs.

Also calculate conditional-on-moving versions by removing stay counts from the source denominators.

The result measures **net side-changing endpoint probability over Delta t**.

Do not claim that the unresolved trajectory crossed the line exactly once. A segment could cross and return between endpoints.

For every `(branch_id, branch_point_id, offset)` retain:

- candidate lon/lat;
- `s_km`;
- `offset_km`;
- local tangent/normal;
- source-window geometry;
- number of source cells on each side;
- total source transition counts on each side;
- stay counts on each side;
- moving counts on each side;
- crossing counts `- -> +` and `+ -> -`;
- non-crossing counts;
- P_minus_to_plus;
- P_plus_to_minus;
- P_cross;
- moving-conditional versions;
- directional asymmetry;
- count-based confidence intervals;
- support flags;
- geometry flags;
- effective search range.

---

# Stage 7: local barrier candidates

For each branch point q, analyze the complete profile:

```math
P_cross(q,d)
```

Do not assume that the barrier is the branch centerline.

Detect zero, one, or several significant local minima.

Possible outcomes must include:
- minimum at branch core;
- one flank barrier;
- two flank barriers;
- several barriers in a multi-jet corridor;
- no robust barrier.

For every minimum save:

- branch ID;
- branch point ID;
- `s_km`;
- offset `d_B`;
- lon/lat;
- P_cross;
- both directional probabilities;
- moving-conditional probabilities;
- asymmetry;
- local minimum prominence;
- local minimum width;
- neighboring maximum/reference permeability used for prominence;
- total support counts;
- confidence interval;
- barrier-candidate quality score or explicit quality flags.

Do not automatically accept every mathematical minimum.

Minimum support, prominence, edge-of-search-range behavior, geometry flags, and confidence interval must be considered.

Save rejected minima too, with rejection reasons, so the analysis is auditable.

---

# Stage 8: connect barrier candidates into barrier segments

Connect compatible local minima between neighboring branch cross-sections.

Connection should consider:
- same branch family;
- cross-stream offset continuity;
- physical distance;
- side/sign relative to branch;
- permeability similarity if useful;
- no unsupported large gaps.

Do not force continuity.

A disappearing minimum produces a barrier gap.

Barrier segments may be detached.
There may be multiple barrier segments associated with one transport corridor.

For each final barrier segment save:
- barrier ID;
- parent branch / branch-family ID;
- ordered points;
- lon/lat;
- along-barrier distance;
- offset from reference branch;
- permeability and directional permeability;
- support;
- quality flags;
- segment length;
- coverage fraction;
- number/length of gaps;
- mean/median/min/max permeability;
- directional asymmetry statistics.

Export barrier and branch geometry to both tabular and geospatial formats.

---

# Stage 9: Southern Ocean interpretation / validation

After branch and barrier detection only:

- optionally annotate the branch family connected to an ACC seed region;
- overlay independent AVISO / SEANOE frontal products;
- calculate distances between external fronts and detected branches/barriers;
- do not use external fronts to change the detected solution.

Later, run the exact same analysis independently for:
- Argo parking-depth transition matrix;
- surface drifter transition matrix.

For direct quantitative surface/depth comparison, use matched Delta t when desired.

---

# Stage 10: hydrographic handoff

Do NOT implement the full hydrographic analysis in this task unless explicitly requested later.

But produce barrier geometry in a form that can be used to assign historical Argo profiles:
- signed cross-barrier distance;
- nearest barrier segment;
- nearest branch segment;
- local tangent/normal;
- along-barrier coordinate.

The later hydrographic study will examine Conservative Temperature, Absolute Salinity, density, spiciness, and possibly fine-scale interleaving diagnostics.

---

# Required output structure

Create a run-specific output directory, for example:

```text
<output_root>/
  run_manifest.json
  run_config_resolved.yaml
  run_summary.txt
  validation/
  transition_geometry/
  modes/
  graph/
  branches/
  permeability/
  barriers/
  figures/
  logs/
```

Never overwrite an existing run unless `--overwrite` is explicitly provided.

## A. Provenance / reproducibility

### `run_manifest.json`
Include:
- analysis name and version;
- UTC timestamp;
- git commit;
- git dirty/clean status;
- Python version;
- package versions;
- full input paths;
- file sizes;
- mtimes;
- SHA256 if practical;
- transition timestep;
- grid metadata;
- all resolved CLI/config parameters;
- random seed, if any;
- output file inventory.

### `run_config_resolved.yaml`
Exact fully resolved configuration used.

### `run_summary.txt`
Human-readable summary of every major count, warning, threshold, and output path.

## B. Validation

### `validation/row_normalization.parquet`
One row per start cell:
- N_i
- sum C_ij
- sum S_ij
- normalization residual
- n destinations
- P_stay
- validation flags.

### `validation/validation_summary.json`
Global minima/maxima/quantiles and failure counts.

### `validation/duplicate_transitions.parquet`
Only if duplicates are found.

## C. Transition geometry

### `transition_geometry/transition_geometry.parquet`
Original sparse rows plus:
- dx_km
- dy_km
- distance_km
- bearing_deg
- is_stay
- conditional moving probability.

### `transition_geometry/cell_diagnostics.nc`
Regular lat/lon fields:
- N_i
- n destinations
- P_stay
- P_move
- mean dx/dy
- mean distance
- mean bearing
- covariance elements
- ellipse major/minor values
- ellipse angle
- anisotropy
- angular entropy/concentration
- number of detected modes
- dominant-mode probability.

## D. Modes

### `modes/modes.parquet`
One row per local mode with all quantities described above.

### `modes/mode_membership.parquet`
One row per sparse transition with mode assignment and assignment diagnostics.

### `modes/rejected_modes.parquet`
Rejected candidates and reasons.

## E. Branch graph

### `graph/mode_nodes.parquet`
All graph nodes.

### `graph/mode_edges_all.parquet`
All candidate edges, including rejected edges and score components.

### `graph/mode_edges_selected.parquet`
Selected graph edges.

### `graph/branch_graph.graphml`
GraphML export if possible.

### `branches/branch_points.parquet`
One row per ordered branch point:
- branch ID
- component ID
- node ID
- point order
- lon/lat
- metric x/y
- raw/smoothed coordinates
- tangent
- normal
- curvature
- radius of curvature
- nearest-branch distance
- local support
- junction/gap flags.

### `branches/branch_summary.csv`
Per-branch statistics.

### `branches/branches.geojson`
Geospatial branch segments.

## F. Permeability

### `permeability/cross_sections.parquet`
One row per `(branch_point, offset)` with every count/probability/geometry field.

### `permeability/cross_sections.nc`
When practical, a gridded/ragged representation for plotting.

### `permeability/source_contributions.parquet`
Optional but strongly preferred under a flag such as `--save-contributions`.
For every `(branch_point, offset, source_cell)` save denominator and crossing contributions.
This is important for auditing why a permeability value is high or low.

### `permeability/permeability_summary.csv`
Per branch and/or along-branch aggregated statistics.

## G. Barrier candidates and final barriers

### `barriers/barrier_candidates_all.parquet`
All mathematical minima, accepted and rejected, with reasons.

### `barriers/barrier_candidates_selected.parquet`
Accepted local minima.

### `barriers/barrier_points.parquet`
Ordered final barrier points.

### `barriers/barrier_segments.geojson`
Final possibly disconnected barrier segments.

### `barriers/barrier_summary.csv`
Per-barrier statistics.

## H. Figures

Produce many diagnostic figures. At minimum:

1. transition support map;
2. stay/move probability;
3. mean transition vectors;
4. variance ellipses;
5. number of local modes per cell;
6. dominant modal probability;
7. local mode vectors;
8. all mode-graph edges vs selected graph edges;
9. branch network map with branch IDs;
10. branch support/quality map;
11. branch curvature map;
12. example outgoing PDFs for automatically selected unimodal and multimodal cells;
13. P_cross(s,d) heatmap for each major branch;
14. example local cross-section P_cross(d) curves;
15. accepted/rejected barrier minima on the cross-section curves;
16. map of final barrier points and segments over branch network;
17. barrier permeability along distance s;
18. directional permeability `P_-to+` and `P_+to-` along each barrier;
19. moving-conditional permeability;
20. barrier directional asymmetry;
21. observational support along barriers;
22. geometry/quality flags;
23. optional overlay with external ACC fronts for validation only.

Every figure must have a corresponding machine-readable data product.

## I. Logging

Use structured logging.

Save:
- INFO log;
- warnings log;
- validation failures;
- number of cells/modes/edges/branches/cross-sections/minima at every stage;
- reasons for each class of rejection.

Never silently discard a cell, mode, edge, branch point, cross-section, or barrier minimum.

---

# Synthetic tests

Before relying on Southern Ocean results, create synthetic sparse transition matrices with known behavior.

At minimum test:

1. **Straight zonal corridor, no barrier**
   - branch detector finds one zonal branch;
   - P_cross does not show a robust local minimum.

2. **Straight zonal corridor, core barrier**
   - branch detector finds centerline;
   - permeability minimum occurs at branch center.

3. **Straight zonal corridor, two flank barriers**
   - one transport branch/corridor;
   - two P_cross minima appear north/south of core.

4. **Bifurcating branch**
   - one upstream mode splits into two supported modes;
   - graph preserves both branches.

5. **Merging branches**
   - graph preserves merge topology.

6. **Detached branch / sampling gap**
   - no artificial connection across unsupported gap.

7. **Strongly curved branch**
   - local cross-sections work;
   - no global-offset loop/self-intersection is created.

8. **Dateline-crossing branch**
   - physical displacement and graph remain continuous.

9. **Stay-dominated cells**
   - distinguish low mobility from low cross-stream permeability using moving-conditional diagnostics.

10. **Directional barrier**
    - `P_-to+` differs from `P_+to-` and the difference is recovered.

11. **Multimodal endpoint PDF whose mean lies between branches**
    - mean-vector diagnostic is misleading;
    - mode detector correctly identifies both pathways.

The synthetic tests should produce small diagnostic figures as well as assertions.

---

# Scientific guardrails

- Full S_ij defines the branches; N/E/S/W maps do not.
- A branch is not automatically a barrier.
- The branch gives the local tangent/normal coordinate frame.
- Barriers are local minima of finite-time net side-changing probability across the branch neighborhood.
- The number of barriers is an output, not an assumption.
- Barriers may be detached/discontinuous.
- Branches may split, merge, terminate, or be disconnected.
- Do not globally offset curved branches.
- Use local cross-stream sections.
- Use physical distances, not raw degree distances.
- Keep directional crossing probabilities separately.
- Use raw counts for aggregation and preserve support.
- Do not call 10-day permeability a diffusivity.
- Do not add pair dispersion in this study.
- Do not use AVISO/SEANOE fronts in branch or barrier scores.
- Preserve rejected candidates and rejection reasons.
- Prefer explicit, auditable intermediate files over clever opaque abstractions.

---

# First real-data execution

After tests pass, run first on the Argo matrix:

```text
F:/PLATFORMS/ARGO/postprocessing/SO_z0850_1150/
gridded_transition_matrix_dt_10d_table.parquet
```

with the companion NetCDF if available.

Do not tune parameters to make the result resemble known ACC fronts.

Produce the full diagnostics first. We will inspect:
- whether branches emerge;
- where they split/merge;
- where they are unresolved;
- the P_cross(s,d) structure;
- whether barrier minima occur at branch cores, flanks, or not at all;
- how results behave at Agulhas, Kerguelen, Campbell Plateau, Drake Passage, and other complex sectors.

Only after seeing these outputs should thresholds or ACC-specific classification choices be refined.
