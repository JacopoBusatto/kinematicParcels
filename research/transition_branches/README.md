# Lagrangian current cores and transport fronts

This package turns a normalized sparse transition matrix into three scientific
products:

```text
transition matrix -> Lagrangian transport -> current cores -> probable fronts
```

It does not identify named currents or classify bifurcations, merging, or other
topology. The results remain geographically and dynamically agnostic.

## Scientific contract

For every populated source cell, supplied transition probabilities must sum to
one within numerical tolerance. `transition_count` is the sampling support and
the code never renormalizes missing probability mass.

Stay and moving transitions remain distinct. Directions condition on moving
transitions, while the physical first moment includes the full population:

```text
U_out_all = P_move * U_out_move
S = |U_out_all|
```

Distances, displacement vectors, and cross-stream samples use WGS84 geodesics.
Unsupported quantities remain `NaN`; missing support is never represented as
zero.

Current cores are q90 local transverse ridges of raw `S` at
`N_out_move >= 10`. The boundary-aware test preserves `two_sided`, `one_sided`,
and `not_evaluable` observability semantics. Connectivity establishes coherent
components and the segment context needed by short along-core compositing; graph
degree is not interpreted as physical topology.

Probable fronts are persistent outward losses of the central-tangent transport
component. The tangent remains fixed across each section. Candidate drops and
segment context remain internal, and the standard front table explicitly
distinguishes an unobservable side from an observable side without a selected
front.

## Modules

- `statistics.py`: normalized-matrix validation, outgoing transport/circular
  statistics, and retained incoming directions for future topology analysis.
- `cores.py`: boundary-aware current-core and component detection.
- `fronts.py`: fixed-tangent cross-stream sampling and canonical front outcomes.
- `geometry.py`: angle, supported-interpolation, smoothing, and grid helpers.
- `validation.py`: optional independent cross-stream-gradient validation.
- `plotting.py`: five standard maps and an optional validation map.
- `io.py`: the three tables plus reproducibility and optional debug files.
- `config.py`: one production realization per configuration.
- `workflow.py`: the single command-line production path.
- `_statistics_kernel.py`, `_ridge_kernel.py`, and `_edge_kernel.py`: verified
  low-level numerical calculations whose detail is required to preserve the
  validated method and exact front locations.

## Run

From the repository root:

```powershell
python -m research.transition_branches.workflow `
  --config research/transition_branches/configs/southern_ocean_compact.yaml
```

## Standard outputs

Every normal run contains only these machine-readable products:

- `cell_statistics.parquet`
- `branch_cores.parquet`
- `fronts.parquet`
- `resolved_config.yaml`
- `manifest.json`

The standard figures are:

- `figures/01_transport_vectors.png`
- `figures/02_R1.png`
- `figures/03_R2.png`
- `figures/04_angular_entropy.png`
- `figures/05_cores_and_fronts.png`

The vector maps include a quantitative arrow key. Every core/front marker in
the final map has a legend label describing its scientific meaning.

`plotting.structure_map_max_percentile` controls the colorbar maximum for
Figure 5 and the optional gradient-validation map. Its default is `100.0`
(the full observed maximum); lower values such as `99.0` or `95.0` reduce the
influence of extreme values without changing any calculated structure.

## Optional development products

Set `write_debug_outputs: true` to additionally write:

- `candidate_drop_zones.parquet`
- `raw_cross_sections.parquet`
- `section_composites.parquet`
- `component_graph_details.parquet`
- `segment_front_candidates.parquet`

Set `run_validation: true` to write `gradient_validation.parquet`. It evaluates
the already-selected fronts and never moves or redefines them. With validation
enabled, `plotting.debug_plots: true` also creates the optional labeled
`figures/debug_gradient_validation.png`.

## Regression

Compare a production run with the validated compact reference outside either
scientific run directory:

```powershell
python -m research.transition_branches.regression `
  --reference-run <validated-compact-run> `
  --production-run <production-run>
```

The comparison covers cell and incoming statistics, core classifications,
internal candidate-drop selection, front coordinates, distances, transport
losses, and missing-value patterns.
