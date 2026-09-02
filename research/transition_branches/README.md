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
or other topology. The products remain geographically and dynamically agnostic.

## Scientific contract

For every populated source cell, supplied transition probabilities must sum to
one within numerical tolerance. `transition_count` is the sampling support and
the code never renormalizes missing probability mass. Stay and moving
transitions remain distinct. Unsupported quantities remain `NaN`; missing
support is never represented as zero or interpreted as a front.

All distances, displacement vectors, component lengths, and cross-stream
samples use WGS84 geodesics. Every selection uses the configured
`statistics.min_moving_support`. The current ARGO and drifter production YAMLs
set that threshold to 2 and 3 respectively; the dataclass default is 10.

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

The validated production method is unchanged. Transport cores are local
transverse ridges of raw `|U_out_all|` above the configured global transport
percentile (q90 in both production YAMLs). The ridge test preserves
`two_sided`, `one_sided`, and `not_evaluable` boundary/support semantics.
Eight-neighbour connectivity establishes neutral components and graph segments;
graph degree is not interpreted as physical topology.

Probable transport fronts are persistent outward losses of the transport vector
projected onto the central transport tangent. The tangent remains fixed within
one transverse section and may change between sections. The standard front
table distinguishes an unobservable side from an observable side without a
retained front.

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
direction, independently of displacement length?** `D_out_all_east`,
`D_out_all_north`, and `D_out_all_magnitude` are dimensionless. A zero moving
probability gives a zero full-population vector but no artificial moving bearing;
a vanishing first harmonic likewise has no artificial direction.

### Directional corridors

Corridors use absolute, bounded, scientifically interpretable thresholds rather
than a global percentile. A cell must have:

- configured moving support;
- `P_move >= directional.minimum_P_move`;
- `R1_out >= directional.minimum_R1`;
- `P_move * R1_out >= directional.minimum_strength`;
- a defined `theta1_out`.

Eligible neighbouring cells connect only when their directions agree under a
circular angular difference and their geodesic connection is approximately
along the local direction at both endpoints. Components shorter than
`directional.minimum_component_cells` are discarded. Direction is evaluated
locally: a corridor may turn through 360/0 degrees and may bend substantially
over long distances as long as neighbouring changes remain compatible. No
fixed component-wide compass bearing is imposed.

There is no transverse-ridge or width requirement. A one-cell-wide or
two-cell-wide coherent band is valid; its physical width is resolution-limited
by the grid. Transverse observability is evaluated separately, so such a narrow
corridor may have two observable flanks. A coastal corridor may have only its
offshore side observable. A cell with neither supported flank can remain a
directional-corridor cell while both possible fronts are explicitly
`side_not_observable`.

### Directional fronts

Each corridor cell defines a WGS84 transverse section using its own central
directional tangent. Within that section the tangent is fixed and

```text
D_parallel(s) = D_out_all(s) dot t_hat_0
              = P_move(s) R1(s) cos(theta1(s) - theta1_0).
```

The tangent rotates from section to section as a corridor curves. `D_parallel`
decreases when directional organization or moving probability weakens, when
nearby flow turns away, or when it reverses; opposing flow may make it negative.

The front calculation reuses the validated geodesic sampling, supported
bilinear interpolation, axis refinement, contiguous smoothing, outward-drop,
and persistence concepts. Persistence is composited only across locally
connected corridor graph neighbours. Missing samples break a profile run and
can never form a candidate drop. Scientist-facing statuses are:

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

## Run

From the repository root:

```powershell
python -m research.transition_branches.workflow `
  --config research/transition_branches/configs/southern_ocean_argo.yaml
```

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

## Regression

The legacy regression command remains available for a validated compact
reference:

```powershell
python -m research.transition_branches.regression `
  --reference-run <validated-compact-run> `
  --production-run <production-run>
```

Transport regressions cover existing cell/circular statistics, core
classifications, internal candidate-drop selection, front coordinates,
distances, losses, and missing-value patterns. Directional tests additionally
verify `D_out_all = P_move * D_out_move`, `|D_out_move| = R1_out`,
`|D_out_all| = P_move * R1_out`, bearing agreement with `theta1_out`, curved
local connectivity, narrow corridors, and unobservable-side semantics.
