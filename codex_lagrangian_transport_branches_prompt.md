# Codex task: robust coherent Lagrangian transport branches and branch-relative candidate fronts from gridded transition matrices

## Scientific motivation

We already implemented a first research workflow under:

```text
research/lagrangian_barriers/
```

that detects local angular modes, builds a mode graph, decomposes it into branch segments, and scans cross-stream permeability.

**Do not modify or replace that implementation. Keep it frozen as the first methodological experiment.**

The first experiment showed that the full local-mode graph is substantially more complicated than necessary for the immediate scientific objective. In particular, the mean transition-vector field reveals the major Southern Ocean transport pathways much more clearly than the resulting graph network.

We therefore want a **second, deliberately simpler and independent analysis**.

The primary scientific objective is:

> identify robust, spatially persistent Lagrangian transport branches from the transition matrix, and determine where the transport aligned with those branches weakens rapidly across-stream.

The main scientific hierarchy is:

```text
transition matrix
    |
    +-- finite-time transport intensity
    |
    +-- directional reliability/coherence
    |
    +-- inward/outward continuity
    |
    +-- spatial directional persistence
            |
            +-- coherent transport branches
                    |
                    +-- branch-relative cross-stream weakening
                            |
                            +-- candidate current/front flanks
                                    |
                                    +-- later permeability analysis
```

Topology is secondary.

Do not attempt to catalogue every local transition topology.

Do not implement a final ACC front line or final transport-barrier line during the initial staged analysis.

---

# Key methodological decisions

## 1. Analyze 30-day and 10-day transition matrices independently

The primary real-data experiment should start with:

```text
dt = 30 days
```

and the 10-day matrix should later be analyzed independently as a temporal-resolution sensitivity experiment.

The 30-day transition matrix must be generated upstream from trajectories using a **simple 30-day resampling**, producing consecutive non-overlapping 30-day displacement segments.

Do not construct the 30-day matrix as:

```text
Q30 = Q10^3
```

and do not build it from overlapping 10-to-30-day pairs.

The purpose of the 30-day resampling is to avoid strong reuse of overlapping trajectory segments and to reduce directional quantization associated with the 1-degree grid.

The 10-day matrix remains scientifically important because 10 days is the native Argo-cycle temporal resolution.

The two analyses therefore answer complementary questions:

```text
30 days:
    more spatially developed transitions;
    reduced grid-direction quantization;
    primary branch-detection experiment.

10 days:
    native Argo-cycle transport;
    more local finite-time structure;
    independent sensitivity experiment.
```

Do not tune one experiment to reproduce the other.

Use identical definitions and, wherever scientifically sensible, identical physical-unit parameters.

After both analyses exist, compare their results explicitly.

Important terminology:

The 30-day resampling produces operationally non-overlapping segments, but raw transition counts must still **not** be described as statistically independent-sample confidence estimates because repeated segments can originate from the same float.

---

# Repository guardrails

Repository:

```text
C:\Users\Jacopo\Documents\GitHub\kinematicParcels
```

Create a new lightweight analysis, separate from the previous graph experiment, for example:

```text
research/
    transition_branches/
        README.md
        transition_branch_analysis.py
        configs/
            southern_ocean_argo_30d.yaml
            southern_ocean_argo_10d.yaml
```

A similarly simple organization is acceptable.

Add focused tests, for example:

```text
tests/test_transition_branch_analysis.py
```

Do not:

- modify `research/lagrangian_barriers/`;
- modify previous baseline outputs;
- replace or alter generic transition-matrix generation behavior;
- reconstruct trajectories inside this research analysis;
- infer the 30-day matrix from the 10-day matrix;
- use external velocity products;
- use external ACC fronts in detection scores;
- tune parameters to known ACC/front positions;
- implement pair dispersion/diffusivity;
- build another global angular-mode graph;
- force every supported cell into a named topology.

The research analysis itself must consume **already generated sparse transition tables and their raw transition counts**.

If the required 30-day transition table does not exist, stop and report that the upstream 30-day matrix must first be generated with the existing trajectory-processing workflow.

---

# Primary real-data inputs

Known 10-day table:

```text
C:\Users\Jacopo\Documents\SIMULATIONS\kinematicParcels\southern_ocean\ARGO_850-1150\postprocessing\gridded_transition_matrix_dt_10d_table.parquet
```

Use the corresponding independently generated 30-day table for the primary run, expected to follow the analogous naming convention, for example:

```text
C:\Users\Jacopo\Documents\SIMULATIONS\kinematicParcels\southern_ocean\ARGO_850-1150\postprocessing\gridded_transition_matrix_dt_30d_table.parquet
```

Do not silently substitute one timestep for the other.

Expected columns:

```text
start_lon_bin
start_lat_bin
end_lon_bin
end_lat_bin
start_lon_center
start_lat_center
end_lon_center
end_lat_center
transition_count
transition_probability
```

Known spatial domain:

```text
longitude: [-180, 180]
latitude:  [-80, -30]
grid:      1 degree x 1 degree
```

The 10-day Argo matrix previously contained approximately:

```text
108,383 sparse links
13,587 populated start cells
687,432 transitions
```

Still validate every input independently.

**Never silently renormalize invalid rows.**

---

# Fundamental transition definitions

Let:

\[
C_{ij}
\]

be the observed count of transitions from source cell \(i\) to destination cell \(j\).

For each source cell:

\[
N_{\mathrm{out,total}}(i)
=
\sum_j C_{ij}.
\]

The forward transition probability is:

\[
Q_{\mathrm{out}}(j|i)
=
\frac{C_{ij}}
{N_{\mathrm{out,total}}(i)}.
\]

For each destination cell:

\[
N_{\mathrm{in,total}}(j)
=
\sum_i C_{ij},
\]

and define the empirical incoming source distribution:

\[
Q_{\mathrm{in}}(i|j)
=
\frac{C_{ij}}
{N_{\mathrm{in,total}}(j)}.
\]

Be explicit that \(Q_{\mathrm{in}}\) is an **empirical destination-conditioned incoming-flux/source distribution**, not another forward Markov transition probability.

---

# Stay and moving transitions

Same-cell transitions must remain explicit.

Define:

\[
C_{ii}
\]

and

\[
P_{\mathrm{stay}}(i)
=
\frac{C_{ii}}
{N_{\mathrm{out,total}}(i)}.
\]

Then:

\[
P_{\mathrm{move}}(i)
=
1-P_{\mathrm{stay}}(i).
\]

Also calculate:

\[
N_{\mathrm{out,move}}(i)
=
\sum_{j\neq i}C_{ij}.
\]

For moving-only directional diagnostics:

\[
\widetilde Q_{\mathrm{out}}(j|i)
=
\frac{C_{ij}}
{N_{\mathrm{out,move}}(i)},
\qquad j\neq i.
\]

Similarly for incoming transitions:

\[
N_{\mathrm{in,move}}(j)
=
\sum_{i\neq j}C_{ij},
\]

\[
\widetilde Q_{\mathrm{in}}(i|j)
=
\frac{C_{ij}}
{N_{\mathrm{in,move}}(j)},
\qquad i\neq j.
\]

## Important interpretation

Do **not** discard staying transitions from the analysis.

Instead separate three concepts:

```text
support
mobility
transport geometry
```

Use:

```text
N_total, N_move
```

for observational support,

```text
P_stay, P_move
```

for mobility,

and moving-conditioned distributions for directional geometry.

A stay transition has no physical direction and must therefore be excluded from:

```text
R1
R2
mean bearing
angular entropy
angular peaks
directional covariance/orientation diagnostics
```

Do not use the raw stay count `C_ii` as an independent scientific filter.

Retain it in outputs, but use the dimensionless quantity:

```text
P_stay
```

for physical interpretation.

---

# Physical transition geometry

Use WGS84 geodesic geometry, not raw degree differences.

`pyproj.Geod` is appropriate.

For every sparse transition save:

- geodesic distance [km];
- source-to-destination bearing;
- local eastward displacement `dx_km`;
- local northward displacement `dy_km`;
- `is_stay`;
- transition count;
- forward transition probability;
- moving-conditioned transition probability.

Handle the antimeridian correctly.

For incoming diagnostics retain two angle definitions explicitly.

## Incoming source-side direction

```text
theta_in_source
```

is the bearing from destination cell \(j\) toward source cell \(i\).

It answers:

> From which side was this destination cell supplied?

## Incoming actual-motion direction

```text
theta_in_motion
```

is the source-to-destination direction of motion.

Use this quantity when comparing incoming flow with outgoing flow.

Never silently mix these definitions.

---

# Stage 0 — validation and observational support

Run Stage 0 first on the **30-day matrix only**.

Validate:

- required columns;
- dtypes;
- finite coordinates;
- finite probabilities;
- positive integer counts;
- probability range `[0,1]`;
- unique sparse transition keys;
- grid-center consistency;
- row sums;
- `transition_probability == transition_count / N_out_total`.

Never renormalize failed rows.

Create cell-level fields for:

```text
N_out_total
N_out_move
N_in_total
N_in_move
C_stay
P_stay
P_move
number of distinct moving destinations
number of distinct moving source cells
```

## Support sensitivity

Do not choose a single final support cutoff.

Create support flags for:

```text
10
20
30
50
100
```

transitions.

Do this separately for:

```text
N_out_total
N_out_move
N_in_total
N_in_move
```

The distinction between total and moving support is essential.

For example, a cell may contain 100 transitions but only 10 moving transitions. Its total population is reasonably sampled, while its estimated angular distribution is poorly supported.

Directional diagnostics must therefore always carry the corresponding moving-support information.

### STOP GATE

After Stage 0:

- save machine-readable support fields;
- report support distributions;
- report map coverage at each threshold;
- report `P_stay` / `P_move` distributions;
- show the main support figures;
- report validation failures;
- stop.

Do not implement Stage 1 until explicitly instructed.

---

# Stage 1 — first statistical moment and finite-time transport intensity

For moving transitions define the moving-conditioned first statistical moment:

\[
\boldsymbol{\mu}_{\mathrm{out,move}}(i)
=
\sum_{j\neq i}
\widetilde Q_{\mathrm{out}}(j|i)
\Delta\mathbf r_{ij}.
\]

Convert it to finite-time mean transition velocity:

\[
\mathbf U_{\mathrm{out,move}}
=
\frac{\boldsymbol{\mu}_{\mathrm{out,move}}}
{\Delta t}.
\]

This answers:

> Among transitions that actually leave the source cell, what is their mean displacement?

Also calculate the total-population first moment, treating stay as zero displacement:

\[
\boldsymbol{\mu}_{\mathrm{out,all}}
=
\sum_j
Q_{\mathrm{out}}(j|i)
\Delta\mathbf r_{ij}.
\]

Because stay displacement is zero:

\[
\boldsymbol{\mu}_{\mathrm{out,all}}
=
P_{\mathrm{move}}
\boldsymbol{\mu}_{\mathrm{out,move}}.
\]

Therefore:

\[
\mathbf U_{\mathrm{out,all}}
=
P_{\mathrm{move}}
\mathbf U_{\mathrm{out,move}}.
\]

Export magnitudes in:

```text
km/day
m/s
```

## Interpretation

`U_out_move` describes the movement conditional on leaving the cell.

`U_out_all` describes the expected finite-time transport of the full observed starting population.

A region dominated by staying transitions therefore naturally has weak `U_out_all`, even if the minority of moving transitions travels rapidly.

For the initial branch analysis:

> **treat `|U_out_all|` as the primary candidate transport-intensity field.**

Keep `|U_out_move|` and `P_move` independently visible so that the cause of weak/strong total transport remains interpretable.

Do not combine them into additional opaque scores.

Also calculate:

\[
\bar r_{\mathrm{out}}
=
\sum_{j\neq i}
\widetilde Q_{\mathrm{out}}(j|i)
|\Delta\mathbf r_{ij}|.
\]

If straightforward, calculate count-weighted transition-distance quantiles.

### STOP GATE

Report:

- distributions of `|U_out_all|`;
- distributions of `|U_out_move|`;
- relationship with `P_move`;
- mean moving distance;
- maps with decimated vectors;
- strong transport regions with weak observational support;
- numerical warnings.

Then stop.

---

# Stage 2 — directional coherence and mean-direction reliability

The first moment alone can produce a misleading mean direction.

For example, two separated outgoing branches may average into an intermediate vector even if almost no transitions actually follow that mean direction.

Therefore calculate both the first and second circular harmonics.

## First circular harmonic

For moving outgoing bearings:

\[
M_{1,\mathrm{out}}
=
\sum_{j\neq i}
\widetilde Q_{\mathrm{out}}(j|i)
e^{i\theta_{ij}}.
\]

\[
R_{1,\mathrm{out}}
=
|M_{1,\mathrm{out}}|,
\]

\[
\theta_{1,\mathrm{out}}
=
\arg(M_{1,\mathrm{out}}).
\]

High \(R_1\) means that moving transitions predominantly follow one direction.

## Second circular harmonic

Calculate:

\[
M_{2,\mathrm{out}}
=
\sum_{j\neq i}
\widetilde Q_{\mathrm{out}}(j|i)
e^{2i\theta_{ij}},
\]

\[
R_{2,\mathrm{out}}
=
|M_{2,\mathrm{out}}|,
\]

\[
\theta_{2,\mathrm{out}}
=
\frac12\arg(M_{2,\mathrm{out}}).
\]

Treat `theta2` as an axis modulo \(180^\circ\).

## Critical interpretation of R2

Do **not** interpret high \(R_2\) as a generic branch criterion.

The purpose of \(R_2\) is to help determine whether the first-moment direction is representative of the underlying transition geometry.

Examples:

### Narrow unimodal directed flow

Typically:

```text
high R1
high R2
```

### Symmetric split around the mean direction

Example:

```text
50% at +45 deg
50% at -45 deg
```

The first moment points east and still has:

\[
R_1 \approx 0.71,
\]

even though almost no transitions move east.

For this geometry:

\[
R_2 \approx 0.
\]

Thus the apparently reasonable first-moment direction is potentially misleading.

### Opposite-direction axial organization

Example:

```text
50% east
50% west
```

gives:

\[
R_1=0,
\qquad
R_2=1.
\]

This is strongly organized along an axis but is not directed through-flow.

Therefore use:

```text
R1
R2
entropy
optional angular peaks
```

jointly as **mean-direction reliability / angular-geometry diagnostics**.

Do not define branch cells from \(R_2\) alone.

## Angular entropy

Calculate normalized Shannon entropy:

\[
H^*
=
-\frac{\sum_k p_k\log p_k}{\log K}.
\]

Use the same fixed angular bins for all runs.

Interpret entropy descriptively.

Do not impose a final entropy threshold.

## Optional angular peaks

Implement a lightweight angular-peak diagnostic if useful.

Save:

```text
n_peaks
dominant_peak_mass
second_peak_mass
angular_separation
peak_prominence
```

This is particularly useful when \(R_1/R_2\) indicate that the first-moment vector may be hiding a split or multimodal distribution.

Do not construct an angular-mode graph.

### STOP GATE

Report:

- where strong first-moment transport also has high directional coherence;
- where the first-moment vector appears potentially misleading;
- representative split/axial/multidirectional cells;
- relationship among `R1`, `R2`, entropy and transport strength.

Then stop.

---

# Stage 3 — inward coherence and through-flow continuity

Repeat the angular diagnostics using:

\[
\widetilde Q_{\mathrm{in}}.
\]

Calculate:

```text
R1_in
theta1_in_source
theta1_in_motion
R2_in
theta2_in
H_in
```

and optional incoming angular peaks.

The incoming distribution is used primarily to establish whether a candidate branch cell is coherently supplied from upstream.

Do not treat the magnitude of the destination-conditioned incoming mean vector as a transport intensity exactly equivalent to `U_out_all`.

The primary transport-intensity field remains outward.

## Incoming/outgoing continuity

Compare:

\[
\theta_{1,\mathrm{in,motion}}
\]

with:

\[
\theta_{1,\mathrm{out}}.
\]

Define:

\[
A_{io}
=
\cos
\left(
\theta_{1,\mathrm{out}}
-
\theta_{1,\mathrm{in,motion}}
\right).
\]

Retain the continuous value.

Values close to 1 indicate that the cell is fed in approximately the same direction as the flow leaving it.

This is the expected geometry of coherent through-flow.

---

# Stage 3B — spatial directional persistence

A coherent cell is not automatically a coherent branch.

A stable branch must persist spatially through neighboring cells.

Calculate neighborhood-direction consistency for supported neighboring cells.

For outgoing motion, for example:

\[
C_{\mathrm{neigh,out}}(i)
=
\frac{
\sum_{k\in\mathcal N_i}
w_k
\cos
\left(
\theta_{1,\mathrm{out}}(i)
-
\theta_{1,\mathrm{out}}(k)
\right)
}{
\sum_{k\in\mathcal N_i}w_k
}.
\]

Start with a transparent configurable neighborhood such as 8-neighbor connectivity.

Keep the weighting scheme explicit.

Initially compare:

```text
uniform weighting
support weighting
transport-intensity weighting
```

only if necessary.

Do not silently choose a complex weighting scheme.

Also calculate an analogous neighborhood consistency using:

```text
theta1_in_motion
```

where incoming support is sufficient.

The distinction is important:

```text
R1_out:
    agreement among transitions leaving one cell

C_neigh_out:
    agreement among mean directions of nearby cells
```

Both are required to characterize a spatially persistent branch.

### STOP GATE

Explicitly answer:

1. Are the strong outward transport pathways coherently fed from upstream?
2. Are incoming actual-motion and outgoing directions aligned?
3. Are their directions spatially persistent across neighboring cells?
4. Where do apparently strong first-moment vectors fail the inward or neighborhood-continuity checks?
5. Which regions remain ambiguous because of split/multimodal angular structure?

Then stop.

---

# Stage 4 — definition of a candidate stable transport branch

The central scientific target is a **stable coherent transport branch**.

Do not collapse all diagnostics into a single opaque scalar.

Conceptually, a stable branch is a spatially persistent corridor whose cells show:

1. adequate observational support;
2. strong total finite-time transport;
3. a representative outgoing direction;
4. coherent incoming feeding;
5. incoming/outgoing through-flow alignment;
6. similar transport direction among neighboring cells.

The primary diagnostics are therefore:

```text
N_out_total
N_out_move
N_in_total
N_in_move

|U_out_all|
|U_out_move|
P_move

R1_out
R2_out
H_out

R1_in
R2_in
H_in

A_io

C_neigh_out
C_neigh_in_motion
```

## R2 remains a reliability/complexity diagnostic

Do not require simply:

```text
R2 > threshold
```

for a branch.

Instead, use \(R_2\), entropy and optional peaks to identify cells where the first-moment direction may not faithfully represent the actual transition geometry.

Such cells may:

- be excluded from the branch backbone;
- be retained as uncertain/complex portions;
- represent real split/merge regions;
- connect otherwise coherent branch segments.

Do not decide this silently.

Preserve the continuous diagnostics.

## Exploratory branch field

The simplest scalar branch field should initially be:

\[
S_{\mathrm{transport}}
=
|\mathbf U_{\mathrm{out,all}}|.
\]

Optionally compare:

\[
U_{\mathrm{coh}}
=
|\mathbf U_{\mathrm{out,all}}|
R_{1,\mathrm{out}}.
\]

However:

> do not multiply every diagnostic into one master score.

In particular, avoid an opaque product such as:

```text
U * R1_out * R1_in * A_io * C_neigh
```

as the baseline definition.

Instead:

- use transport intensity to locate candidate corridors;
- use directional/inward/neighborhood diagnostics to assess whether those corridors are stable and interpretable.

### STOP GATE

Before branch extraction:

- compare the transport-intensity maps;
- show `R1`, `R2`, inward coherence, alignment and neighborhood persistence;
- identify which field provides the cleanest candidate branch backbone;
- document remaining scientific choices;
- stop.

Do not choose the branch field silently.

---

# Stage 5 — robust main branch extraction

Only proceed after Stage 4 has been reviewed.

The objective is to construct the **main spatial branch line(s)** of strong coherent transport.

Multiple significant branches are allowed.

Disconnected branches are allowed.

Do not force one circumpolar line.

Do not bridge unsupported gaps.

## Branch philosophy

Prefer a lightweight ridge/corridor method operating on the reviewed transport-intensity field.

A candidate branch should follow spatially persistent maxima of strong transport while remaining consistent with the directional diagnostics.

Do not assume a predefined ACC shape.

Do not assume the branch is zonal.

Do not assume a Gaussian or bell-shaped cross-stream profile.

Do not tune the branch to external fronts.

## Stability criteria

Branch extraction and ranking should consider, separately:

```text
transport intensity
R1_out
R1_in
A_io
C_neigh_out
C_neigh_in_motion
moving support
R2/entropy ambiguity flags
spatial persistence
```

The first-moment vector proposes the transport direction.

The angular diagnostics determine whether that direction is trustworthy.

The neighborhood diagnostics determine whether the direction forms a spatial branch.

## Branch outputs

For every candidate component calculate:

- branch ID;
- branch cells;
- branch centerline/ridge;
- number of cells;
- physical length;
- longitude span;
- latitude span;
- mean/median/max `|U_out_all|`;
- mean/median `|U_out_move|`;
- mean `P_move`;
- mean `R1_out`;
- mean `R1_in`;
- mean alignment;
- mean neighborhood coherence;
- fraction of angularly ambiguous cells;
- support statistics;
- unsupported gaps;
- continuity/persistence statistics.

Rank branches transparently.

Do not collapse all properties into one opaque significance score.

## Geometry warnings

Optionally report:

```text
high curvature
self intersection
closed loop
fragmentation
large unsupported gap
possible split
possible merge
```

These are warnings/descriptors only.

### STOP GATE

Show:

- all unfiltered candidate branches;
- retained/ranked major branches;
- discarded candidates and reasons;
- sensitivity to transport threshold;
- sensitivity to support threshold;
- sensitivity to smoothing scale;
- sensitivity to minimum branch length;
- branch-level summary table.

Then stop.

---

# Stage 6 — PRIMARY candidate front diagnostic: branch-relative loss of along-stream transport

This is now the primary candidate front-edge diagnostic.

Do not assume that the cross-stream transport profile is Gaussian, bell-shaped, symmetric, monotonic, or otherwise prescribed.

The observed profile is expected to be noisy.

The algorithm must therefore search for **robust and spatially persistent weakening of branch-aligned transport**, not fit an expected functional shape.

## Local branch frame

Let a retained branch centerline be:

\[
\mathbf B(s),
\]

where \(s\) is along-branch distance.

Estimate a local tangent:

\[
\hat{\mathbf t}_B(s)
\]

using a configurable physical smoothing/window scale.

Define the local normal:

\[
\hat{\mathbf n}_B(s)
=
(-t_y,t_x).
\]

For signed cross-stream distance \(d\):

\[
\mathbf x(s,d)
=
\mathbf B(s)
+
d\,\hat{\mathbf n}_B(s).
\]

Treat the two sides separately:

```text
d < 0
d > 0
```

Refer initially to:

```text
left branch flank
right branch flank
```

rather than globally north/south.

## Along-stream transport

At each sampled cross-stream position calculate the projection of the finite-time transport vector onto the local branch tangent:

\[
U_\parallel(s,d)
=
\mathbf U_{\mathrm{out,all}}(s,d)
\cdot
\hat{\mathbf t}_B(s).
\]

Retain the signed value.

Also save, for interpretation:

```text
|U_out_all|
|U_out_move|
P_move
R1_out
R2_out
R1_in
A_io
C_neigh_out
support
```

across each section where available.

The primary scientific question is:

> starting from the coherent transport branch, where does transport aligned with that branch rapidly and persistently weaken on either side?

## No expected cross-stream shape

Do not:

- fit a Gaussian as the baseline method;
- impose a bell shape;
- impose symmetry;
- assume equal widths on the two sides;
- require monotonic decay from the center;
- define the front using a fixed geometrical offset.

Keep the raw sampled profiles.

A robust smoother may be used only as a configurable diagnostic aid.

The raw profile must always remain available.

## Candidate sudden-drop diagnostics

Explore transparent quantities such as:

\[
\frac{\partial U_\parallel}{\partial d},
\]

finite-distance transport loss:

\[
\Delta U_\parallel
=
U_\parallel(d_{\mathrm{inner}})
-
U_\parallel(d_{\mathrm{outer}}),
\]

relative transport:

\[
U_\parallel^*(s,d)
=
\frac{
U_\parallel(s,d)
}{
U_\parallel(s,0)
},
\]

and optional half-strength/e-folding distances where the data actually support them.

Do not require all cross-sections to possess a half-maximum or e-folding point.

## Persistence requirement

Do not define a candidate edge from a single large one-cell finite difference.

A robust candidate flank should show some combination of:

- a substantial loss of along-branch transport over a finite physical cross-stream distance;
- transport remaining lower outside the drop;
- similar drop position over several neighboring along-branch sections;
- adequate observational support across the relevant cells;
- no obvious contamination by another major transport branch.

The precise persistence criterion must remain configurable and be inspected before adoption.

## Along-branch aggregation

Because individual sections are expected to be noisy, calculate local along-branch composite diagnostics over a configurable window.

For example, retain robust statistics such as:

```text
median
25th percentile
75th percentile
number of valid sections
```

for nearby cross-stream profiles.

Do not average away raw profiles.

The purpose is to distinguish persistent branch flanks from isolated grid-cell fluctuations.

## No global offset lines

Do not construct continuous geometrical curves by simply offsetting the branch centerline by a fixed distance.

Such curves can generate artificial:

- loops;
- self-intersections;
- crossings;
- pathologies in high-curvature regions.

Treat each local cross-section independently.

If candidate drop positions later form persistent spatial structures, save the points/bands for inspection.

Do not connect them automatically into final fronts in this stage.

## Cross-section quality flags

Flag sections affected by:

- insufficient support;
- excessive local branch curvature;
- branch self-proximity;
- intersection with another significant branch;
- missing cross-stream cells;
- inadequate physical resolution;
- ambiguous local branch direction.

### STOP GATE

Explicitly answer:

1. Does along-branch transport weaken clearly on either side of the main branches?
2. Are the weakening locations persistent over neighboring along-branch sections?
3. Are both flanks detectable?
4. Are flank positions symmetric or strongly asymmetric?
5. Does the transport remain weak outside the candidate drop?
6. How strongly do results depend on the smoothing/composite window?
7. Where are cross-sections too noisy or poorly sampled to define an edge?
8. Do different major branches show different cross-stream widths?

Then stop.

---

# Stage 7 — secondary global gradient diagnostic

Retain the original two-dimensional cross-stream-gradient idea as an **independent secondary diagnostic**, not the primary front detector.

Where a local mean transport direction is reliable:

\[
\hat t
=
\frac{\boldsymbol\mu_{\mathrm{out}}}
{|\boldsymbol\mu_{\mathrm{out}}|},
\]

\[
\hat n=(-t_y,t_x).
\]

For a scalar field \(S\):

\[
G_{\perp,S}
=
\hat n\cdot\nabla S,
\]

\[
G_{\parallel,S}
=
\hat t\cdot\nabla S.
\]

Calculate physical gradients using latitude-dependent grid spacing.

At minimum compare:

```text
S = |U_out_all|
S = |U_out_move|
S = U_coh, if retained
```

The purpose is now:

> determine whether the unconstrained 2-D gradient field independently supports the branch-relative flank locations.

Do not make the global gradient field define the fronts.

### STOP GATE

Compare:

```text
branch-relative edge diagnostics
vs
global cross-stream gradients
```

and report where they agree or disagree.

---

# Stage 8 — grid adequacy

Calculate physical grid dimensions with WGS84:

```text
Lx_km(latitude)
Ly_km
sqrt(Lx * Ly)
```

Compare moving transition distance with grid scale:

\[
\chi_x
=
\frac{\bar r_{\mathrm{out}}}{L_x},
\]

\[
\chi_y
=
\frac{\bar r_{\mathrm{out}}}{L_y},
\]

\[
\chi_A
=
\frac{\bar r_{\mathrm{out}}}
{\sqrt{L_xL_y}}.
\]

Also map:

```text
P_stay
P_move
number of moving destinations
number of distinct bearings
transition-distance quantiles
```

The purpose is to answer:

> how many local grid-cell widths does a typical transition span?

and:

> where is the directional distribution likely to be strongly quantized by the 1-degree grid?

Do not claim to test a sub-1-degree grid from an already aggregated matrix.

A true finer-grid sensitivity requires regenerating the transition matrix upstream.

---

# Stage 9 — representative diagnostics

Automatically select representative cells and branch sections including:

- strongest well-supported branch core;
- second/third independent branch;
- high transport + strong directional coherence;
- high transport but angularly ambiguous first moment;
- split-flow case where the first moment is misleading;
- high-R2/low-R1 axial case;
- high-stay case;
- weak-support case;
- clean branch-relative flank;
- noisy/ambiguous branch-relative flank;
- region containing two nearby branches.

For representative cells show:

- sparse outgoing transitions;
- sparse incoming transitions;
- angular distributions;
- first-moment vector;
- `R1`;
- `R2`;
- entropy;
- support;
- stay/move fraction;
- neighborhood directions.

For representative branch sections show:

- branch centerline;
- local tangent/normal;
- raw cross-stream `U_parallel`;
- any robust-smoothed/composite profile;
- support versus distance;
- candidate drop positions;
- neighboring cross-sections;
- secondary global-gradient diagnostics.

---

# Stage 10 — independent 10-day sensitivity experiment

Only after the 30-day methodology has been scientifically reviewed should the same analysis be repeated independently using the native 10-day transition matrix.

Do not derive one matrix from the other.

Use the same mathematical definitions.

Do not tune the 10-day parameters to force agreement with the 30-day result.

Parameters defined in physical units should remain comparable where possible.

Explicitly compare:

- support;
- `P_stay`;
- `P_move`;
- moving distance/grid-scale ratios;
- `|U_out_all|`;
- `|U_out_move|`;
- `R1_out`;
- `R2_out`;
- angular ambiguity;
- inward/outward alignment;
- neighborhood directional coherence;
- number and position of major branches;
- branch rank;
- branch continuity;
- branch width;
- left/right candidate flank positions;
- percentage of branch length with robust flank detection.

The key scientific question is:

> do the dominant coherent transport branches and their cross-stream weakening persist when the temporal lag changes from 30 to 10 days?

Agreement is scientifically valuable.

Disagreement must be reported rather than tuned away.

---

# Synthetic tests

Before interpreting the real-data stages, create focused synthetic transition matrices.

At minimum test:

## 1. Coherent eastward through-flow

Expected:

```text
high |U_out_all|
high R1_out
high R1_in
aligned incoming/outgoing motion
high neighborhood direction consistency
```

## 2. Symmetric split around a false mean direction

For example:

```text
50% NE
50% SE
```

Expected:

- first moment points east;
- `R1` can remain moderately high;
- `R2` reveals a geometry inconsistent with a narrow unimodal eastward distribution;
- optional peak detector identifies two groups;
- the mean vector is flagged as potentially misleading.

## 3. Opposite axial flow

For example:

```text
50% east
50% west
```

Expected:

```text
R1 ~ 0
R2 ~ 1
```

Do not classify as directed through-flow.

## 4. Noisy multidirectional field

Should not rank as a main branch.

## 5. High-stay coherent mover

Verify:

\[
U_{\mathrm{all}}
=
P_{\mathrm{move}}U_{\mathrm{move}}.
\]

A cell with strong moving transport but dominant staying probability must have weak total-population transport intensity.

## 6. Low-support apparently coherent cell

A tiny sample may produce high `R1`, but support flags must identify it as poorly constrained.

## 7. Coherent curved jet

Branch extraction should follow the curve without requiring a predefined shape.

## 8. Two parallel coherent jets

Both must remain available as independent branches.

Do not force them into one branch.

## 9. Unsupported gap

Do not bridge automatically.

## 10. Noisy cross-stream jet profile

Create a coherent branch whose transverse transport profile contains substantial cell-scale noise.

The branch-relative algorithm must:

- make no Gaussian assumption;
- recover persistent cross-stream weakening where possible;
- avoid selecting isolated one-cell drops;
- retain uncertain sections as uncertain.

## 11. Asymmetric jet profile

One flank should be allowed to be sharper or farther from the core than the other.

## 12. Nearby second branch

A cross-section intersecting another coherent branch must be flagged rather than interpreted as a simple flank.

## 13. Dateline crossing

Geodesic geometry must remain correct.

## 14. Latitude-dependent grid size

Zonal spacing must decrease poleward while meridional spacing remains approximately constant.

---

# Required machine-readable outputs

Create a separate timestamped directory for each timestep.

For example:

```text
<postprocessing>/transition_branches/
    argo_30d_<timestamp>/
    argo_10d_<timestamp>/
```

Each run should contain:

```text
run_manifest.json
run_config_resolved.yaml
run_summary.txt

validation/
transition_geometry/
cell_fields/
branches/
cross_sections/
front_edges/
figures/
logs/
```

At minimum produce:

## `transition_geometry/links.parquet`

Sparse transitions plus physical geometry and moving-conditioned probabilities.

## `cell_fields/outward_inward_fields.parquet`

One row per cell containing all continuous diagnostics and support fields.

## `cell_fields/outward_inward_fields.nc`

Regular lat/lon representation.

## `branches/candidate_branch_cells.parquet`

Every candidate branch and its member cells.

## `branches/branch_summary.parquet`

Branch-level diagnostics.

## `branches/branch_threshold_sensitivity.csv`

Report combinations of:

```text
support threshold
transport threshold
coherence threshold
smoothing scale
minimum branch length
```

with resulting:

- valid coverage;
- number of candidate branches;
- number of retained branches;
- length distributions;
- strongest-branch rank stability;
- disconnected significant branches.

## `cross_sections/branch_cross_sections.parquet`

At minimum:

```text
branch_id
section_id
along_branch_distance_km
cross_stream_distance_km
side
U_parallel_all
U_parallel_move
U_parallel_relative
U_out_all
U_out_move
P_move
R1_out
R2_out
R1_in
A_io
C_neigh_out
N_out_total
N_out_move
quality flags
```

## `front_edges/candidate_flank_points.parquet`

Candidate left/right weakening positions and all supporting diagnostics.

Do not convert these automatically into final continuous front lines.

## timestep comparison

After the 10-day run, produce:

```text
timestep_comparison/
    branch_comparison.parquet
    flank_comparison.parquet
    comparison_summary.txt
```

---

# Required diagnostic figures

At minimum produce:

1. total/moving outward support;
2. total/moving inward support;
3. `P_stay`;
4. `P_move`;
5. mean moving transition distance;
6. physical grid size;
7. `|U_out_all|` with decimated vectors;
8. `|U_out_move|` with decimated vectors;
9. `R1_out`;
10. `R2_out`;
11. outward entropy;
12. representative angular distributions;
13. `R1_in`;
14. `R2_in`;
15. incoming/outgoing alignment;
16. neighborhood directional coherence;
17. transport intensity + candidate corridors;
18. unfiltered candidate branches;
19. retained/ranked branches;
20. branch robustness/sensitivity;
21. representative raw branch-relative cross-sections;
22. along-branch composites of cross-stream transport;
23. candidate left/right weakening positions;
24. support across representative cross-sections;
25. global `G_perp` as a secondary diagnostic;
26. agreement between branch-relative and global-gradient diagnostics;
27. grid-adequacy diagnostics;
28. 30-day versus 10-day branch comparison;
29. 30-day versus 10-day flank comparison.

Every plotted field must have a machine-readable counterpart.

---

# Scientific terminology

Use:

```text
first statistical moment
finite-time Lagrangian transport vector
mean transition velocity
transport intensity
moving-conditioned transport
total-population transport
directional coherence
mean-direction reliability
spatial directional persistence
coherent transport branch
branch core
left/right branch flank
candidate front edge
candidate transport edge
```

Do not use:

```text
momentum
```

for the first statistical moment.

Do not call:

- `mu/dt` an Eulerian current measurement;
- low `R1` alone mixing;
- high `R2` alone a coherent branch;
- a first-moment vector automatically representative of a multimodal transition PDF;
- a sudden transport drop automatically a barrier;
- candidate flank points final fronts;
- raw count support independent-sample confidence.

---

# Main scientific questions

The completed staged workflow should eventually answer:

1. Does the total first-moment transport field reveal major current-like branches without a graph?
2. Does removing stay only from the directional statistics improve physical interpretability?
3. Which strong transport regions are weakened because `P_stay` is large?
4. Are branch cells predominantly fed from one upstream direction and discharged toward one downstream direction?
5. Are incoming actual-motion and outgoing directions aligned?
6. Are neighboring branch directions spatially consistent?
7. Where does the first-moment direction become misleading because of split or axial transport?
8. Which major coherent branches are robust?
9. Are multiple/disconnected branches physically significant?
10. Does branch-aligned transport weaken abruptly on one or both sides of those branches?
11. Are the weakening locations persistent along the branch?
12. Do these positions agree with independent global-gradient diagnostics?
13. How does the 1-degree grid interact with the typical transition distance?
14. Are the same dominant branches recovered independently at 30 and 10 days?
15. Are candidate flank positions stable across the two temporal resolutions?
16. Which quantities provide the simplest defensible basis for a later front definition?
17. Which later candidate fronts should be tested with actual side-changing probability \(P_{\rm cross}\)?

Do not proceed automatically to the permeability calculation.

---

# Coding strategy and mandatory stop gates

Work strictly stage by stage.

For each stage:

1. explain briefly what is being calculated and why;
2. implement only that stage;
3. run focused tests relevant to that stage;
4. run the stage on the current real-data timestep;
5. save machine-readable outputs;
6. create diagnostic figures;
7. report:
   - exact input file;
   - timestep;
   - parameters;
   - thresholds;
   - number of valid/invalid cells or links;
   - support statistics;
   - important intermediate values;
   - files created;
   - warnings;
8. explain what the diagnostics appear to show;
9. identify the scientific choices still unresolved;
10. **STOP and wait for explicit instruction.**

Do not pre-implement later stages for convenience.

Do not silently discard:

- low-support cells;
- angularly ambiguous cells;
- disconnected branches;
- noisy cross-sections;
- failed flank detections.

Retain them with explicit flags/reason codes.

The first real-data analysis must begin with the **30-day matrix and Stage 0 only**.

The 10-day analysis comes later as an independent sensitivity experiment.

The guiding principle is:

> **First locate strong finite-time transport. Then determine whether its direction is trustworthy, whether it is coherently supplied from upstream, and whether that direction persists spatially. Only after a stable branch exists should we ask where branch-aligned transport weakens across-stream.**