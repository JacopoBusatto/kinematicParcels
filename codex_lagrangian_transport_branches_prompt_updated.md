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

## 1. Analyze supplied transition matrices independently

The primary real-data experiment should start with a supplied:

```text
dt = 30 days
```

transition table.

A supplied 10-day transition table should later be analyzed independently as a temporal-resolution sensitivity experiment.

Additional transition tables may also be analyzed later, including alternative constructions at the same timestep.

**Transition-matrix generation is outside the scope of this research analysis.**

In particular, choices concerning:

- trajectory resampling;
- overlapping versus non-overlapping displacement segments;
- temporal sampling phase/offset;
- trajectory QC;
- lag construction;
- generation of transition counts and probabilities;

belong to the upstream transition-matrix-generation workflow.

The analysis under `research/transition_branches/` must:

1. accept the supplied sparse transition table as authoritative input;
2. analyze it exactly as supplied;
3. never reconstruct trajectories;
4. never infer whether trajectory segments overlap;
5. never enforce a particular sampling strategy;
6. never regenerate or modify the transition matrix;
7. record available timestep and provenance information when provided.

The same branch-analysis code must therefore be usable without modification on, for example:

```text
30-day matrix, construction A
30-day matrix, construction B
10-day matrix
other future transition matrices
```

Comparison among such matrices is a scientific sensitivity experiment, not part of the branch-detection algorithm itself.

Do not assume inside this research analysis that one matrix-construction strategy is intrinsically preferable.

The 30-day matrix is analyzed first because its longer displacement scale may reduce angular quantization by the 1-degree grid. The 10-day matrix remains scientifically important because 10 days is the native Argo-cycle temporal resolution.

Do not tune one experiment to reproduce the other.

Use identical mathematical definitions and, wherever scientifically sensible, identical physical-unit parameters.

Raw transition counts must **not** be described as independent-sample confidence estimates because repeated transitions can originate from the same float, regardless of the upstream segment-construction strategy.

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
- modify or regenerate generic transition matrices as part of this research task;
- add a preprocessing or `Stage -1` matrix-generation step;
- reconstruct trajectories inside this research analysis;
- infer trajectory-level overlap/resampling properties from an aggregated sparse table;
- reject a supplied matrix because of its overlap/resampling strategy;
- derive one transition matrix from another inside this analysis;
- use external velocity products;
- use external ACC fronts in detection scores;
- tune parameters to known ACC/front positions;
- implement pair dispersion/diffusivity;
- build another global angular-mode graph;
- force every supported cell into a named topology.

The research analysis itself must consume **already generated sparse transition tables and their raw transition counts**.

If the requested transition table does not exist or its path is invalid, stop and report the missing input. Do not generate it inside this research analysis.

# Primary real-data inputs

Primary supplied 30-day table for the first run:

```text
C:\Users\Jacopo\Documents\SIMULATIONS\kinematicParcels\southern_ocean\ARGO_850-1150\postprocessing\gridded_transition_matrix_dt_30d_table.parquet
```

Known 10-day table for the later temporal-resolution sensitivity run:

```text
C:\Users\Jacopo\Documents\SIMULATIONS\kinematicParcels\southern_ocean\ARGO_850-1150\postprocessing\gridded_transition_matrix_dt_10d_table.parquet
```

These are independently supplied analysis inputs. The research workflow must not infer how either table was generated from trajectories.

Do not silently substitute one timestep or matrix construction for another.

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

These values are only contextual checks for the known 10-day table. Do not assume analogous counts for the 30-day table.

Still validate every supplied input independently.

**Never silently renormalize invalid rows.**

Record in the run manifest, when available:

- exact input path;
- timestep supplied in config;
- file size;
- file modification time;
- cryptographic input hash;
- grid/domain metadata;
- any matrix-generation provenance supplied externally.

Do not attempt to infer missing trajectory-generation provenance from the aggregated table.

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
- source-to-destination forward bearing evaluated at the source;
- destination-to-source back bearing evaluated at the destination;
- actual arrival-motion bearing expressed in the destination-local tangent frame;
- local eastward displacement `dx_km`;
- local northward displacement `dy_km`;
- `is_stay`;
- transition count;
- forward transition probability;
- moving-conditioned transition probability.

Handle the antimeridian correctly.

For incoming diagnostics retain the angle definitions explicitly.

## Incoming source-side direction

```text
theta_in_source
```

is the bearing from destination cell \(j\) toward source cell \(i\), evaluated locally at the destination.

It answers:

> From which side was this destination cell supplied?

With `pyproj.Geod.inv`, this is the back azimuth at the destination.

## Incoming actual-motion direction in the destination-local frame

```text
theta_in_motion_destination
```

is the direction in which the transition was actually arriving at destination cell \(j\), expressed in the local tangent frame of the destination.

If the geodesic back azimuth at the destination points from destination toward source, then:

\[
\theta_{\mathrm{in,motion,destination}}
=
\theta_{\mathrm{in,source}} + 180^\circ
\pmod{360^\circ}.
\]

Use this destination-local arrival direction when comparing incoming motion with outgoing transport from the same cell.

Do **not** simply reuse the source-local forward bearing as the destination-local arrival direction.

Never silently mix these angle definitions.

# Stage 0 — validation and observational support

Run Stage 0 first on the **supplied 30-day matrix only**.

Stage 0 validates only properties represented by the supplied sparse transition table.

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

Do **not** attempt to validate or infer:

- trajectory-segment independence;
- segment overlap;
- original trajectory resampling;
- temporal sampling phase;
- trajectory QC decisions;
- the upstream construction of the timestep itself.

Those are upstream matrix-generation properties and are outside this research analysis.

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

Record input provenance available at table/config level, including the exact input file and cryptographic hash.

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
- report the exact input file/hash and available provenance;
- stop.

Do not implement Stage 1 until explicitly instructed.

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

## Distance-weighted first-moment direction

Where the displacement first moment is non-zero, save its direction explicitly:

\[
\theta_{\mu,\mathrm{out}}
=
\arg
\left(
\boldsymbol{\mu}_{\mathrm{out,move}}
\right).
\]

Because `mu_out_all` is a positive scalar multiple of `mu_out_move` whenever `P_move > 0`, the two have the same direction.

This is a **distance-weighted transport direction**: longer displacement vectors contribute more strongly to the first moment.

Do not assume yet that this direction is identical to the probability-weighted circular mean direction calculated in Stage 2.

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
- `theta_mu_out` where defined;
- strong transport regions with weak observational support;
- numerical warnings.

Then stop.

# Stage 2 — directional coherence and mean-direction reliability

The displacement first moment alone can produce a misleading mean direction.

For example, two separated outgoing branches may average into an intermediate vector even if almost no transitions actually follow that mean direction.

Therefore calculate both the first and second circular harmonics and compare the probability-weighted angular mean with the distance-weighted first-moment direction from Stage 1.

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

`theta1_out` is a **probability/count-weighted angular mean direction**. Unlike `theta_mu_out`, it does not give longer transitions greater directional leverage simply because they travel farther.

## Distance-weighted versus circular-mean direction

Retain both:

```text
theta_mu_out
theta1_out
```

and calculate their wrapped angular disagreement:

\[
\Delta\theta_{\mu,1,\mathrm{out}}
=
\left|
\operatorname{wrap}_{[-180^\circ,180^\circ)}
\left(
\theta_{\mu,\mathrm{out}}
-
\theta_{1,\mathrm{out}}
\right)
\right|.
\]

Save this in degrees in `[0,180]`.

Interpretation:

- small disagreement: the distance-weighted first moment and angular probability distribution support a similar mean direction;
- large disagreement: long and short transitions emphasize different directions, so the first-moment transport vector may not summarize the transition geometry well.

This disagreement is a **mean-direction reliability diagnostic**, not a topology classifier.

Do not silently decide that either `theta_mu_out` or `theta1_out` is the definitive branch direction before inspecting the real-data diagnostics.

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
small theta_mu vs theta1 disagreement
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
theta_mu vs theta1 disagreement
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

This is particularly useful when \(R_1/R_2\) or `delta_theta_mu1_out` indicate that the first-moment vector may be hiding a split or multimodal distribution.

Do not construct an angular-mode graph.

### STOP GATE

Report:

- where strong first-moment transport also has high directional coherence;
- where `theta_mu_out` and `theta1_out` agree or disagree;
- where the first-moment vector appears potentially misleading;
- representative split/axial/multidirectional cells;
- relationship among `R1`, `R2`, angular disagreement, entropy and transport strength.

Then stop.

# Stage 3 — inward coherence and through-flow continuity

Repeat the angular diagnostics using:

\[
\widetilde Q_{\mathrm{in}}.
\]

For each incoming link use both the source-side direction and the **actual arrival-motion direction expressed in the destination-local tangent frame**.

Calculate:

```text
R1_in
theta1_in_source
theta1_in_motion_destination
R2_in
theta2_in
H_in
```

and optional incoming angular peaks.

Where useful, also calculate an incoming distance-weighted first-moment direction in the destination-local frame:

```text
theta_mu_in_motion_destination
```

and its disagreement with the incoming circular mean:

```text
delta_theta_mu1_in
```

This is diagnostic only. Do not interpret the magnitude of the destination-conditioned incoming first moment as a transport intensity exactly equivalent to `U_out_all`.

The incoming distribution is used primarily to establish whether a candidate branch cell is coherently supplied from upstream.

The primary transport-intensity field remains outward.

## Incoming/outgoing continuity

For the primary circular-mean continuity diagnostic compare:

\[
\theta_{1,\mathrm{in,motion,destination}}
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
\theta_{1,\mathrm{in,motion,destination}}
\right).
\]

Retain the continuous value.

Values close to 1 indicate that the cell is fed in approximately the same local direction as the flow leaving it.

Where both distance-weighted directions are defined, also retain a secondary diagnostic:

\[
A_{io,\mu}
=
\cos
\left(
\theta_{\mu,\mathrm{out}}
-
\theta_{\mu,\mathrm{in,motion,destination}}
\right).
\]

Do not silently choose between circular-mean and first-moment continuity before inspecting the diagnostics.

This is the expected geometry of coherent through-flow.

# Stage 3B — spatial directional persistence

A coherent cell is not automatically a coherent branch.

A stable branch must persist spatially through neighboring cells.

Calculate neighborhood-direction consistency for supported neighboring cells.

For the probability-weighted outgoing circular mean, for example:

\[
C_{\mathrm{neigh,out},1}(i)
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

Where `theta_mu_out` is defined, also calculate the analogous first-moment directional persistence:

\[
C_{\mathrm{neigh,out},\mu}(i)
=
\frac{
\sum_{k\in\mathcal N_i}
w_k
\cos
\left(
\theta_{\mu,\mathrm{out}}(i)
-
\theta_{\mu,\mathrm{out}}(k)
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

Also calculate analogous neighborhood consistency using:

```text
theta1_in_motion_destination
```

where incoming support is sufficient.

If `theta_mu_in_motion_destination` is retained, its neighborhood consistency may also be calculated as a secondary diagnostic.

The distinction is important:

```text
R1_out:
    agreement among transitions leaving one cell

C_neigh_out_1:
    agreement among probability-weighted angular mean directions of nearby cells

C_neigh_out_mu:
    agreement among distance-weighted first-moment directions of nearby cells
```

These diagnose different aspects of spatial persistence.

Do not silently choose a definitive directional representation before Stage 4 review.

### STOP GATE

Explicitly answer:

1. Are the strong outward transport pathways coherently fed from upstream?
2. Are incoming destination-local actual-motion and outgoing directions aligned?
3. Are their directions spatially persistent across neighboring cells?
4. Do `theta_mu_out`-based and `theta1_out`-based neighborhood persistence agree?
5. Where do apparently strong first-moment vectors fail the inward or neighborhood-continuity checks?
6. Which regions remain ambiguous because of split/multimodal angular structure?

Then stop.

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

theta_mu_out
theta1_out
delta_theta_mu1_out
R1_out
R2_out
H_out

theta1_in_motion_destination
theta_mu_in_motion_destination, where retained
R1_in
R2_in
H_in

A_io
A_io_mu, where defined

C_neigh_out_1
C_neigh_out_mu
C_neigh_in_motion
```

## Directional representation remains an open scientific choice

The distance-weighted first-moment direction `theta_mu_out` and the probability-weighted circular mean `theta1_out` need not coincide.

Do not choose one silently as the branch direction.

Instead inspect:

```text
theta_mu_out
theta1_out
delta_theta_mu1_out
R1_out
R2_out
entropy
optional angular peaks
neighborhood persistence
```

together.

A stable directed branch should generally have a transport direction that is both physically strong and geometrically representative of the underlying transition distribution.

## R2 remains a reliability/complexity diagnostic

Do not require simply:

```text
R2 > threshold
```

for a branch.

Instead, use \(R_2\), entropy, `delta_theta_mu1_out`, and optional peaks to identify cells where the first-moment direction may not faithfully represent the actual transition geometry.

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
- use angular reliability, inward continuity and neighborhood persistence to assess whether those corridors are stable and interpretable.

### STOP GATE

Before branch extraction:

- compare the transport-intensity maps;
- compare `theta_mu_out` and `theta1_out`;
- map their angular disagreement;
- show `R1`, `R2`, inward coherence, alignment and neighborhood persistence;
- identify which field provides the cleanest candidate branch backbone;
- document which directional representation appears most defensible for branch interpretation;
- document remaining scientific choices;
- stop.

Do not choose the branch field or branch-direction convention silently.

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
theta_mu_out vs theta1_out disagreement
A_io
A_io_mu, where defined
C_neigh_out_1
C_neigh_out_mu
C_neigh_in_motion
moving support
R2/entropy/peak ambiguity flags
spatial persistence
```

The first-moment vector provides a physically distance-weighted transport direction.

The circular diagnostics determine whether that direction is representative of the transition-angle distribution.

The inward diagnostics determine whether the cell is coherently supplied from upstream.

The neighborhood diagnostics determine whether the direction forms a spatially persistent branch.

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
- mean `delta_theta_mu1_out`;
- mean circular incoming/outgoing alignment;
- mean first-moment incoming/outgoing alignment where defined;
- mean neighborhood coherence for the retained directional representations;
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
- sensitivity to the directional-reliability choices if relevant;
- branch-level summary table.

Then stop.

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
theta_mu_out
theta1_out
delta_theta_mu1_out
R1_out
R2_out
R1_in
A_io
A_io_mu, where defined
C_neigh_out_1
C_neigh_out_mu
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

Only after the supplied 30-day matrix has been scientifically reviewed should the same analysis be repeated independently using the supplied native 10-day transition matrix.

The branch-analysis workflow must treat both matrices as authoritative inputs and remain unaware of their trajectory-level construction details.

Do not derive one matrix from the other inside this analysis.

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
- `theta_mu_out`;
- `theta1_out`;
- `delta_theta_mu1_out`;
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

> do the dominant coherent transport branches and their cross-stream weakening persist when the supplied temporal lag changes from 30 to 10 days?

Agreement is scientifically valuable.

Disagreement must be reported rather than tuned away.

If additional transition tables are later supplied, including alternative 30-day constructions, run them through the same workflow without modifying the branch-analysis algorithm and compare them as separate sensitivity experiments.

# Synthetic tests

Before interpreting the real-data stages, create focused synthetic transition matrices.

At minimum test:

## 1. Coherent eastward through-flow

Expected:

```text
high |U_out_all|
high R1_out
high R1_in
small theta_mu_out vs theta1_out disagreement
aligned destination-local incoming/outgoing motion
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


## 15. Distance-weighted versus circular-mean direction

Create a transition distribution in which many short transitions favor one direction while fewer long transitions favor another.

Expected:

- `theta1_out` follows the probability/count-weighted angular distribution;
- `theta_mu_out` is shifted toward the longer displacement vectors;
- `delta_theta_mu1_out` is non-zero and correctly quantified;
- the disagreement is retained as a reliability warning rather than silently resolved.

## 16. Destination-local arrival direction

Create a sufficiently long geodesic transition for which the source-local forward azimuth and destination-local arrival direction differ measurably.

Expected:

- source forward bearing is calculated correctly;
- `theta_in_source` is the destination-to-source back azimuth;
- `theta_in_motion_destination` is the actual arrival-motion direction in the destination-local tangent frame;
- incoming/outgoing alignment uses the destination-local arrival direction.

---

# Required machine-readable outputs

Create a separate timestamped directory for each supplied matrix/run.

For example:

```text
<postprocessing>/transition_branches/
    argo_30d_<timestamp>/
    argo_10d_<timestamp>/
```

If alternative matrices with the same timestep are supplied later, give each run a distinct identifier rather than overwriting an existing run.

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

`run_manifest.json` must record, when available:

```text
exact input path
input file hash
timestep
grid/domain metadata
supplied provenance metadata
code/git revision
```

At minimum produce:

## `transition_geometry/links.parquet`

Sparse transitions plus physical geometry and moving-conditioned probabilities.

Include the explicit geodesic angle conventions needed later, including:

```text
source_forward_bearing
theta_in_source
theta_in_motion_destination
```

## `cell_fields/outward_inward_fields.parquet`

One row per cell containing all continuous diagnostics and support fields.

At minimum retain the directional fields:

```text
theta_mu_out
theta1_out
delta_theta_mu1_out
R1_out
R2_out
H_out

theta1_in_source
theta1_in_motion_destination
theta_mu_in_motion_destination, where retained
delta_theta_mu1_in, where retained
R1_in
R2_in
H_in

A_io
A_io_mu, where defined

C_neigh_out_1
C_neigh_out_mu
C_neigh_in_motion
```

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
theta_mu_out
theta1_out
delta_theta_mu1_out
R1_out
R2_out
R1_in
A_io
A_io_mu
C_neigh_out_1
C_neigh_out_mu
N_out_total
N_out_move
quality flags
```

Use null/NaN for optional diagnostics that are undefined rather than inventing replacement values.

## `front_edges/candidate_flank_points.parquet`

Candidate left/right weakening positions and all supporting diagnostics.

Do not convert these automatically into final continuous front lines.

## timestep / matrix comparison

After the 10-day run, produce:

```text
timestep_comparison/
    branch_comparison.parquet
    flank_comparison.parquet
    comparison_summary.txt
```

If alternative supplied matrices are analyzed later, add analogous comparison outputs without changing the core analysis algorithm.

# Required diagnostic figures

At minimum produce:

1. total/moving outward support;
2. total/moving inward support;
3. `P_stay`;
4. `P_move`;
5. mean moving transition distance;
6. physical grid size;
7. `|U_out_all|` with decimated first-moment vectors;
8. `|U_out_move|` with decimated first-moment vectors;
9. `theta_mu_out` and `theta1_out` comparison;
10. `delta_theta_mu1_out`;
11. `R1_out`;
12. `R2_out`;
13. outward entropy;
14. representative angular distributions, including misleading first-moment examples;
15. destination-local incoming-motion directions;
16. `R1_in`;
17. `R2_in`;
18. incoming/outgoing alignment;
19. `C_neigh_out_1` and `C_neigh_out_mu`;
20. transport intensity + candidate corridors;
21. unfiltered candidate branches;
22. retained/ranked branches;
23. branch robustness/sensitivity;
24. representative raw branch-relative cross-sections;
25. along-branch composites of cross-stream transport;
26. candidate left/right weakening positions;
27. support across representative cross-sections;
28. global `G_perp` as a secondary diagnostic;
29. agreement between branch-relative and global-gradient diagnostics;
30. grid-adequacy diagnostics;
31. 30-day versus 10-day branch comparison;
32. 30-day versus 10-day flank comparison.

Every plotted field must have a machine-readable counterpart.

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
3. Which strong moving-transport regions are weakened in total transport because `P_stay` is large?
4. Where do `theta_mu_out` and `theta1_out` agree, and where do transition distances make them diverge?
5. Are branch cells predominantly fed from one upstream direction and discharged toward one downstream direction?
6. Are destination-local incoming actual-motion and outgoing directions aligned?
7. Are neighboring branch directions spatially consistent?
8. Where does the first-moment direction become misleading because of split, axial, multimodal, or distance-dependent transport?
9. Which major coherent branches are robust?
10. Are multiple/disconnected branches physically significant?
11. Does branch-aligned transport weaken abruptly on one or both sides of those branches?
12. Are the weakening locations persistent along the branch?
13. Do these positions agree with independent global-gradient diagnostics?
14. How does the 1-degree grid interact with the typical transition distance?
15. Are the same dominant branches recovered in the supplied 30-day and 10-day matrices?
16. Are candidate flank positions stable across the supplied temporal resolutions?
17. If alternative matrices with the same timestep are supplied, which results persist across matrix constructions?
18. Which quantities provide the simplest defensible basis for a later front definition?
19. Which later candidate fronts should be tested with actual side-changing probability \(P_{\rm cross}\)?

Do not proceed automatically to the permeability calculation.

# Coding strategy and mandatory stop gates

Work strictly stage by stage.

For each stage:

1. explain briefly what is being calculated and why;
2. implement only that stage;
3. run focused tests relevant to that stage;
4. run the stage on the currently supplied transition table;
5. save machine-readable outputs;
6. create diagnostic figures;
7. report:
   - exact input file;
   - input file hash;
   - supplied timestep;
   - available provenance metadata;
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

Do not add a matrix-generation or trajectory-preprocessing stage to this research workflow.

Do not silently discard:

- low-support cells;
- angularly ambiguous cells;
- cells with large `theta_mu_out` versus `theta1_out` disagreement;
- disconnected branches;
- noisy cross-sections;
- failed flank detections.

Retain them with explicit flags/reason codes.

The first real-data analysis must begin with the **supplied 30-day matrix and Stage 0 only**.

The supplied 10-day matrix comes later as an independent sensitivity experiment.

Any additional supplied matrix should be analyzable with the same workflow without changing the branch-analysis algorithm.

The guiding software principle is:

> **Matrix generation and matrix interpretation are separate problems. The transition-branch analysis should know the contents of the supplied sparse matrix, not the trajectory-level choices used to construct it.**

The guiding scientific principle is:

> **First locate strong finite-time transport. Then determine whether its direction is trustworthy, whether it is coherently supplied from upstream, and whether that direction persists spatially. Only after a stable branch exists should we ask where branch-aligned transport weakens across-stream.**
