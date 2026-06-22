## **Parcels tail writing status**
The remaining problem is that Parcels scheduled output is regular, but deleted particles/grouped entities can still be selected for writing and serialized with their deletion time, so the raw zarr gets off-grid tail points.
The current tail cleaning works only because it rewrites the zarr after integration, but that rewrite is the bottleneck.
The idea to investigate next is an online fix in the runner: intercept the Parcels write-selection/output path before ParticleFile.write and make sure only positions whose times are multiples of outputdt_hours are written.

Prompt to send when tokens are back:

We need to analyze again the Parcels tail-writing issue in this repo.

Context from the latest work:
- The duplicate initial record problem and the tail-writing problem are separate.
- In src/kinematicparcels/runner/run_experiment.py we already narrowed the duplicate-start fix so it only checks obs=0 vs obs=1 and compacts that duplicate in place.
- The remaining bottleneck is tail cleaning: Parcels still writes a particle/grouped-entity final state even when its status is Delete, and that state can have a time that is not a multiple of outputdt_hours.
- The current post-hoc cleaner rewrites the zarr to nullify these off-grid tail points, but that zarr rewrite is expensive and is the bottleneck.
- We previously reasoned that the raw output should ideally contain only records whose times are multiples of outputdt_hours.
- We also investigated that Parcels scheduled output is regular, but deleted particles may still be selected for writing and serialized with their deletion time at the next output event.

What I want now:
1. Read the current repo state first, especially:
  - src/kinematicparcels/runner/run_experiment.py
  - src/kinematicparcels/runner/kernels.py
  - src/kinematicparcels/runner/grouped_kernels.py
  - tests/test_continuous_release.py
  - TO_DO.md
2. Inspect the currently installed local Parcels version/API as needed.
3. Re-analyze the exact tail-writing control flow carefully.
4. Decide the least invasive way to prevent off-grid tail points from being written online.
5. Prefer avoiding a post-hoc zarr rewrite if possible.
6. Do not assume previous edits are still present; verify the current files first.

Please answer these questions clearly:
- Exactly where in Parcels does a deleted particle/grouped entity remain eligible for output?
- Is the correct hook point ParticleSet.execute, ParticleFile.write, particledata._to_write_particles, or something else?
- If we patch the write-selection step, can we guarantee that only positions whose times are multiples of outputdt_hours are written?
- Is filtering Delete-state particles enough, or do we need an explicit time-grid filter as well?
- Could grouped entities behave differently from singleton particles in this output path?
- What is the safest runner-local patch in this repo that minimizes maintenance risk against Parcels updates?

Please give me:
1. A concise status recap of the issue
2. The exact control-flow explanation for the tail write
3. The best candidate implementation strategy
4. The main risks/tradeoffs of that strategy
5. A minimal validation plan before changing production behavior

Important constraints:
- Keep the solution local to this repo if possible
- Prefer a small runner-side patch over broad postprocessing changes
- Focus on preventing off-grid tail writes online
- Be explicit about what is already solved (duplicate obs0/obs1) versus what remains (tail writing)

or just trim the gifs maybe its easier


## CLUSTER STRENGTH
I want to add a new postprocessing analysis module to this repository:

Please read the existing postprocessing architecture before editing. In particular inspect:
- src/kinematicparcels/postprocessing/config/models.py
- src/kinematicparcels/postprocessing/config/loader.py
- src/kinematicparcels/postprocessing/core/gridding.py
- src/kinematicparcels/postprocessing/runner/run_postprocessing.py
- src/kinematicparcels/postprocessing/workflows/run_density.py
- src/kinematicparcels/postprocessing/analyses/density.py
- src/kinematicparcels/postprocessing/animations/density.py
- POSTPROCESSING.md

Add a new analysis called `cluster_strength`.

Scientific definition
---------------------
Use the cluster strength definition from Huntley et al. 2015, “Clusters, deformation, and dilation: Diagnostics for material accumulation regions”, stored in the repo reference folder (`references\JGR Oceans - 2015 - Huntley - Clusters  deformation  and dilation  Diagnostics for material accumulation regions.pdf`).

For each target grid point x* and each time t, compute:
```LaTex
$ C(x*, t) = sum_n exp(- (d(x*, x_n(t)) / L)^2 ) $
```
where:
- $x_n(t)$ is the position of particle n at time t
- $d(x)$ is the selected distance metric
- $L$ is the tunable length scale, provided by config as `scale_km`
- the output variable should be named `cluster_strength`

Configuration
-------------
Add a new optional YAML section:

```yaml
cluster_strength:
  scale_km: float              # required, positive
  distance: haversine          # default haversine;
  mask: true                   # if true, compute/output only grid cells explored at least once
  animate: false               # if true, create an animation
  plot_snaps: false            # if true, plot selected snapshots
  timestep_snaps: null         # int or list[int]; required if plot_snaps is true
  vmin: 0
  vmax: null
  cmap: viridis
```
Please use the corrected spelling `cluster_strength` everywhere. Do not introduce the misspelled `cluster_strenght`.

Integration points
------------------
Follow the existing postprocessing pattern.

Add:
- `src/kinematicparcels/postprocessing/analyses/cluster_strength.py`
- `src/kinematicparcels/postprocessing/workflows/run_cluster_strength.py`
- `experiments/configs/examples/postprocessing/10_cluster_strength.yml`, an example postprocessing YAML


Update:
- `src/kinematicparcels/postprocessing/analyses/__init__.py`
- `src/kinematicparcels/postprocessing/config/models.py`
- `src/kinematicparcels/postprocessing/config/loader.py`
- `src/kinematicparcels/postprocessing/runner/run_postprocessing.py`
- `POSTPROCESSING.md`

The workflow should:
1. get the trajectory table using the existing base-products helpers;
2. build/reuse the regular grid using the existing grid section and `build_grid_from_config`;
3. compute cluster strength for every saved timestep;
4. save:
   - `cluster_strength.nc`
   - optionally a table if consistent with existing products
   - optional snapshot PNGs
   - optional GIF animation

Grid and mask behavior
----------------------
Use the regular grid already implemented in `postprocessing/core/gridding.py` that is defined in the `grid` section of the .

If `mask: true`, define valid target grid cells as cells that are visited by at least one particle at least once during the whole trajectory dataset. Compute/store cluster strength only on those cells; all other grid cells should be NaN.

If mask: true, apply the mask only to the target grid cells where cluster strength is evaluated/output. Do not use the mask to pre-filter the particle table. For each valid target grid cell, all particles present at that timestep may contribute if they are within the distance cutoff.

Distance design
---------------
Implement only the default distance now.
`distance: haversine` should use great-circle/haversine distance in km.

However, design the code so future distance backends can be added cleanly. In the future I want to add a fjord/skeleton distance:
- user passes `distance: skeleton_kml`
- user passes `kml_path`
- the KML contains connected segments defining the fjord skeleton
- distance between A and B is:
  distance from A to closest point on skeleton
  + shortest path length along the skeleton
  + distance from closest skeleton point to B

Do not implement this future skeleton/KML metric now, but avoid hard-coding the cluster-strength computation to one distance function.

Efficiency requirements
-----------------------
The naive computation is O(n_time * n_grid * n_particles), which can become too slow.

Please implement the default distance efficiently:
- avoid building a full dense distance matrix for large datasets;
- process one timestep at a time;
- use a finite Gaussian cutoff, default internally to 4 * scale_km unless there is a better local convention;
- preferably use scipy.spatial.cKDTree on local projected coordinates to find candidate particles near each target grid point, then compute the Gaussian only for candidates;
- handle empty timesteps gracefully;
- keep memory bounded by chunking grid points if needed.

The exact haversine distance may be used for final candidate distances after the KDTree neighbor query. If you use a local projection for both neighbor query and distance approximation, document the approximation clearly and keep it appropriate for regional domains.

Outputs
-------
The NetCDF should have dimensions:

time, lat, lon

and variable:

cluster_strength(time, lat, lon)

Include useful attributes:
- scale_km
- distance metric
- formula
- cutoff_factor or cutoff_km if used
- grid metadata consistent with existing gridded outputs

Plotting
--------
For snapshots:
- `plot_snaps: true` requires `timestep_snaps`.
- `timestep_snaps` can be one integer or a list of integers.
- Support negative indices like Python indexing: -1 means the last timestep.
- Save files with clear names such as `cluster_strength_timestep_DATE:TIME.png`.

For animation:
- reuse the style of the existing density animation where practical.
- respect `vmin`, `vmax`, and `cmap`.

Validation and tests
--------------------
Add focused tests if the test structure supports it.

At minimum test:
1. a tiny synthetic dataset where one particle exactly on one grid point gives contribution 1 at that point;
2. two particles give the sum of two Gaussian contributions;
3. `mask: true` leaves never-explored cells as NaN;
4. invalid config: missing or non-positive `scale_km`;
5. snapshot index parsing including negative indices.

Please run the relevant tests or, if the environment cannot run them, report exactly what could not be run and why.

Before implementing, ask me all the questions you might have.