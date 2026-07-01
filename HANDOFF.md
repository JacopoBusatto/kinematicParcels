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
