# Lagrangian transport branches and finite-time barriers

This research workflow analyses a sparse gridded transition matrix without
reconstructing trajectories. Local moving-direction modes define a directed
transport graph. Branch segments from that graph provide local tangent/normal
frames; barriers are independently diagnosed as supported minima of finite-time
side-changing endpoint probability.

The analysis deliberately does **not** estimate diffusivity or pair dispersion.
The existing generic gridded-transition-matrix implementation is not modified.

Run the Southern Ocean Argo baseline from the repository root:

```powershell
python -m research.lagrangian_barriers.run_lagrangian_barrier_analysis `
  --config research/lagrangian_barriers/configs/southern_ocean_argo.yaml
```

Use `--stop-after validation|geometry|modes|graph|branches|permeability|barriers|figures`
for a checkpoint run. `--resume` resumes the most recent run with an identical
resolved configuration and input SHA256. An existing run is never reused
implicitly.

All thresholds are explicit in the resolved YAML. Rejected modes, graph edges,
unsupported cross-sections, and rejected minima remain in audit tables with
reason codes. Positive cross-stream offset is the left side when following the
directed branch.

Map projections are configured under `plotting`. Supported values are
`PlateCarree`, `SouthPolarStereo`, `NorthPolarStereo`, `Robinson`, and
`Mercator`. Regular-grid scalar diagnostics use `pcolormesh`; scatter and line
layers are reserved for sparse modes, branches, and barrier geometry.

For a circular Southern Ocean map:

```yaml
plotting:
  projection: SouthPolarStereo
  central_longitude: 0
  circular_boundary: true
  draw_coastlines: true
```
