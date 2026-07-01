## Rtraj 2 zarr
I want to add a new dedicated conversion tool, not modify the existing CSV converter too much.

Relevant existing files:
- src/kinematicparcels/tools/argo_to_zarr.py
- src/kinematicparcels/tools/zarr_writer.py
- experiments/configs/southern_ocean/ARGO_to_zarr.yml

Please inspect these files first to understand the existing output format and the current resampling / region filtering logic.

Goal:
Create a new script:

    src/kinematicparcels/tools/rtraj_to_zarr.py

This script should convert original Argo trajectory NetCDF files, named like:

    1902267_Rtraj.nc
    4903848_Rtraj.nc

into the same final Zarr format produced by the current ARGO CSV converter and compatible with my Lagrangian code outputs.

Important:
Do not try to preserve CSV compatibility inside this new script. The old CSV converter can stay as it is. This new script is only for original Argo Rtraj NetCDF files.

Input:
The new YAML config should support:
```yaml
    input:
      rtraj_dir: F:/ARGO/netcdf/Rtraj/core
      pattern: "*_Rtraj.nc"
```

and optionally:
```yaml
    input:
      rtraj_files:
        - F:/ARGO/netcdf/Rtraj/core/1902267_Rtraj.nc
        - F:/ARGO/netcdf/Rtraj/core/4903848_Rtraj.nc
```
Output:
Use the existing build_dataset_from_trajectories(...) and build_zarr_encoding(...) functions so that the final dataset is compatible with the current Lagrangian trajectory Zarr outputs.

Required per-observation dataframe columns:
- platform_code
- time
- lat
- lon
- z

Also preserve a trajectory index and obs index in the same way as the existing converter.

Data extraction from Rtraj:
- platform_code: read `PLATFORM_NUMBER` and convert it to integer `WMO`.
- time: prefer `JULD_ADJUSTED` when valid, otherwise use `JULD`.
- lat/lon: use `LATITUDE` and `LONGITUDE`.
- parking pressure: use `REPRESENTATIVE_PARK_PRESSURE` from the `N_CYCLE` group/axis.
- cycle number: prefer `CYCLE_NUMBER_ADJUSTED` when valid, otherwise `CYCLE_NUMBER`. Cycle numbers should only be used internally to map `REPRESENTATIVE_PARK_PRESSURE` from the cycle axis to the measurement rows. Do not include `cycle_number` in the final output dataset unless it is explicitly useful for debugging. Since the trajectories are later resampled onto a different time grid, `cycle_number` should not be interpolated or treated as a physical trajectory variable.
Parking pressure to z:
- `REPRESENTATIVE_PARK_PRESSURE` is pressure in dbar, not exact geometric depth.
- For now use the approximation:

      `z = REPRESENTATIVE_PARK_PRESSURE`

  with positive values, consistent with the existing converter where `z = 1000.0`.
- Add dataset attributes explaining that z is approximated from Argo `REPRESENTATIVE_PARK_PRESSURE` in dbar.
- If later there is a clean local pattern for optional gsw support, you may add it, but do not make gsw a hard dependency.

Mapping parking pressure to observations:
- In Argo trajectory files, `REPRESENTATIVE_PARK_PRESSURE` is per cycle.
- Map it to measurement rows using cycle numbers:
  - Prefer `CYCLE_NUMBER_ADJUSTED` matched against `CYCLE_NUMBER_INDEX_ADJUSTED`.
  - Fall back to `CYCLE_NUMBER` matched against `CYCLE_NUMBER_INDEX`.
- If a row cannot be mapped to a valid representative park pressure:
  - use `processing.parking_depth.fallback_value` if provided;
  - otherwise keep z as NaN.
- Print a summary of missing z values.

Filtering:
Do not implement the old segmentation logic in this new script.
The trajectory is already complete per WMO, so we do not want `split_as_new` / `longest` / `max_gap_days` / `max_speed_km_per_day` logic here.

Do not use the old cut_from_first_entry boolean in this new script. Instead implement an explicit region selection policy.

Keep region filtering, but simplify it:
- Reuse the existing `RegionManager` / `ALL_REGIONS` logic from argo_to_zarr.py if possible.
- Support config:
```yaml
    processing:
      regions:
        names_or_labels:
          - DP
        selection_mode: from_first_entry
        input_lon_mode: "-180_180"
```
Behavior:
- If no regions are provided, keep all trajectories unchanged.
- If regions are provided, apply the selected selection_mode.
- Drop trajectories that do not satisfy the selected region policy.
- Do not cut a trajectory again if it later leaves the region.

Allowed selection_mode values:

1. from_first_entry
   Keep trajectories that enter one of the selected regions at least once.
   For each kept trajectory, cut it from the first point inside the selected region onward.
   Do not cut it again if it later leaves the region.

2. full_if_enters
   Keep trajectories that enter one of the selected regions at least once.
   Keep the full original trajectory.

3. initial_inside
   Keep only trajectories whose first valid point is inside one of the selected regions.
   Keep the full trajectory for those selected floats.
   This corresponds to floats effectively starting/released inside the region of interest.

Resampling:
Keep the same resampling strategy as the existing argo_to_zarr.py.
During resampling, do not linearly interpolate `platform_code` or z. Treat z as a non-interpolated variable and fill it with nearest/forward-backward fill, because parking pressure is a cycle-level property rather than a continuously sampled depth.
If the existing shared resampling code currently interpolates all numeric columns except `platform_code`, update the new script so that z is also excluded from numeric interpolation.

It should support:
- enabled
- frequency
- interpolate
- reference_time
- shared_time
- shift_start_to_reference

Reuse the existing helper functions if reasonable, or move shared code into a small internal helper module if that is cleaner. Do not do a large refactor unless necessary.

Example config:
Create a new config file, for example:

    experiments/configs/southern_ocean/RTRAJ_to_zarr.yml

with:
```yaml
    input:
      rtraj_dir: F:/ARGO/netcdf/Rtraj/core
      pattern: "*_Rtraj.nc"

    output:
      path: F:/ARGO/zarr/DP_rtraj.zarr

    processing:
      parking_depth:
        mode: representative_park_pressure
        fallback_value: 1000.0

    regions:
      names_or_labels:
        - DP
      selection_mode: from_first_entry
      input_lon_mode: "-180_180"

    resample:
      enabled: true
      frequency: 10d
      interpolate: time
      reference_time: 2000-01-01T00:00:00Z
      shared_time: true
      shift_start_to_reference: false
```

CLI:
The script should be runnable like:

    `python -m kinematicparcels.tools.rtraj_to_zarr experiments/configs/southern_ocean/RTRAJ_to_zarr.yml`

Testing / validation:
- Add focused tests if the repo has a test framework.
- At minimum, make helper functions small and testable.
- Run a minimal import/CLI help check.
- If possible, create a tiny synthetic xarray Dataset mimicking an Rtraj file and test:
  - PLATFORM_NUMBER extraction
  - JULD_ADJUSTED fallback to JULD
  - cycle-to-REPRESENTATIVE_PARK_PRESSURE mapping
  - resampling does not break the final dataframe shape
  - region selection_mode behavior
Implementation style:
- Keep the new script close in style to argo_to_zarr.py.
- Avoid broad refactors.
- Do not modify the old CSV converter unless extracting shared helper functions is clearly useful and low risk.
- Keep error messages explicit.