# ARGO Rtraj Conversion

This folder contains example YAML files for the `convert-rtraj-to-zarr` tool.

The converter reads original ARGO trajectory NetCDF files named like `1902267_Rtraj.nc`, extracts one trajectory per WMO platform, maps cycle-level representative parking pressure onto measurement rows, optionally splits each trajectory into contiguous depth-bin segments, optionally applies region selection, optionally resamples the trajectory, and writes Parcels-compatible Zarr datasets.

This tool is intentionally separate from the ARGO CSV converter. It is only for original ARGO Rtraj NetCDF files.

## Run Command

```bash
convert-rtraj-to-zarr experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.rtraj_to_zarr experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml
```

## Current Workflow

For each selected Rtraj file the tool currently does the following:

1. Opens the NetCDF file with `xarray.open_dataset`.
2. Reads `PLATFORM_NUMBER` and converts it to integer `platform_code`.
3. Reads time from `JULD_ADJUSTED` where valid, falling back to `JULD`.
4. Reads position from `LATITUDE` and `LONGITUDE`.
5. Maps cycle-level `REPRESENTATIVE_PARK_PRESSURE` onto measurement rows using cycle numbers.
6. Writes that mapped pressure directly to `z`, using positive values in dbar.
7. Sorts the platform trajectory by time.
8. Optionally splits each trajectory at large raw-`JULD` gaps.
9. Optionally discards cycles with excessive near-surface time or insufficient parking time, splitting around discarded cycles.
10. Drops singleton fragments produced by the Rtraj cleaning segmentation.
11. Optionally splits each trajectory into contiguous depth-bin segments.
12. Applies optional region selection to each resulting segment.
13. Applies optional time resampling and interpolation.
14. Writes Parcels-compatible `trajectory x obs` Zarr dataset(s).

The converter does not apply the ARGO CSV `split_as_new`, `longest`, or speed policy. Rtraj-specific cleaning is controlled by `processing.frequency` and `processing.near_surface`, and it happens before depth-bin splitting and region selection.

## Output Schema

The generated Zarr dataset is designed to be readable by the existing Parcels postprocessing tools in this repository.

The `trajectory` coordinate is a sequential output index. The original ARGO float identifier is preserved separately in `platform_code`.

Coordinates:

- `trajectory`
- `obs`

Always written variables:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`

Conditionally written trajectory-level variables:

- `depth_bin`
- `depth_bin_interval`

The output dataset includes attributes documenting that `z` is approximated from Argo `REPRESENTATIVE_PARK_PRESSURE` in dbar, without pressure-to-geometric-depth conversion.

## Configuration Reference

### `input`

Defines which Rtraj NetCDF files are converted.

Supported keys:

- `rtraj_files`: explicit list of Rtraj NetCDF file paths
- `rtraj_glob`: a glob expression such as `C:/data/ARGO/netcdf/Rtraj/core/*_Rtraj.nc`
- `rtraj_dir`: directory containing Rtraj NetCDF files
- `pattern`: filename pattern used together with `rtraj_dir`; default is `*_Rtraj.nc`

Notes:

- You can use one or more of these selectors together.
- The converter merges all resolved files, removes duplicate paths, sorts them, and then processes them.
- If no files are found, conversion stops with `FileNotFoundError`.

Example:

```yaml
input:
  rtraj_dir: C:/data/ARGO/netcdf/Rtraj/core
  pattern: "*_Rtraj.nc"
```

### `processing.parking_depth`

Controls how `z` is assigned.

Supported keys:

- `mode`
- `fallback_value`

Currently supported mode:

- `representative_park_pressure`

Example:

```yaml
processing:
  parking_depth:
    mode: representative_park_pressure
    fallback_value: 1000.0
```

Behavior details:

- `REPRESENTATIVE_PARK_PRESSURE` is read from the cycle axis.
- The value is mapped to each measurement row using cycle numbers.
- `z = REPRESENTATIVE_PARK_PRESSURE` for now, in dbar and positive downward.
- If a row cannot be mapped, `fallback_value` is used when provided.
- If `fallback_value` is omitted, unmapped rows keep `z = NaN`.
- The converter prints a summary of unmapped and final missing `z` values.

### Cycle Mapping

The tool prefers adjusted cycle mapping when available:

- measurement cycle: `CYCLE_NUMBER_ADJUSTED`
- cycle-axis index: `CYCLE_NUMBER_INDEX_ADJUSTED`

Rows that cannot be mapped through adjusted cycle values fall back to:

- measurement cycle: `CYCLE_NUMBER`
- cycle-axis index: `CYCLE_NUMBER_INDEX`

Cycle numbers are used only internally for parking-pressure mapping. They are not written to the final dataset.

### `processing.frequency`

Controls optional splitting when consecutive source Rtraj fixes are separated by too much time.

Supported keys:

- `enabled`
- `source_max_gap_days`

Example:

```yaml
processing:
  frequency:
    enabled: true
    source_max_gap_days: 29
```

Behavior details:

- If `enabled` is `false` or the section is omitted, this cleaning rule is skipped.
- Gaps are computed from consecutive raw `JULD` values.
- If `delta_days > source_max_gap_days`, the trajectory is split between the two fixes.
- Both fixes are kept, but they are placed in different segments.
- Segments produced by this rule must contain more than one point; singleton fragments are dropped.
- This rule runs before depth-bin splitting and before region selection.

### `processing.near_surface`

Controls optional cycle filtering based on the ratio between parking time and time spent outside parking phases.

Supported keys:

- `enabled`
- `parking_to_near_surface_ratio`
- `near_surface_max_hours`
- `parking_min_hours`
- `vertical_speed_m_per_s`
- `transmission_fallback_hours`

Example:

```yaml
processing:
  near_surface:
    enabled: true
    parking_to_near_surface_ratio: 4
    near_surface_max_hours: 18
    parking_min_hours: 72
```

Behavior details:

- If `enabled` is `false` or the section is omitted, this cleaning rule is skipped.
- Any omitted threshold is not checked.
- A cycle is discarded when `parking_hours / near_surface_hours < parking_to_near_surface_ratio`.
- A cycle is discarded when `near_surface_hours > near_surface_max_hours`.
- A cycle is discarded when `parking_hours < parking_min_hours`.
- The discarded cycle's measurement rows are removed, and the trajectory is split around it.
- Segments produced by this rule must contain more than one point; singleton fragments are dropped.
- This rule runs before depth-bin splitting and before region selection.

Near-surface duration is the sum of:

- `JULD_DESCENT_END - JULD_DESCENT_START`
- `JULD_DEEP_DESCENT_END - JULD_DESCENT_END`
- `JULD_ASCENT_END - JULD_ASCENT_START`
- `JULD_ASCENT_START - JULD_DEEP_ASCENT_START`
- `JULD_TRANSMISSION_END - JULD_TRANSMISSION_START`

Parking duration is `JULD_PARK_END - JULD_PARK_START`. Deep-park timing is not included.

When a near-surface phase duration cannot be computed from timestamps, the converter uses these fallbacks:

- transmission: `transmission_fallback_hours`, default `1`
- descent/ascent: `REPRESENTATIVE_PARK_PRESSURE / vertical_speed_m_per_s` seconds
- deep descent: `(max(PRES) for the cycle - REPRESENTATIVE_PARK_PRESSURE) / vertical_speed_m_per_s` seconds, clamped at zero distance
- deep ascent: `max(PRES) for the cycle / vertical_speed_m_per_s` seconds

The default vertical speed is `0.1 m/s`. Parking-duration checks require valid `JULD_PARK_START` and `JULD_PARK_END`; the converter does not invent a parking duration if those timestamps are missing.

### `processing.depth_bins`

Controls optional splitting into contiguous segments of similar parking pressure.

Supported keys:

- `enabled`
- `output_mode`
- `bins`

Currently supported `output_mode`:

- `per_bin`

Example:

```yaml
processing:
  depth_bins:
    enabled: true
    output_mode: per_bin
    bins:
      - label: z0000_0750
        min: 0.0
        max: 750.0

      - label: z0750_1250
        min: 750.0
        max: 1250.0

      - label: z2500_inf
        min: 2500.0
        max: null
```

Behavior details:

- If `enabled` is `false` or the section is omitted, trajectories are not split by depth.
- Bins are lower-inclusive and upper-exclusive: `[min, max)`.
- A bin with `max: null` means `[min, +inf)`.
- Bin labels are used in output paths and are sanitized to path-safe strings.
- Bins must not overlap.
- Gaps between bins are allowed. Rows whose `z` does not fall in any bin are dropped from depth-bin output.
- Splitting preserves contiguous runs. The converter does not merge separated runs from the same bin.

Example depth sequence:

```text
1000, 1000, 1800, 1800, 1000
```

With bins that place `1000` and `1800` in different ranges, this becomes three trajectory segments:

```text
1000, 1000
1800, 1800
1000
```

The first and third segments keep the same `platform_code` but receive different output `trajectory` IDs. With `output_mode: per_bin`, both are written into the same depth-bin Zarr output, while the `1800` segment is written into another depth-bin Zarr output.

Output path behavior:

```yaml
output:
  path: ./outputs/argo/dp_rtraj_ARGO.zarr
```

with labels `z0750_1250` and `z1500_2500` writes:

```text
./outputs/argo/dp_rtraj_ARGO_z0750_1250.zarr
./outputs/argo/dp_rtraj_ARGO_z1500_2500.zarr
```

### `processing.regions`

Filters trajectories according to whether they enter one or more configured regions.

Supported keys:

- `names_or_labels`
- `selection_mode`
- `input_lon_mode`

Defaults:

```yaml
processing:
  regions:
    names_or_labels: []
    selection_mode: from_first_entry
    input_lon_mode: "-180_180"
```

Supported values for `selection_mode`:

- `from_first_entry`: keep trajectories that enter the selected regions at least once, and cut each kept trajectory from the first point inside the selected region onward.
- `full_if_enters`: keep trajectories that enter the selected regions at least once, but keep the full original trajectory.
- `initial_inside`: keep only trajectories whose first valid point is inside one of the selected regions, and keep the full trajectory.

Behavior details:

- `names_or_labels` may contain region names or short labels already defined in the repository region system.
- If the list is empty or omitted, no region selection is applied.
- Trajectories are not cut again if they later leave the selected region.
- `input_lon_mode` is passed to the region matcher and should match the longitude convention of the Rtraj coordinates.

Example:

```yaml
processing:
  regions:
    names_or_labels:
      - DP
    selection_mode: from_first_entry
    input_lon_mode: "-180_180"
```

### `processing.resample`

Controls optional temporal resampling and interpolation.

Supported keys:

- `enabled`
- `frequency`
- `interpolate`
- `reference_time`
- `shared_time`
- `shift_start_to_reference`

Defaults:

```yaml
processing:
  resample:
    enabled: false
    interpolate: time
```

Behavior details:

- If `enabled` is `false` and no `frequency` is provided, trajectories are left unchanged.
- If `enabled` is `true`, `frequency` should be a pandas-compatible frequency string such as `10d`, `1d`, or `12h`.
- The converter lowercases the frequency string before using it.
- `lon` and `lat` are interpolated on the time grid.
- `platform_code`, `z`, `depth_bin`, and `depth_bin_interval` are not linearly interpolated. They are forward-filled and backward-filled because `z` is a cycle-level parking-pressure property and depth-bin metadata is trajectory-level context.
- Longitude interpolation unwraps longitudes before interpolation and wraps them back to `[-180, 180)`.
- The final timestamp of the original trajectory is preserved when no shared target grid is used.
- `shared_time: true` builds one common grid for all trajectories and requires `reference_time`.
- `shift_start_to_reference: true` shifts each trajectory so its first point lands on `reference_time`.

Example:

```yaml
processing:
  resample:
    enabled: true
    frequency: 10d
    interpolate: time
    reference_time: 2000-01-01T00:00:00Z
    shared_time: true
    shift_start_to_reference: false
```

### `output`

Defines where the converted Zarr dataset is written.

Supported keys:

- `path`

Example:

```yaml
output:
  path: ./outputs/argo/dp_rtraj_ARGO.zarr
```

Notes:

- `output.path` is required.
- Parent directories are created automatically.
- Existing content at that Zarr path is overwritten because the converter writes with `mode="w"`.
- When `processing.depth_bins.enabled: true`, the configured path is used as a base path and each non-empty depth bin is written to a suffixed Zarr path.

## Minimal Config Example

```yaml
input:
  rtraj_files:
    - C:/data/ARGO/netcdf/Rtraj/core/1902267_Rtraj.nc

output:
  path: ./outputs/argo/minimal_rtraj_ARGO.zarr
```

This is valid because the converter will use representative parking pressure for `z`, no fallback value, no region selection, and no resampling.

## Full Example

See [rtraj_to_zarr_example.yml](rtraj_to_zarr_example.yml) for a complete example.

## Current Limitations

- `z` is an approximation from parking pressure in dbar, not geometric depth in meters.
- Optional `gsw` pressure-to-depth conversion is not implemented.
- The tool does not preserve arbitrary extra Rtraj variables.
- Region selection uses the existing repository region definitions only.
- Depth bins are based on the mapped `z` approximation, so they inherit the same pressure-vs-depth limitation.
