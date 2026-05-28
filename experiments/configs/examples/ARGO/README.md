# ARGO Conversion

This folder contains example YAML files for the `convert-argo-to-zarr` tool.

The converter reads one or more ARGO CSV files, extracts one surface fix per transmission phase, optionally splits and filters the resulting trajectories, optionally resamples them, and writes a Parcels-compatible Zarr dataset.

## Run Command

```bash
convert-argo-to-zarr experiments/configs/examples/ARGO/argo_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.argo_to_zarr experiments/configs/examples/ARGO/argo_to_zarr_example.yml
```

## Current Workflow

For each selected CSV file the tool currently does the following:

1. Reads the CSV with `pandas.read_csv`.
2. Maps the configured column names to the internal names `platform_code`, `time`, `lat`, `lon`, and optionally `pressure`.
3. Groups rows by `(platform_code, time, lat, lon)` and keeps one row per group.
4. If a pressure column is available, keeps the row with the smallest pressure in each group.
5. Assigns a fixed parking depth directly to the Zarr `z` variable.
6. Sorts each platform trajectory by time.
7. Applies the configured segmentation policy, using time gaps and optionally a speed-based jump threshold.
8. Applies optional region filtering.
9. Applies optional time resampling and interpolation.
10. Writes a Parcels-compatible `trajectory x obs` Zarr dataset.

## Output Schema

The generated Zarr dataset is designed to be readable by the existing Parcels postprocessing tools in this repository.

The `trajectory` coordinate is a sequential segment index in the exported dataset. The original ARGO float identifier is preserved separately in `platform_code`.

Coordinates:

- `trajectory`
- `obs`

Always written variables:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`

Conditionally written variables:

- any variables listed in `variables.optional`

## Configuration Reference

### `input`

Defines which CSV files are converted.

Supported keys:

- `csv_files`: explicit list of CSV file paths
- `csv_glob`: a glob expression such as `C:/data/ARGO/*.csv`
- `csv_dir`: directory containing CSV files
- `pattern`: filename pattern used together with `csv_dir`; default is `*.csv`

Notes:

- You can use one or more of these selectors together.
- The converter merges all resolved files, removes duplicates, sorts them, and then processes them.
- If no files are found, conversion stops with `FileNotFoundError`.

Example:

```yaml
input:
  csv_dir: C:/Users/Jacopo/OneDrive - CNR/ARGO/ACC
  pattern: PR_PF_*.csv
```

### `columns`

Maps CSV column names to the converter's expected fields.

Supported keys:

- `platform_code`
- `time`
- `lat`
- `lon`
- `pressure`

Defaults:

```yaml
columns:
  platform_code: PLATFORM_CODE
  time: DATE (YYYY-MM-DDTHH:MI:SSZ)
  lat: LATITUDE (degree_north)
  lon: LONGITUDE (degree_east)
  pressure: PRES_ADJUSTED (decibar)
```

Notes:

- `platform_code`, `time`, `lat`, and `lon` are required.
- `pressure` is optional but recommended because it improves surface-point selection.
- If the configured pressure column is missing, the tool falls back to `PRES (decibar)` when present.
- If no pressure column is available at all, the first row in each `(platform_code, time, lat, lon)` group is kept.

### `variables`

Controls additional CSV columns to carry into the output dataset.

Supported keys:

- `optional`: list of extra CSV columns to preserve

Example:

```yaml
variables:
  optional:
    - TEMP_ADJUSTED (degree_Celsius)
    - PSAL_ADJUSTED (psu)
```

Notes:

- Every requested optional variable must exist in every selected CSV file.
- These columns are written as trajectory variables in the Zarr output.

### `processing.parking_depth`

Controls how parking depth is assigned.

Supported keys:

- `mode`
- `value`

Currently supported mode:

- `fixed`

Example:

```yaml
processing:
  parking_depth:
    mode: fixed
    value: 1000.0
```

Notes:

- In the current implementation this is the only supported mode.
- The selected value is written to `z`.
- Any other mode raises `ValueError`.

### `processing.segment`

Controls splitting of long trajectories when continuity breaks are detected.

Supported keys:

- `mode`
- `max_gap_days`
- `min_duration_days`
- `max_speed_km_per_day`

Supported values for `mode`:

- `ignore`: keep each platform as one irregular trajectory
- `longest`: split on large gaps, then keep only the longest segment
- `split_as_new`: split on large gaps and keep each retained segment as a separate trajectory

Accepted aliases:

- `split` becomes `split_as_new`
- `separate` becomes `split_as_new`
- `irregular` becomes `ignore`

Defaults:

```yaml
processing:
  segment:
    mode: ignore
    max_gap_days: 10.0
    min_duration_days: 0.0
    max_speed_km_per_day: null
```

Behavior details:

- A new segment starts whenever the gap between two consecutive observations is greater than `max_gap_days`.
- If `max_speed_km_per_day` is provided, a new segment also starts when the implied great-circle speed between consecutive observations exceeds that threshold.
- In `ignore` mode, `min_duration_days` is ignored.
- In `longest` mode, the tool keeps the segment with the most observations; if tied, it keeps the one with the longest duration.
- In `split_as_new` mode, segments shorter than `min_duration_days` are dropped.
- In `split_as_new` mode, each retained segment becomes its own trajectory entry in the output dataset.
- Output `trajectory` values are sequential integer indices, while `platform_code` keeps the original ARGO identifier.
- `max_speed_km_per_day` uses great-circle distance, so dateline crossings are handled geometrically rather than by raw longitude difference.

Example:

```yaml
processing:
  segment:
    mode: split_as_new
    max_gap_days: 10.0
    min_duration_days: 15.0
    max_speed_km_per_day: 500.0
```

### `processing.regions`

Filters trajectories according to whether they enter one or more configured regions.

Supported keys:

- `names_or_labels`
- `cut_from_first_entry`
- `input_lon_mode`

Defaults:

```yaml
processing:
  regions:
    names_or_labels: []
    cut_from_first_entry: false
    input_lon_mode: "-180_180"
```

Behavior details:

- `names_or_labels` may contain region names or short labels already defined in the repository region system.
- If the list is empty or omitted, no region filtering is applied.
- A trajectory is kept only if at least one observation falls inside any selected region.
- If `cut_from_first_entry` is `true`, the trajectory is trimmed so it starts at the first observation inside the selected region set.
- `input_lon_mode` is passed to the region matcher and should match the longitude convention of your ARGO coordinates.

Example:

```yaml
processing:
  regions:
    names_or_labels:
      - sic
      - Ionian Sea 1
    cut_from_first_entry: false
    input_lon_mode: "-180_180"
```

### `processing.resample`

Controls optional temporal resampling and interpolation.

Supported keys:

- `enabled`
- `frequency`
- `interpolate`

Defaults:

```yaml
processing:
  resample:
    enabled: false
    interpolate: time
```

Behavior details:

- If `enabled` is `false` and no `frequency` is provided, trajectories are left unchanged.
- If `enabled` is `true`, `frequency` should be a pandas-compatible frequency string such as `1d`, `12h`, or `6h`.
- The converter lowercases the frequency string before using it.
- Numeric variables are interpolated with `DataFrame.interpolate(method=...)`.
- Non-numeric variables are forward-filled and backward-filled.
- The final timestamp of the original segment is always preserved, even if it does not fall exactly on the resampling grid.

Example:

```yaml
processing:
  resample:
    enabled: true
    frequency: 1d
    interpolate: time
```

### `output`

Defines where the converted Zarr dataset is written.

Supported keys:

- `path`

Example:

```yaml
output:
  path: ./outputs/argo/acc_ARGO.zarr
```

Notes:

- `output.path` is required.
- Parent directories are created automatically.
- Existing content at that Zarr path is overwritten because the converter writes with `mode="w"`.

## Minimal Config Example

```yaml
input:
  csv_files:
    - C:/data/ARGO/PR_PF_1900042.csv

output:
  path: ./outputs/argo/minimal_argo.zarr
```

This is valid because the converter will use default column mappings, fixed parking depth of `1000.0`, no splitting, no region filtering, and no resampling.

## Full Example

See [argo_to_zarr_example.yml](argo_to_zarr_example.yml) for a complete example.

## Current Limitations

- Parking depth estimation is not yet inferred from the vertical profile; only a fixed value is supported.
- Surface detection is based on repeated `(platform_code, time, lat, lon)` groups, not on a more detailed phase classifier.
- Region filtering uses the existing repository region definitions only.
- Optional variables must be available in all input files.