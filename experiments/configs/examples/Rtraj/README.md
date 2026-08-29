# ARGO Rtraj Conversion

This folder contains example YAML files for the staged `convert-rtraj-to-zarr`
tool. The legacy converter example has been moved to `legacy/`.

The converter reads original ARGO Rtraj NetCDF files named like
`1902267_Rtraj.nc`, builds one representative trajectory fix per cycle, applies
QC and jump controls, splits by parking-depth bins, optionally filters by
region, optionally resamples the segments, samples configured original Rtraj
observations, and writes Parcels-compatible Zarr datasets.

## Run

Diagnostics mode is the recommended first pass:

```bash
convert-rtraj-to-zarr experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml
```

Diagnostics mode can limit the number of files with `run.max_files` and can
write raw/controlled/resampled comparison plots.

For conversion, set:

```yaml
run:
  mode: convert

output:
  write_zarr: true
```

Then run the same command.

Equivalent module form:

```bash
python -m kinematicparcels.tools.rtraj_to_zarr experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml
```

## Main Stages

For each selected Rtraj file the converter currently does the following:

1. Opens the Rtraj NetCDF file.
2. Reads time from `JULD_ADJUSTED` where valid, falling back to `JULD`.
3. Reads longitude, latitude, cycle number, measurement code, and QC flags.
4. Infers parking depth from `REPRESENTATIVE_PARK_PRESSURE` when available, then
   from pressure values inside the park window, then configured fill/fallback
   rules.
5. Selects one representative fix per cycle.
6. Applies configured QC filters.
7. Merges compatible QC-separated segments.
8. Drops isolated bad jump points/short bad-location blocks when the bridge is
   physically plausible.
9. Splits remaining non-physical jumps.
10. Splits segments by parking-depth bin.
11. Applies region selection.
12. Applies optional duration filtering and resampling.
13. Samples configured observations onto the final resampled points.
14. Writes one Zarr per depth bin when `depth_bins.output_mode: per_bin`.

## Progress Output

The tool reports high-level progress only:

- `Processing RTRAJ files`
- `Writing depth-bin Zarr` when writing per-bin outputs

Inner per-segment resampling bars are suppressed in the staged converter because they are too
small and misleading when processing many files.

## Output Schema

The generated Zarr dataset is designed to be readable by the existing Parcels
postprocessing tools in this repository.

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

Configured observation variables are written with the output names under
`observations.variables`. The example adds `temp` and `psal`; matching time and
pressure are internal and are not written.

`z` is parking pressure in dbar with positive values and no
pressure-to-geometric-depth conversion.

## Configuration Notes

Observation sampling is configured generically:

```yaml
observations:
  enabled: true
  sample_at_fallback_depth: false
  time:
    adjusted: JULD_ADJUSTED
    fallback: JULD
  pressure:
    adjusted: PRES_ADJUSTED
    adjusted_qc: PRES_ADJUSTED_QC
    fallback: PRES
    fallback_qc: PRES_QC
    valid_qc: ["0", "1", "2"]
    missing_qc: reject
  variables:
    temp:
      adjusted: TEMP_ADJUSTED
      adjusted_qc: TEMP_ADJUSTED_QC
      fallback: TEMP
      fallback_qc: TEMP_QC
      valid_qc: ["0", "1", "2"]
      missing_qc: reject
      valid_min: null
      valid_max: null
    psal:
      adjusted: PSAL_ADJUSTED
      adjusted_qc: PSAL_ADJUSTED_QC
      fallback: PSAL
      fallback_qc: PSAL_QC
      valid_qc: ["0", "1", "2"]
      missing_qc: reject
      valid_min: null
      valid_max: null
```

Adjusted values are preferred row by row, with raw values used wherever the
adjusted value is missing or non-finite. Use `adjusted: null` for variables
without an adjusted counterpart. To add another observed variable, add another
entry under `observations.variables`; no Python change is needed.

Numeric pressure and variable sources can be filtered before matching with
source-specific `adjusted_qc` and `fallback_qc` fields. When `valid_qc` is set,
the QC belonging to the chosen adjusted or raw source must be listed there.
An explicit QC `"0"` is distinct from missing QC and is accepted only when it
appears in `valid_qc`. `missing_qc` accepts `accept` or `reject`; a QC variable
that is absent, the wrong length, blank, or fill-valued follows that policy.
If a finite adjusted value fails QC, it is rejected rather than replaced with
the raw value from the same measurement.

Observation time retains its existing adjusted/raw fallback behavior and is
not filtered by these numeric-source controls.

Optional finite `valid_min` and `valid_max` bounds are inclusive and are
applied after QC. Values outside the range become `NaN` before candidate
ranking. The generic converter supplies no numerical defaults; choose bounds
appropriate for the parameter, region, and depth range. Omitted QC fields and
null bounds preserve the previous sampling behavior.

The local Southern Ocean configuration uses `[-3, 15]` degree Celsius for
temperature and `[30, 40]` for practical salinity in its 850-1150 dbar
product.

Observations are attached only after trajectory resampling and are never
interpolated. Candidates must be in the trajectory point's current depth bin,
then are ranked by closest time, closest representative parking pressure, and
original measurement index. No pressure tolerance is applied.
`sample_at_fallback_depth: false` preserves the default behavior: points whose
parking pressure came from `parking_depth.fallback_value` are skipped and
receive `NaN` for every observation variable. Set it to `true` to make those
points eligible for the same depth-bin and ranking procedure, using the
configured numeric fallback pressure.

- `run.mode: diagnostics` runs the full staged processing and optionally writes
  plots, but `output.write_zarr` controls whether Zarr files are written.
- `run.max_files` is intended for diagnostics subsets.
- `diagnostics.plots.raw_vs_qc_segments: false` disables plot generation.
- `controlled_stage_summary.csv` includes per-file adjusted/raw acceptance and
  QC, missing-QC, and range-rejection counts under
  `observation_filter_counts`; diagnostics mode also prints aggregate counts.
- `output.overwrite: false` uses safe Zarr write mode and fails if the output
  already exists.
- `output.overwrite: true` overwrites the configured Zarr output path.

See [rtraj_to_zarr_example.yml](rtraj_to_zarr_example.yml) for a complete
example.

## Legacy Converter

The previous converter is preserved for reproducibility:

- Legacy module: `kinematicparcels.tools.legacy.rtraj_to_zarr`
- Legacy example: `legacy/rtraj_to_zarr_example.yml`
- Legacy Southern Ocean config:
  `experiments/configs/southern_ocean/legacy/RTRAJ_to_zarr.yml`
