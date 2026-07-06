# ARGO Rtraj Conversion

This folder contains example YAML files for the staged `convert-rtraj-to-zarr-v2`
tool. The legacy converter example has been moved to `legacy/`.

The v2 converter reads original ARGO Rtraj NetCDF files named like
`1902267_Rtraj.nc`, builds one representative trajectory fix per cycle, applies
QC and jump controls, splits by parking-depth bins, optionally filters by
region, optionally resamples the segments, and writes Parcels-compatible Zarr
datasets.

## Run

Diagnostics mode is the recommended first pass:

```bash
convert-rtraj-to-zarr-v2 experiments/configs/examples/Rtraj/rtraj_to_zarr_v2_example.yml
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
python -m kinematicparcels.tools.rtraj_to_zarr_v2 experiments/configs/examples/Rtraj/rtraj_to_zarr_v2_example.yml
```

## Main Stages

For each selected Rtraj file the v2 converter currently does the following:

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
13. Writes one Zarr per depth bin when `depth_bins.output_mode: per_bin`.

## Progress Output

The v2 tool reports high-level progress only:

- `Processing RTRAJ files`
- `Writing depth-bin Zarr` when writing per-bin outputs

Inner per-segment resampling bars are suppressed in v2 because they are too
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

`z` is parking pressure in dbar with positive values and no
pressure-to-geometric-depth conversion.

## Configuration Notes

- `run.mode: diagnostics` runs the full staged processing and optionally writes
  plots, but `output.write_zarr` controls whether Zarr files are written.
- `run.max_files` is intended for diagnostics subsets.
- `diagnostics.plots.raw_vs_qc_segments: false` disables plot generation.
- `output.overwrite: false` uses safe Zarr write mode and fails if the output
  already exists.
- `output.overwrite: true` overwrites the configured Zarr output path.

See [rtraj_to_zarr_v2_example.yml](rtraj_to_zarr_v2_example.yml) for a complete
example.

## Legacy Converter

The previous converter is preserved for reproducibility:

- Legacy module: `kinematicparcels.tools.legacy.rtraj_to_zarr`
- Legacy example: `legacy/rtraj_to_zarr_example.yml`
- Legacy Southern Ocean config:
  `experiments/configs/southern_ocean/legacy/RTRAJ_to_zarr.yml`
