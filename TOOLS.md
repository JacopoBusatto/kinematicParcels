# Tools

This document summarizes the utilities currently living under `src/kinematicparcels/tools`.

It separates:

- supported user-facing command-line tools
- internal helper modules used by those tools
- ad hoc inspection scripts that are useful during development but are not packaged as stable commands

---

## User-Facing CLI Tools

The following tools are installed through the package entry points defined in `pyproject.toml`.

### `convert-argo-to-zarr`

**Purpose**

Convert one or more ARGO CSV files into a Parcels-compatible trajectory Zarr dataset.

The converter:

- extracts one surface fix per transmission phase
- keeps the original ARGO identifier in `platform_code`
- optionally splits or filters the trajectories
- optionally resamples and interpolates them
- writes a `trajectory x obs` Zarr dataset readable by the repository postprocessing tools

**Run**

```bash
convert-argo-to-zarr experiments/configs/examples/ARGO/argo_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.argo_to_zarr experiments/configs/examples/ARGO/argo_to_zarr_example.yml
```

**CLI arguments**

- `config`: path to the YAML configuration file

**Configuration file structure**

The command itself only accepts the YAML path. The actual processing options live in the YAML file.

Supported top-level sections:

- `input`
- `output`
- `columns`
- `variables`
- `processing`

#### `input`

Controls which CSV files are read.

Supported keys:

- `csv_files`: explicit list of file paths
- `csv_glob`: glob expression
- `csv_dir`: input directory
- `pattern`: filename pattern used with `csv_dir`

#### `output`

Supported keys:

- `path`: output Zarr path

#### `columns`

Maps CSV columns to the internal names expected by the converter.

Supported keys:

- `platform_code`
- `time`
- `lat`
- `lon`
- `pressure`

#### `variables`

Supported keys:

- `optional`: list of extra CSV columns to preserve in the output dataset

#### `processing.parking_depth`

Current supported mode:

- `mode: fixed`
- `value`: constant depth assigned to `z`

#### `processing.segment`

Controls segmentation of a platform trajectory.

Supported keys:

- `mode`: `ignore`, `longest`, or `split_as_new`
- accepted `mode` aliases: `split` and `separate` map to `split_as_new`; `irregular` maps to `ignore`
- `max_gap_days`
- `min_duration_days`
- `max_speed_km_per_day`

#### `processing.regions`

Optional trajectory filtering by geographical region.

Supported keys:

- `names_or_labels`
- `cut_from_first_entry`
- `input_lon_mode`

#### `processing.resample`

Optional temporal resampling and interpolation.

Supported keys:

- `enabled`
- `frequency`
- `interpolate`
- `reference_time`
- `shared_time`
- `shift_start_to_reference`
- `align_start`

#### `processing.resample.align_start`

Compatibility block for aligning all trajectories to a common start time.

Supported keys:

- `enabled`
- `start_time`

When `align_start.enabled: true` is used, the converter uses `start_time` as the resampling reference time and, unless explicitly overridden, also enables both `shared_time` and `shift_start_to_reference`.

**Important note on time layout**

Even when `shared_time: true` is used, the written Zarr output is collapsed back to each trajectory's valid local observation window. The output keeps absolute `time`, but `obs` remains a local per-trajectory index rather than a global synchronized time coordinate.

**Output**

Typical variables written:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`
- any columns listed in `variables.optional`

**Examples**

- Example YAML: `experiments/configs/examples/ARGO/argo_to_zarr_example.yml`
- Example notes: `experiments/configs/examples/ARGO/README.md`

---

### `convert-drifter-to-zarr`

**Purpose**

Convert one or more drifter CSV files into a Parcels-compatible trajectory Zarr dataset.

The converter:

- keeps the source drifter identifier in `platform_code`
- optionally clips trajectories at `drogue_lost_date`
- optionally filters platforms by minimum drogue length
- optionally splits irregular trajectories into segments
- optionally filters by region and resamples/interpolates the time axis
- writes a `trajectory x obs` Zarr dataset readable by the repository postprocessing tools

**Run**

```bash
convert-drifter-to-zarr experiments/configs/examples/DRIFTERS/drifter_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.drifter_to_zarr experiments/configs/examples/DRIFTERS/drifter_to_zarr_example.yml
```

**CLI arguments**

- `config`: path to the YAML configuration file

**Configuration file structure**

Supported top-level sections:

- `input`
- `output`
- `columns`
- `processing`

#### `input`

Controls which CSV files are read.

Supported keys:

- `csv_files`: explicit list of file paths
- `csv_glob`: glob expression
- `csv_dir`: input directory
- `pattern`: filename pattern used with `csv_dir`

#### `output`

Supported keys:

- `path`: output Zarr path

#### `columns`

Maps CSV columns to the internal names expected by the converter.

Supported keys:

- `platform_code`
- `time`
- `lat`
- `lon`
- `drogue_lost_time`
- `drogue_length`

#### `processing.drogue`

Supported keys:

- `clip_after_loss`
- `minimum_length_m`
- `min_length_m`: accepted alias for `minimum_length_m`

#### `processing.segment`

Controls segmentation of a platform trajectory from its nominal cadence.

Supported keys:

- `mode`: `ignore`, `longest`, or `split_as_new`
- accepted `mode` aliases: `split` and `separate` map to `split_as_new`; `irregular` maps to `ignore`
- `step_hours`
- `time_step_hours`, `expected_step_hours`, `resolution_hours`: accepted aliases for `step_hours`
- `tolerance_minutes`

#### `processing.regions`

Optional trajectory filtering by geographical region.

Supported keys:

- `names_or_labels`
- `cut_from_first_entry`
- `input_lon_mode`

#### `processing.resample`

Optional temporal resampling and interpolation.

Supported keys:

- `enabled`
- `frequency`
- `interpolate`
- `reference_time`
- `shared_time`
- `shift_start_to_reference`
- `align_start`

#### `processing.resample.align_start`

This uses the same alignment behavior as the ARGO converter because the drifter tool reuses the shared resampling pipeline.

Supported keys:

- `enabled`
- `start_time`

**Output**

Typical variables written:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`

**Examples**

- Example YAML: `experiments/configs/examples/DRIFTERS/drifter_to_zarr_example.yml`

---

### `convert-aoml-drifter-to-zarr`

**Purpose**

Convert AOML Global Drifter Program interpolated 6-hour NetCDF files into a Parcels-compatible trajectory Zarr dataset.

The converter:

- reads files named like `drifter_6h_XXXX.nc`
- keeps the AOML `ID` as trajectory-level `platform_code`
- drops `WMO` while opening the dataset, because `ID` is the stable identifier and `WMO` can trigger xarray fill-value warnings in these files
- filters drifters by metadata `DrogueLength`
- clips observations to the drogued interval
- optionally filters by region and resamples/interpolates the time axis using the shared RTRAJ-style pipeline
- writes a `trajectory x obs` Zarr dataset readable by the repository postprocessing tools

**Run**

```bash
convert-aoml-drifter-to-zarr experiments/configs/examples/DRIFTERS/aoml_drifter_to_zarr_example.yml
```

Southern Ocean configuration:

```bash
convert-aoml-drifter-to-zarr experiments/configs/southern_ocean/AOML_drifter_to_zarr.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.aoml_drifter_to_zarr experiments/configs/examples/DRIFTERS/aoml_drifter_to_zarr_example.yml
```

**CLI arguments**

- `config`: path to the YAML configuration file

**Configuration file structure**

Supported top-level sections:

- `input`
- `output`
- `processing`

#### `input`

Controls which NetCDF files are read.

Supported keys:

- `netcdf_files` or `drifter_files`: explicit list of file paths
- `netcdf_glob` or `drifter_glob`: glob expression
- `netcdf_dir`, `netcdf_dirs`, `drifter_dir`, or `drifter_dirs`: input directory/directories
- `pattern`: filename pattern used with directory inputs, default `drifter_6h_*.nc`

Example for the split AOML archive:

```yaml
input:
  drifter_glob: F:/PLATFORMS/DRIFTERS/netcdf_6h/netcdf_*/drifter_6h_*.nc
```

Equivalent explicit-directory form:

```yaml
input:
  drifter_dirs:
    - F:/PLATFORMS/DRIFTERS/netcdf_6h/netcdf_1_5000
    - F:/PLATFORMS/DRIFTERS/netcdf_6h/netcdf_5001_15000
    - F:/PLATFORMS/DRIFTERS/netcdf_6h/netcdf_10001_15000
    - F:/PLATFORMS/DRIFTERS/netcdf_6h/netcdf_15001_current
  pattern: "drifter_6h_*.nc"
```

#### `output`

Supported keys:

- `path`: output Zarr path

#### `processing.drogue`

Supported keys:

- `clip_to_drogued_period`: when true, keeps `start_date <= time < drogue_lost_date`; if no drogue-loss date exists, keeps `start_date <= time <= end_date`
- `clip_after_loss`: accepted alias for `clip_to_drogued_period`
- `minimum_length_m`
- `min_length_m`: accepted alias for `minimum_length_m`

#### `processing.regions`

Optional trajectory selection by geographical region.

Supported keys:

- `names_or_labels`
- `selection_mode`: `from_first_entry`, `full_if_enters`, or `initial_inside`
- `input_lon_mode`

#### `processing.resample`

Optional temporal resampling and interpolation.

Supported keys:

- `enabled`
- `frequency`
- `interpolate`
- `reference_time`
- `shared_time`
- `shift_start_to_reference`
- `align_start`

**Output**

Variables written:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`

`z` is set to `0.0` for all observations.

**Examples**

- Example YAML: `experiments/configs/examples/DRIFTERS/aoml_drifter_to_zarr_example.yml`
- Southern Ocean YAML: `experiments/configs/southern_ocean/AOML_drifter_to_zarr.yml`
- Example notes: `experiments/configs/examples/DRIFTERS/README.md`

---

### `convert-rafos-to-zarr`

**Purpose**

Convert AOML/WHOI RAFOS/SOFAR subsurface float NetCDF downloads into Parcels-compatible trajectory Zarr datasets.

The converter:

- reads flat tabledap-style files with a `row` dimension
- groups rows by `(floatID, trajectoryID)`
- writes `platform_code` as `floatID::trajectoryID`
- keeps `floatID`, `trajectoryID`, and `float_type` as trajectory-level variables
- copies `pressure` to `z` without pressure-to-depth conversion
- optionally clips observations with `time >= surface_date`
- optionally splits trajectories by rtraj-style depth bins and writes one Zarr per non-empty bin
- optionally filters by region and resamples/interpolates the time axis

**Run**

```bash
convert-rafos-to-zarr experiments/configs/examples/RAFOS/rafos_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.rafos_to_zarr experiments/configs/examples/RAFOS/rafos_to_zarr_example.yml
```

**Configuration file structure**

Supported top-level sections:

- `input`
- `output`
- `processing`
- `regions`
- `resample`
- `segmentation`
- `depth_bins`

`input` accepts `netcdf_files`, `rafos_files`, `netcdf_glob`, `rafos_glob`, `netcdf_dir`, `netcdf_dirs`, `rafos_dir`, `rafos_dirs`, and `pattern`.

`processing.surface.clip_after_surface_date` defaults to `true`.

`depth_bins` supports the same `enabled`, `output_mode`, `missing_depth`, `isolated_outlier`, and `bins` structure used by the staged RTRAJ converter.

**Output**

Typical variables written:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`
- `floatID`
- `trajectoryID`
- `float_type`
- `depth_bin` and `depth_bin_interval` when depth-bin output is enabled

**Examples**

- Example YAML: `experiments/configs/examples/RAFOS/rafos_to_zarr_example.yml`
- Southern Ocean YAML: `experiments/configs/southern_ocean/RAFOS_to_zarr.yml`

---

### `convert-drf-to-zarr`

**Purpose**

Convert IOS `.drf` drifter files into a Parcels-compatible trajectory Zarr dataset.

The converter:

- parses IOS header/body DRF text format
- derives `platform_code` from instrument `ID` (with filename fallback)
- filters rows by `At_Sea` flag policy (default keeps only `1`)
- optionally segments irregular trajectories by expected cadence
- optionally filters by region and resamples/interpolates the time axis
- reports input cadence diagnostics and stores summary metadata in dataset attrs

**Run**

```bash
convert-drf-to-zarr experiments/configs/examples/DRIFTERS/drf_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.drf_to_zarr experiments/configs/examples/DRIFTERS/drf_to_zarr_example.yml
```

**CLI arguments**

- `config`: path to the YAML configuration file

**Configuration file structure**

Supported top-level sections:

- `input`
- `output`
- `processing`

#### `input`

Controls which DRF files are read.

Supported keys:

- `drf_files`: explicit list of file paths
- `drf_glob`: glob expression
- `drf_dir`: input directory
- `pattern`: filename pattern used with `drf_dir` (default `*.drf`)

#### `processing.quality`

Supported keys:

- `keep_at_sea_flags`: list of integer flags to keep (default `[1]`)

#### `processing.segment`

Supported keys:

- `mode`: `ignore`, `longest`, or `split_as_new`
- `step_hours`
- `tolerance_minutes`

#### `processing.regions`

Same behavior and keys as `convert-argo-to-zarr` / `convert-drifter-to-zarr`.

#### `processing.resample`

Same behavior and keys as `convert-argo-to-zarr` / `convert-drifter-to-zarr`.

**Output**

Typical variables written:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`

Cadence diagnostics are persisted in attrs including:

- `cadence_n_trajectories`
- `cadence_mode_step_seconds`
- `cadence_common_steps_seconds`
- `cadence_step_histogram`

**Examples**

- Example YAML: `experiments/configs/examples/DRIFTERS/drf_to_zarr_example.yml`

---

### `convert-rtraj-to-zarr`

**Purpose**

Convert original ARGO Rtraj NetCDF files into Parcels-compatible trajectory Zarr datasets using the staged converter.

The converter:

- reads files named like `1902267_Rtraj.nc`
- keeps the ARGO WMO identifier in `platform_code`
- reads `JULD_ADJUSTED` where available, falling back to `JULD`
- keeps one representative trajectory fix per cycle when configured
- infers parking depth from representative pressure or pressure inside the park window
- applies QC, jump cleanup, and remaining-jump splitting
- splits by parking-depth bins before writing one Zarr per non-empty bin
- optionally filters by region and resamples/interpolates the time axis
- optionally writes raw/controlled/resampled diagnostic plots

**Run**

```bash
convert-rtraj-to-zarr experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.rtraj_to_zarr experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml
```

**CLI arguments**

- `config`: path to the YAML configuration file

**Configuration file structure**

Supported top-level sections:

- `run`
- `input`
- `output`
- `source_variables`
- `normalized_variables`
- `observations`
- `trajectory_fixes`
- `parking_depth`
- `qc`
- `segmentation`
- `jump_qc`
- `depth_bins`
- `regions`
- `resample`
- `diagnostics`

#### `run`

Controls mode and optional diagnostics subsetting.

Supported keys:

- `name`: run label
- `mode`: `diagnostics` or `convert`
- `max_files`: optional file limit intended for diagnostics subsets

#### `input`

Controls which Rtraj NetCDF files are read.

Supported keys:

- `rtraj_files`: explicit list of file paths
- `rtraj_glob`: glob expression
- `rtraj_dir`: input directory
- `pattern`: filename pattern used with `rtraj_dir`, default `*_Rtraj.nc`

#### `output`

Controls Zarr writing.

Supported keys:

- `zarr_path`: output path
- `write_zarr`: whether to write Zarr outputs
- `overwrite`: whether existing Zarr outputs may be overwritten

#### `parking_depth`

Supported keys:

- `mode`: currently only `representative_park_pressure`
- `fallback_value`: optional value used when representative or inferred parking pressure cannot be assigned
- `fill_missing`: forward/backward fill missing cycle depths from neighboring valid cycles
- `infer_from_park_window`: controls pressure inference between `JULD_PARK_START` and `JULD_PARK_END`

#### `observations`

Optionally samples original Rtraj measurements onto final resampled trajectory
points. Configure adjusted/raw sources for observation time and pressure, then
map each desired output name to its adjusted/raw Rtraj variable. Adjusted values
are preferred row by row; missing or non-finite adjusted values fall back to
raw values. `adjusted: null` is supported.

Numeric pressure and observation-variable sources optionally accept
`adjusted_qc`, `fallback_qc`, `valid_qc`, `missing_qc`, `valid_min`, and
`valid_max`. QC is read from the same adjusted or raw source selected for that
row. An explicit `"0"` flag can be accepted through `valid_qc`; it is not the
same as absent, blank, or fill-valued QC. `missing_qc` is `accept` by default
for backward compatibility and can be set to `reject`. A finite adjusted value
that fails QC or inclusive numerical bounds becomes `NaN` and does not fall
back to raw. Raw fallback is attempted only when the adjusted value itself is
missing or non-finite.

Observation time retains its existing adjusted/raw fallback behavior; these
QC and numerical-range controls apply to numeric pressure and output variables.

Sampling runs after resampling, so observed values are not interpolated. Each
variable is matched independently by same depth bin, closest time, closest
representative parking pressure, then original measurement index. There is no
pressure tolerance. `sample_at_fallback_depth` defaults to `false`, which skips
points whose parking pressure came from `parking_depth.fallback_value` and
leaves their configured observations as `NaN`. Set it to `true` to match those
points using the fallback pressure, assigned depth bin, and normal ranking
order.

Adding another variable requires only another `observations.variables` YAML
entry. Observation time and pressure remain internal matching fields. The
controlled-stage CSV records per-file observation filtering counts, and the
diagnostics summary aggregates adjusted/raw acceptance, QC rejection,
missing-QC rejection, and numerical-range rejection.

#### `qc`

Controls the first segmentation/filtering rule based on configured QC variables.

#### `segmentation.merge`

Controls whether compatible QC-separated or post-jump segments are reconnected.

Supported keys include:

- `enabled`
- `max_gap_points`
- `max_gap_duration_days`
- `max_bridge_speed_m_per_s`
- `max_bridge_vertical_rate_m_per_day`

Null thresholds are skipped.

#### `jump_qc`

Controls trajectory jump cleanup after QC segmentation.

The converter can:

- drop isolated spike points when the bridge is physically plausible
- drop short bad-location blocks when the bridge is physically plausible
- split any remaining non-physical jumps

#### `depth_bins`

Optional contiguous splitting by mapped parking pressure.

Supported keys:

- `enabled`
- `output_mode`: currently `per_bin`
- `bins`: list of `label`, `min`, and `max` mappings
- `missing_depth`: short missing-bin fill rules
- `isolated_outlier`: short isolated-bin repair rules

When depth bins are enabled with `output_mode: per_bin`, the converter writes one Zarr per non-empty bin.

#### `regions`

Optional trajectory selection by geographical region.

Supported keys:

- `names_or_labels`
- `selection_mode`: `from_first_entry`, `full_if_enters`, or `initial_inside`
- `input_lon_mode`

QC, jump cleanup, and depth-bin splitting run before this region stage.

#### `resample`

Optional temporal resampling and interpolation.

Supported keys:

- `enabled`
- `frequency`
- `interpolate`
- `reference_time`
- `shared_time`
- `shift_start_to_reference`
- `min_duration_days`

#### `diagnostics`

Controls summary and plot outputs.

Supported keys:

- `output_dir`
- `plots.raw_vs_qc_segments`
- `formats`

The diagnostic plot has raw, controlled, and resampled panels.

**Output**

Typical variables written:

- `time`
- `lon`
- `lat`
- `z`
- `platform_code`
- `depth_bin` and `depth_bin_interval` when depth-bin output is enabled
- configured observation variables such as `temp` and `psal`

**Examples**

- Example YAML: `experiments/configs/examples/Rtraj/rtraj_to_zarr_example.yml`
- Southern Ocean YAML: `experiments/configs/southern_ocean/RTRAJ_to_zarr.yml`
- Example notes: `experiments/configs/examples/Rtraj/README.md`
- Legacy example YAML: `experiments/configs/examples/Rtraj/legacy/rtraj_to_zarr_example.yml`

The previous converter implementation is preserved under `kinematicparcels.tools.legacy.rtraj_to_zarr` for reproducibility, but the legacy CLI entry point has been retired.

---

### `couple-trajectories`

**Purpose**

Build grouped-entity pair trajectories from a Parcels-compatible input Zarr dataset.

The tool:

- reads an input trajectory Zarr
- synchronizes candidate pairs on absolute `time`
- computes pair distance on the shared time overlap
- accepts pairs that satisfy the distance threshold, and optionally a minimum post-closest-approach lifetime
- optionally requires the threshold crossing to occur inside one or more target regions
- writes one grouped-entity trajectory per accepted pair with `group_size = 2`

This tool is designed for datasets that have already been temporally regularized upstream when needed. It does not perform resampling itself.

**Run**

```bash
couple-trajectories INPUT.zarr OUTPUT.zarr --threshold-km 50
```

Example with optional filters:

```bash
couple-trajectories INPUT.zarr OUTPUT.zarr --threshold-km 50 --minimum-life-days 15 --regions med_cpf sesc
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.couple_trajectories INPUT.zarr OUTPUT.zarr --threshold-km 50
```

**CLI arguments**

- `input_zarr`: input Parcels-compatible trajectory Zarr
- `output_zarr`: output grouped-entity Zarr
- `--threshold-km`: required maximum allowed closest-approach distance
- `--minimum-life-days`: optional minimum duration after the selected closest-approach point
- `--regions`: optional list of region labels or names

**Acceptance logic without regions**

If `--regions` is not provided, a pair is accepted when:

- the two trajectories share absolute timestamps
- at least two synchronized observations exist
- the minimum pair distance over the overlap satisfies `min(distance) <= threshold_km`
- if `--minimum-life-days` is set, the pair segment from closest approach onward lasts at least that long

The output pair starts at the global closest-approach point.

**Acceptance logic with regions**

If `--regions` is provided, the rule becomes stricter:

1. the tool first finds synchronized points where `distance <= threshold_km`
2. only those points are checked against the requested regions
3. the pair is accepted only if at least one of those below-threshold points lies inside one of the target regions
4. if several points match, the pair starts at the one with minimum distance among the in-region matches

The current region test is applied to the pair center, which is consistent with the grouped-entity output written by the tool.

**Output**

Each accepted pair becomes one grouped-entity trajectory with variables such as:

- `time`
- `lon`, `lat` as the pair center
- `center_lon`, `center_lat`
- `lon_1`, `lat_1`
- `lon_2`, `lat_2`
- `group_id`
- `group_size`
- `z`

If the input dataset contains `platform_code`, the output also stores:

- `platform_code_1`
- `platform_code_2`

**Metadata written to dataset attrs**

- `source = "Trajectory pair coupling"`
- `pair_threshold_km`
- `minimum_life_days` when provided
- `pair_regions` when regions are provided

**Assumptions**

- pair synchronization is done on absolute `time`, not on `obs`
- if the input cadence should be regular, that resampling must already have happened upstream

---

## Internal Helper Modules

These modules support the CLI tools but are not intended to be run directly by users.

### `zarr_writer.py`

Shared low-level helper for building Parcels-style `trajectory x obs` Zarr datasets from lists of trajectory tables.

It currently provides:

- `build_dataset_from_trajectories(...)`
- `build_zarr_encoding(...)`

This module exists to avoid duplicating the same writer logic in both `_to_zarr.py` converters and `couple_trajectories.py`.

---

## Development / Inspection Scripts

These files are currently useful for quick checks, but they are not packaged as stable command-line tools.

### `check_argo_data.py`

Ad hoc plotting script for quick local inspection of raw ARGO CSV content.

Current characteristics:

- uses hardcoded local file paths
- generates quick plots with `matplotlib`
- is intended for manual inspection only

It should not be considered part of the stable public tool interface in its current form.

---

## Current Tool Inventory

Under `src/kinematicparcels/tools` the current files are:

- `argo_to_zarr.py`: user-facing CLI tool
- `drifter_to_zarr.py`: user-facing CLI tool
- `aoml_drifter_to_zarr.py`: user-facing CLI tool
- `drf_to_zarr.py`: user-facing CLI tool
- `rtraj_to_zarr.py`: user-facing ARGO Rtraj CLI tool
- `legacy/rtraj_to_zarr.py`: legacy ARGO Rtraj converter
- `couple_trajectories.py`: user-facing CLI tool
- `trajectory_processing.py`: internal shared trajectory filtering/resampling helper
- `zarr_writer.py`: internal shared helper
- `check_argo_data.py`: development inspection script
- `__init__.py`: package marker
