# DRIFTERS Examples

This folder contains example configurations for drifter converters.

## AOML GDP 6-hour NetCDF

Use `aoml_drifter_to_zarr_example.yml` for the AOML Global Drifter Program 6-hour NetCDF files named like `drifter_6h_XXXX.nc`.

The script is implemented in `src/kinematicparcels/tools/aoml_drifter_to_zarr.py`. It converts the AOML GDP interpolated 6-hour NetCDF archive into the Parcels-compatible `trajectory x obs` Zarr layout used by the postprocessing tools.

Run:

```bash
convert-aoml-drifter-to-zarr experiments/configs/examples/DRIFTERS/aoml_drifter_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.aoml_drifter_to_zarr experiments/configs/examples/DRIFTERS/aoml_drifter_to_zarr_example.yml
```

The converter supports either a single glob:

```yaml
input:
  drifter_glob: F:/DRIFTERS/netcdf_6h/netcdf_*/drifter_6h_*.nc
```

or explicit archive directories:

```yaml
input:
  drifter_dirs:
    - F:/DRIFTERS/netcdf_6h/netcdf_1_5000
    - F:/DRIFTERS/netcdf_6h/netcdf_5001_15000
    - F:/DRIFTERS/netcdf_6h/netcdf_10001_15000
    - F:/DRIFTERS/netcdf_6h/netcdf_15001_current
  pattern: "drifter_6h_*.nc"
```

Processing behavior:

- uses `ID` as trajectory-level `platform_code`
- drops `WMO` while opening the NetCDF file, because `ID` is the stable identifier and `WMO` can trigger xarray fill-value warnings
- filters by file metadata `DrogueLength`
- clips to `start_date <= time < drogue_lost_date` when `drogue_lost_date` exists
- otherwise clips to `start_date <= time <= end_date`
- writes only Parcels essentials: `time`, `lon`, `lat`, `z`, and `platform_code`
- applies the shared region selection and resampling pipeline used by the ARGO RTRAJ converter

Supported YAML sections:

- `input`: accepts `drifter_glob`, `netcdf_glob`, `drifter_files`, `netcdf_files`, `drifter_dir`, `drifter_dirs`, `netcdf_dir`, `netcdf_dirs`, and `pattern`
- `output.path`: destination Zarr path
- `processing.drogue.minimum_length_m`: minimum accepted `DrogueLength`
- `processing.drogue.clip_to_drogued_period`: clips observations to the drogued interval
- `processing.regions`: same `names_or_labels`, `selection_mode`, and `input_lon_mode` options as the RTRAJ converter
- `processing.resample`: same shared resampling options as the RTRAJ converter

For the Southern Ocean setup, use:

```bash
convert-aoml-drifter-to-zarr experiments/configs/southern_ocean/AOML_drifter_to_zarr.yml
```

## CSV and DRF

Use `drifter_to_zarr_example.yml` for CSV drifter files and `drf_to_zarr_example.yml` for IOS `.drf` files.
