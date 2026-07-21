# RAFOS Examples

This folder contains example configurations for RAFOS/SOFAR subsurface float conversion.

Use `rafos_to_zarr_example.yml` for the AOML/WHOI `RAFOS_SOFAR_Floats` tabledap NetCDF download.

Run:

```bash
convert-rafos-to-zarr experiments/configs/examples/RAFOS/rafos_to_zarr_example.yml
```

Equivalent module form:

```bash
python -m kinematicparcels.tools.rafos_to_zarr experiments/configs/examples/RAFOS/rafos_to_zarr_example.yml
```

Processing behavior:

- groups rows by the pair `(floatID, trajectoryID)`, because `floatID` is a serial number and `trajectoryID` is the experiment-specific float identifier
- writes `platform_code` as `floatID::trajectoryID`
- keeps `floatID`, `trajectoryID`, and `float_type` as trajectory-level variables
- copies `pressure` to `z` without pressure-to-depth conversion
- optionally clips observations with `time >= surface_date`
- optionally splits output into rtraj-style depth-bin Zarr datasets

For the Southern Ocean setup, use:

```bash
convert-rafos-to-zarr experiments/configs/southern_ocean/RAFOS_to_zarr.yml
```
