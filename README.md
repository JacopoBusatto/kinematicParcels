# kinematicParcels

Lightweight framework to run **Lagrangian particle experiments with OceanParcels** using YAML-driven configuration files.

The goal of this project is to **separate experiment configuration from simulation logic**, allowing multiple experiments to be run without creating many different Python scripts.

All experiment parameters are defined in YAML files:
- release region
- initial particle grid
- vertical levels
- velocity fields
- simulation duration
- integration timestep
- output timestep

The framework provides a **generic experiment runner** that reads the configuration file and executes the simulation.

---

# Project Structure

kinematicParcels/

├── src/

│   └── kinematicparcels/

│       ├── __init__.py

│       ├── runner/

│       │   ├── __init__.py

│       │   ├── run_experiment.py

│       │   └── run_experiment_series.py

│       ├── postprocessing/

│       │   ├── __init__.py

│       │   ├── analyses/

│       │   │   ├── __init__.py

│       │   │   ├── beaching_times.py

│       │   │   ├── density.py

│       │   │   └── start_end_regions.py

│       │   ├── animations/

│       │   │   ├── __init__.py

│       │   │   ├── density.py

│       │   │   ├── trajectories.py

│       │   │   └── utils.py

│       │   ├── config/

│       │   │   ├── __init__.py

│       │   │   ├── loader.py

│       │   │   └── models.py

│       │   ├── core/

│       │   │   ├── __init__.py

│       │   │   ├── filters.py

│       │   │   ├── gridding.py

│       │   │   └── summaries.py

│       │   ├── io/

│       │   │   ├── __init__.py

│       │   │   ├── exports.py

│       │   │   └── parcels.py

│       │   ├── plotting/

│       │   │   ├── __init__.py

│       │   │   ├── maps.py

│       │   │   ├── projections.py

│       │   │   └── trajectories.py

│       │   ├── runner/

│       │   │   ├── __init__.py

│       │   │   ├── cli.py

│       │   │   ├── run_postprocessing.py

│       │   │   └── run_postprocessing_series.py

│       │   └── workflows/

│       │       ├── __init.__.py

│       │       ├── base_products.py

│       │       ├── quicklook.py

│       │       ├── run_beaching_times.py

│       │       ├── run_density.py

│       │       ├── run_start_end_regions.py

│       │       ├── run_summary.py

│       │       └── run_trajectories.py

│       └── utilities/

│           ├── __init__.py

│           ├── compare_region_shape.py

│           ├── geographicalRegions.py

│           ├── geographicalRegions_rectangles.py

│           ├── init_checks.py

│           ├── init_depths.py

│           └── regions_definitions.py

│

├── experiments/

│   └── configs/

│       ├── exp_NPstg_surface.yml

│       ├── exp_NPstg_multilevel.yml

│       ├── fjords_template_PFunion.yml

│       ├── fjords_template_postprocessing.yml

│       ├── fjords_test_1p.yml

│       ├── fjords_test_PFunion.yml

│       ├── multiple_simulation_postprocessing.yml

│       └── multiple_simulation_run.yml

│

├── fields/          (input velocity fields, not tracked)

├── outputs/         (simulation outputs, not tracked)

├── logs/            (optional run logs)

├── testing_utils/

│   ├── interpolate_field_over_trajectory.py

│   ├── test.py

│   └── test_shapes.py

├── environment.yml

├── pyproject.toml

├── LICENSE.md

├── POSTPROCESSING.md

├── .gitignore

└── README.md

experiments/configs  
YAML configuration files describing experiments.

fields  
Input velocity fields (NetCDF). Not versioned in Git.

outputs  
Simulation outputs (Zarr). Not versioned in Git.

---

# Installation

Clone the repository:

git clone https://github.com/JacopoBusatto/kinematicParcels.git

cd kinematicParcels


Create the Conda environment:

conda env create -f environment.yml

Activate it:

conda activate parcels


Install the package in editable mode:

pip install -e .

This allows modifying the code without reinstalling the package.

---

# Running an Experiment

Experiments are defined through YAML configuration files.

Example configuration:

experiments/configs/exp_NPstg_surface.yml

Run the simulation with:

run-parcels-experiment experiments/configs/exp_NPstg_surface.yml

Example output:

Experiment: NPstg_surface_test  
Found 31 input files  
ParticleSet created with 192 particles  
Run completed: outputs/output_NPstg_surface.zarr

---

# Alternative Execution Method

The experiment runner can also be executed as a Python module:

python -m kinematicparcels.runner.run_experiment experiments/configs/exp_NPstg_surface.yml

---

# YAML Configuration Example

experiment:
  name: NPstg_surface_test
  output_dir: ./outputs

fieldset:
  file_pattern: ./fields/*.nc
  variables:
    U: ugos
    V: vgos
  dimensions:
    lon: longitude
    lat: latitude
    time: time
  mesh: spherical

release:
  region_label: NPstg
  dlon: 5
  dlat: 5
  filter_domain: true

simulation:
  runtime_days: 10
  dt_hours: 1
  outputdt_hours: 24
  particle_type: scipy

output:
  zarr_name: output_NPstg_surface.zarr

---

---

# Supported Velocity Field Grids

The framework supports both **regular lat/lon grids** and **curvilinear grids**.
Velocity files can be provided as a single NetCDF file or as a time series
using wildcards:

file_pattern: ./fields/*.nc

### Regular Grid Example

Typical global reanalysis products:

```yaml
dimensions:
  lon: longitude
  lat: latitude
  time: time
```

Velocity variables must have dimensions:

```
(time, lat, lon)
```

---

### Curvilinear Grid Example

Regional ocean models (e.g. **ROMS**, **OpenDrift-ready fields**) often use
curvilinear grids where longitude and latitude are **2-D coordinates**.

Example dataset structure:

```
Dimensions:
(time, depth, xi_rho, eta_rho)

Coordinates:
lon_rho(xi_rho, eta_rho)
lat_rho(xi_rho, eta_rho)
```

Configuration example:

```yaml
fieldset:
  file_pattern: ./fields/patagonia.nc

  variables:
    U: x_sea_water_velocity
    V: y_sea_water_velocity

  dimensions:
    lon: lon_rho
    lat: lat_rho
    time: time
    depth: depth

  mesh: spherical
```

The runner automatically builds a **CurvilinearZGrid** in Parcels.

---

# Longitude Convention Handling

Velocity fields may use different longitude conventions:

```
[-180 , 180]
or
[0 , 360]
```

The framework **automatically detects the longitude convention of the fieldset**
and converts release grids accordingly.

Example log:

```
[fieldset] detected longitude mode: -180_180
[release] fieldset longitude mode detected: -180_180
```

This avoids mismatches when running experiments across the dateline.

No configuration parameter is required.

---

# Periodic Halo

For global simulations where particles may cross the dateline, the fieldset can
be extended with a periodic halo.

Example configuration:

```yaml
fieldset:
  periodic_halo: true
  periodic_halo_size: 5
```

When enabled the runner executes:

```
fieldset.add_periodic_halo(zonal=True, halosize=5)
```

This is recommended for global ocean fields.

---

# Domain Filtering

Initial particle positions are automatically checked against the velocity field
domain.

Example log:

```
[domain check]
fieldset lon = [-73.659, -72.453], lat = [-52.503, -51.414]
points total = 25
points inside = 25
points outside = 0
```

If `filter_domain: true`, points outside the field domain are removed before the
ParticleSet is created.

---

# Output Logs

During execution the runner prints diagnostic information such as:

- detected longitude convention
- number of particles released
- domain filtering statistics
- simulation progress

Example:

```
ParticleSet created with 25 particles
INFO: Output files are stored in outputs/output_PNnf_surface.zarr.
```

---

# Time Domain Limitation

The simulation runtime must be compatible with the temporal extent of the
velocity fields.

If the runtime exceeds the available data, Parcels raises:

```
TimeExtrapolationError
```

Solutions:

- reduce `runtime_days`
- provide more velocity files
- enable time extrapolation (not recommended for physical simulations)

---

# Multi-Level Particle Release

Particles can be released at multiple depths.

Example configuration:

depth:
  enabled: true
  values: [0, 50, 100]
  mode: snap_to_field
  request_convention: positive_down
  snap_method: nearest
  remove_duplicate_depths: true

Available modes:

as_requested  
Uses the exact requested depths.

snap_to_field  
Adapts release depths to the vertical levels of the velocity field.

If multiple requested depths collapse onto the same field level, duplicates can be automatically removed.

---

# Utilities

geographicalRegions.py  
Defines geographic regions and builds regular release grids.

init_checks.py  
Checks initial particle positions and removes points outside the velocity field domain.

init_depths.py  
Handles vertical coordinates, snapping to field levels, and duplicate removal.

---

# Output Format

Simulation outputs are stored as **Zarr datasets**.

Example:

outputs/output_experiment.zarr

Load them with xarray:

import xarray as xr

ds = xr.open_zarr("outputs/output_experiment.zarr")

---

# Running Multiple Simulations (Series Mode)

In addition to single experiments, the framework supports launching **multiple simulations in sequence** using a master configuration file.

This is useful when running:

* multiple start dates
* sensitivity experiments
* batch simulations for statistics

---

## Concept

The workflow is based on two layers:

1. A **template YAML** (single experiment)
2. A **master YAML** defining the time schedule

The system automatically:

* generates one YAML per simulation
* creates a dedicated output folder for each run
* launches simulations sequentially

---

## Master Configuration Example

```
template_config: .\experiments\configs\fjords_01.yml

series:
  output_root: C:/Users/Jacopo/Documents/DATI/PATAGONIA/simulation_series

  start_time: "2026-01-01 00:00"
  frequency: "1D"
  duration: "10D"

  output_subdir_format: "%Y%m%d-%H%M"
  config_filename: "experiment.yml"
  runner_exe: "run-parcels-experiment.exe"
```

---

## Generated Structure

```
simulation_series/
  20260101-0000/
    experiment.yml
    output.zarr
  20260102-0000/
    experiment.yml
    output.zarr
  ...
```

Each run is:

* isolated
* reproducible
* fully defined by its local YAML file

---

## Running the Series

Generate configurations only:

```
python run_experiment_series.py master_series.yml --generate-only
```

Generate and execute:

```
python run_experiment_series.py master_series.yml
```

---

## Start Time Handling

The single experiment YAML supports:

```
simulation:
  start_time: "2026-01-01 00:00"
```

This defines the release time of all particles.

If not provided, the simulation behaves as before.

---

## Design Philosophy

This approach ensures:

* full reproducibility (each run has its own YAML)
* easy debugging (runs are independent)
* compatibility with HPC workflows
* clean separation between orchestration and simulation logic

---

# Project Goals

The framework aims to:

- simplify the configuration of Lagrangian experiments
- improve reproducibility
- avoid duplication of simulation scripts
- provide modular utilities for particle initialization

---

# Future Extensions

Possible future developments:

- time-dependent particle release
- custom Parcels kernels
- automatic trajectory plotting
- integration with HPC batch systems
- additional command-line tools

---

# Author

Jacopo Busatto, PhD
CNR ISMAR, Rome, Italy
jacopobusatto@cnr.it

---

# License

MIT License