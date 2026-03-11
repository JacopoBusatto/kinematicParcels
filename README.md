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

│       │   └── run_experiment.py

│       └── utilities/

│           ├── geographicalRegions.py

│           ├── init_checks.py

│           └── init_depths.py

│

├── experiments/

│   └── configs/

│       ├── exp_NPstg_surface.yml

│       └── exp_NPstg_multilevel.yml

│

├── fields/          (input velocity fields, not tracked)

├── outputs/         (simulation outputs, not tracked)

├── logs/            (optional run logs)

│

├── environment.yml

├── pyproject.toml

├── .gitignore

└── README.md


Explanation of the main directories:

src/kinematicparcels  
Contains the installable Python package.

runner  
Contains the generic experiment runner.

utilities  
Helper modules for particle initialization and domain checks.

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

Jacopo Busatto  
CNR ISMAR

---

# License

MIT License