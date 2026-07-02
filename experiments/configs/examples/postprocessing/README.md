# Postprocessing Examples

This folder contains one self-contained YAML example for each supported postprocessing analysis:

- `01_summary.yml`
- `02_trajectories.yml`
- `03_density.yml`
- `04_beaching_times.yml`
- `05_fsle.yml`
- `06_start_end_regions.yml`
- `07_meridional_crossing.yml`
- `08_transition_probability.yml`
- `09_exponent_maps.yml`
- `10_cluster_strength.yml`
- `11_meridional_excursion.yml`

Each file is intended as an option reference rather than a minimal runnable config.
The shared sections (`dataset`, `analysis`, `output`, `exports`, `cleaning`, `release`, `plotting`, and `grid` where relevant) are kept explicit so a user can copy a single file and edit it locally.

The examples use repository-relative sample paths where practical, but you should still adjust `dataset.input_path` and `output.output_dir` to match your local run.
