# Example Configurations

This folder contains self-contained YAML examples for the main experiment configurations.

## Suggested Coverage

- 01_point_list_single_2d.yml
- 02_point_list_grouped_2d.yml
- 03_region_grid_single_2d.yml
- 04_region_grid_grouped_2d.yml
- 05_circle_uniform_2d.yml
- 06_circle_gaussian_2d.yml
- 07_lkm_grouped_2d.yml
- 08_circle_multistart_2d.yml
- ARGO/argo_to_zarr_example.yml
- DRIFTERS/drifter_to_zarr_example.yml
- postprocessing/README.md

## Notes

- Keep each example minimal and runnable.
- Use explicit comments only where behavior is non-obvious.
- Prefer test_fields/ for reproducible examples in this repository.
- Region-grid examples use `region_label: med_cpf`.
- Scheduled-release examples show `release.continuous.max_age` both for classic continuous releases and for scheduled circle releases.
- The ARGO example documents an external-data workflow, so its input path is a placeholder that should be adjusted locally.
- The drifter example documents an external-data workflow, so its input path is a placeholder that should be adjusted locally.
- The `postprocessing/` subfolder is intentionally more exhaustive: it is a reference set for available postprocessing options, not a minimal experiment example set.
