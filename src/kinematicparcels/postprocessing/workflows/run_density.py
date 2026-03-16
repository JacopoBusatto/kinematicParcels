from __future__ import annotations

from pathlib import Path

from ..config.models import PostprocessConfig
from ..io import load_trajectory_table, save_dataset_netcdf, save_grid_table
from ..core import RegularGrid
from ..analyses import compute_time_density


def run_density(cfg: PostprocessConfig, context: dict) -> None:
    """
    Density workflow.
    """
    if "trajectory_table" not in context:
        print("Loading trajectory table")

        df = load_trajectory_table(
            cfg.dataset.input_path,
            truncate_stagnant=cfg.cleaning.truncate_stagnant,
            stagnant_tol=cfg.cleaning.stagnant_tol,
            stagnant_min_consecutive=cfg.cleaning.stagnant_min_consecutive,
        )

        context["trajectory_table"] = df

    df = context["trajectory_table"]

    if "grid" not in context:
        print("Building grid")

        g = cfg.grid
        if g is None:
            raise ValueError("Density analysis requires a 'grid' section in the config.")

        if g.mode == "explicit_edges":
            grid = RegularGrid(
                lon_min=g.lon_min,
                lon_max=g.lon_max,
                lat_min=g.lat_min,
                lat_max=g.lat_max,
                dlon=g.dlon,
                dlat=g.dlat,
            )

        elif g.mode == "from_initial_centers":
            time_col = cfg.density.time_col
            if time_col not in df.columns:
                raise KeyError(
                    f"Time column '{time_col}' required to build grid from initial centers."
                )

            t0 = df[time_col].min()
            df0 = df.loc[df[time_col] == t0].copy()

            if df0.empty:
                raise ValueError("Cannot build grid from initial centers: no points at first time.")

            grid = RegularGrid.from_aligned_initial_centers(
                df0[cfg.density.lon_col],
                df0[cfg.density.lat_col],
                lon_min=g.lon_min,
                lon_max=g.lon_max,
                lat_min=g.lat_min,
                lat_max=g.lat_max,
                dlon=g.dlon,
                dlat=g.dlat,
            )

        else:
            raise ValueError(f"Unsupported grid mode: {g.mode}")

        context["grid"] = grid

    print("Computing density")

    density_table, density_ds = compute_time_density(
        df,
        grid=grid,
        lon_col=cfg.density.lon_col,
        lat_col=cfg.density.lat_col,
        time_col=cfg.density.time_col,
        normalize_active=cfg.density.normalize_active,
        normalize_total=cfg.density.normalize_total,
    )

    outdir = Path(cfg.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid_table_path = outdir / f"density_table.{cfg.exports.table_format}"
    density_nc_path = outdir / "density.nc"

    print("Saving density table:", grid_table_path)
    save_grid_table(
        density_table,
        grid_table_path,
        format=cfg.exports.table_format,
    )

    print("Saving density dataset:", density_nc_path)
    save_dataset_netcdf(density_ds, density_nc_path)