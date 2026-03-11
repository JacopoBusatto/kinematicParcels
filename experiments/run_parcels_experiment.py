from __future__ import annotations
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"parcels\.particledata"
)
import argparse
from datetime import timedelta
from glob import glob
from pathlib import Path
import warnings

import yaml
import numpy as np

from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4

from utilities.geographicalRegions import get_region_by_label, make_regular_grid_in_region
from utilities.init_checks import (
    summarize_initial_points,
    check_initial_points_in_domain,
    filter_inside_domain,
)
from utilities.init_depths import (
    summarize_depth_axis,
    build_multilevel_release,
)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_particle_class(name: str):
    name = name.lower()
    if name == "scipy":
        return ScipyParticle
    if name == "jit":
        return JITParticle
    raise ValueError(f"particle_type non supportato: {name}")


def build_fieldset(cfg: dict) -> FieldSet:
    fs_cfg = cfg["fieldset"]

    files = sorted(glob(fs_cfg["file_pattern"]))
    if len(files) == 0:
        raise FileNotFoundError(f"Nessun file trovato con pattern: {fs_cfg['file_pattern']}")

    print(f"Trovati {len(files)} file di input")
    for f in files[:3]:
        print(" ", f)

    variables = fs_cfg["variables"]

    # mapping Parcels: per U e V separati
    filenames = {
        "U": files,
        "V": files,
    }

    dims_cfg = fs_cfg["dimensions"]

    dimensions = {
        "U": {
            "lon": dims_cfg["lon"],
            "lat": dims_cfg["lat"],
            "time": dims_cfg["time"],
        },
        "V": {
            "lon": dims_cfg["lon"],
            "lat": dims_cfg["lat"],
            "time": dims_cfg["time"],
        },
    }

    if "depth" in dims_cfg:
        dimensions["U"]["depth"] = dims_cfg["depth"]
        dimensions["V"]["depth"] = dims_cfg["depth"]

    fieldset = FieldSet.from_netcdf(
        filenames=filenames,
        variables=variables,
        dimensions=dimensions,
        mesh=fs_cfg.get("mesh", "spherical"),
    )

    print(fieldset)
    return fieldset


def build_release(cfg: dict, fieldset: FieldSet):
    rel_cfg = cfg["release"]

    region = get_region_by_label(rel_cfg["region_label"])
    lons2d, lats2d = make_regular_grid_in_region(
        region,
        dlon=rel_cfg["dlon"],
        dlat=rel_cfg["dlat"],
    )

    summarize_initial_points(lons2d, lats2d, name="raw release grid")

    if rel_cfg.get("filter_domain", True):
        check_initial_points_in_domain(lons2d, lats2d, fieldset, verbose=True)
        lons2d, lats2d = filter_inside_domain(lons2d, lats2d, fieldset)
        summarize_initial_points(lons2d, lats2d, name="filtered release grid")

    depth_cfg = rel_cfg.get("depth", {})
    use_depth = depth_cfg.get("enabled", False)

    if use_depth:
        summarize_depth_axis(fieldset)

        lons, lats, depths = build_multilevel_release(
            lons2d=lons2d,
            lats2d=lats2d,
            requested_depths=depth_cfg["values"],
            fieldset=fieldset,
            depth_mode=depth_cfg.get("mode", "as_requested"),
            request_convention=depth_cfg.get("request_convention", "positive_down"),
            snap_method=depth_cfg.get("snap_method", "nearest"),
            remove_duplicate_depths=depth_cfg.get("remove_duplicate_depths", True),
            verbose=True,
        )
        return lons, lats, depths

    return lons2d, lats2d, None


def build_particleset(cfg: dict, fieldset: FieldSet, lons, lats, depths=None):
    sim_cfg = cfg["simulation"]
    pclass = get_particle_class(sim_cfg.get("particle_type", "scipy"))

    kwargs = dict(
        fieldset=fieldset,
        pclass=pclass,
        lon=lons,
        lat=lats,
    )

    if depths is not None:
        kwargs["depth"] = depths

    pset = ParticleSet.from_list(**kwargs)
    print(f"ParticleSet creato con {len(lons)} particelle")
    return pset


def run_simulation(cfg: dict, pset: ParticleSet):
    sim_cfg = cfg["simulation"]
    out_cfg = cfg["output"]
    exp_cfg = cfg["experiment"]

    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = output_dir / out_cfg["zarr_name"]

    output_file = pset.ParticleFile(
        name=str(zarr_path),
        outputdt=timedelta(hours=sim_cfg["outputdt_hours"]),
    )

    pset.execute(
        AdvectionRK4,
        runtime=timedelta(days=sim_cfg["runtime_days"]),
        dt=timedelta(hours=sim_cfg["dt_hours"]),
        output_file=output_file,
    )

    print(f"Run completato: {zarr_path}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", message=".*where used without 'out'.*")

    parser = argparse.ArgumentParser(description="Generic Parcels experiment runner from YAML config")
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Esperimento: {cfg['experiment']['name']}")

    fieldset = build_fieldset(cfg)
    lons, lats, depths = build_release(cfg, fieldset)
    pset = build_particleset(cfg, fieldset, lons, lats, depths=depths)
    run_simulation(cfg, pset)


if __name__ == "__main__":
    main()