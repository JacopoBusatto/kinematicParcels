from __future__ import annotations

import argparse
from datetime import timedelta
from glob import glob
from pathlib import Path
import warnings

import numpy as np
import yaml

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r".*parcels\.particledata",
)
warnings.filterwarnings(
    "ignore",
    message=r".*where.*without 'out'.*",
    category=UserWarning,
)

from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4

from kinematicparcels.utilities.geographicalRegions import (
    get_region_by_label,
    make_regular_grid_in_region,
)
from kinematicparcels.utilities.init_checks import (
    summarize_initial_points,
    check_initial_points_in_domain,
    filter_inside_domain,
)
from kinematicparcels.utilities.init_depths import (
    summarize_depth_axis,
    build_multilevel_release,
)


# -----------------------------------------------------------------------------
# Helpers
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
    raise ValueError(f"Unsupported particle_type: {name}")


def build_fieldset(cfg: dict) -> FieldSet:
    fs_cfg = cfg["fieldset"]

    files = sorted(glob(fs_cfg["file_pattern"]))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found with pattern: {fs_cfg['file_pattern']}")

    print(f"Found {len(files)} input files")
    for f in files[:3]:
        print(" ", f)

    variables = fs_cfg["variables"]

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


def _build_release_points_from_region(rel_cfg: dict):
    region = get_region_by_label(rel_cfg["region_label"])
    lons2d, lats2d = make_regular_grid_in_region(
        region,
        dlon=rel_cfg["dlon"],
        dlat=rel_cfg["dlat"],
    )
    return lons2d, lats2d


def _build_release_points_from_list(rel_cfg: dict):
    points = rel_cfg.get("points", [])
    if len(points) == 0:
        raise ValueError("release.points is empty")

    lons = []
    lats = []

    for i, p in enumerate(points):
        if isinstance(p, dict):
            if "lon" not in p or "lat" not in p:
                raise ValueError(f"Point #{i} must contain 'lon' and 'lat'")
            lons.append(float(p["lon"]))
            lats.append(float(p["lat"]))
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            lons.append(float(p[0]))
            lats.append(float(p[1]))
        else:
            raise ValueError(
                f"Point #{i} must be either {{lon: ..., lat: ...}} or [lon, lat]"
            )

    return np.asarray(lons), np.asarray(lats)


def build_release(cfg: dict, fieldset: FieldSet):
    rel_cfg = cfg["release"]
    release_mode = rel_cfg.get("mode", "region_grid")

    if release_mode == "region_grid":
        lons_raw, lats_raw = _build_release_points_from_region(rel_cfg)

    elif release_mode == "point_list":
        lons_raw, lats_raw = _build_release_points_from_list(rel_cfg)

    else:
        raise ValueError(
            f"Unsupported release.mode: {release_mode}. "
            "Use 'region_grid' or 'point_list'."
        )

    summarize_initial_points(lons_raw, lats_raw, name="raw release points")

    if rel_cfg.get("filter_domain", True):
        check_initial_points_in_domain(lons_raw, lats_raw, fieldset, verbose=True)
        lons_raw, lats_raw = filter_inside_domain(lons_raw, lats_raw, fieldset)
        summarize_initial_points(lons_raw, lats_raw, name="filtered release points")

    depth_cfg = rel_cfg.get("depth", {})
    use_depth = depth_cfg.get("enabled", False)

    if use_depth:
        summarize_depth_axis(fieldset)

        lons, lats, depths = build_multilevel_release(
            lons2d=lons_raw,
            lats2d=lats_raw,
            requested_depths=depth_cfg["values"],
            fieldset=fieldset,
            depth_mode=depth_cfg.get("mode", "as_requested"),
            request_convention=depth_cfg.get("request_convention", "positive_down"),
            snap_method=depth_cfg.get("snap_method", "nearest"),
            remove_duplicate_depths=depth_cfg.get("remove_duplicate_depths", True),
            verbose=True,
        )
        return lons, lats, depths

    return lons_raw, lats_raw, None


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
    print(f"ParticleSet created with {len(lons)} particles")
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

    print(f"Run completed: {zarr_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generic Parcels experiment runner from YAML config"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Experiment: {cfg['experiment']['name']}")

    fieldset = build_fieldset(cfg)
    lons, lats, depths = build_release(cfg, fieldset)
    pset = build_particleset(cfg, fieldset, lons, lats, depths=depths)
    run_simulation(cfg, pset)


if __name__ == "__main__":
    main()