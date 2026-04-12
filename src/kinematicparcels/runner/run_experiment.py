from __future__ import annotations

import argparse
from datetime import timedelta, datetime
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

from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4, Variable
from parcels.tools.statuscodes import StatusCode

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
from kinematicparcels.utilities.group_expansion import expand_groups


# ============================================================================
# Custom Particle Classes with Grouped-Release Metadata Variables
# ============================================================================
class ScipyParticleGrouped(ScipyParticle):
    """ScipyParticle with group_id, group_member, group_size variables."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)


class JITParticleGrouped(JITParticle):
    """JITParticle with group_id, group_member, group_size variables."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)


# ============================================================================
# Helpers
# ============================================================================
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_particle_class(name: str):
    name = name.lower()
    if name == "scipy":
        return ScipyParticleGrouped
    if name == "jit":
        return JITParticleGrouped
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


def DeleteErrorParticle(particle, fieldset, time):
    if particle.state in (
        StatusCode.ErrorOutOfBounds,
        StatusCode.ErrorInterpolation,
    ):
        particle.delete()


def parse_datetime_like(value: str) -> datetime:
    value = str(value).strip()
    for fmt in (
        "%Y%m%d-%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(
        f"Unsupported date format: {value}. "
        "Use one of: YYYYMMDD-HH:MM, YYYY-MM-DD HH:MM, YYYY-MM-DDTHH:MM, YYYYMMDD"
    )


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

    # =========================================================================
    # GROUPED-RELEASE EXPANSION (new)
    # =========================================================================
    group_cfg = rel_cfg.get("group", {})
    group_size = group_cfg.get("size", 1)

    if group_size > 1:
        lons_raw, lats_raw, group_id, group_member, group_size_arr = expand_groups(
            lons_base=lons_raw,
            lats_base=lats_raw,
            fieldset=fieldset,
            group_size=group_size,
            radius_km=group_cfg.get("radius_km", 0.1),
            placement=group_cfg.get("placement", "random"),
        )
        summarize_initial_points(lons_raw, lats_raw, name="grouped release points")

        metadata = {
            "group_id": group_id,
            "group_member": group_member,
            "group_size": group_size_arr,
        }
    else:
        # Single mode: trivial metadata (one particle per base center)
        metadata = {
            "group_id": np.arange(len(lons_raw), dtype=int),
            "group_member": np.ones(len(lons_raw), dtype=int),
            "group_size": np.ones(len(lons_raw), dtype=int),
        }

    # =========================================================================
    # DEPTH EXPANSION (existing, now applied to expanded lons_raw/lats_raw)
    # =========================================================================
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
        return lons, lats, depths, metadata

    return lons_raw, lats_raw, None, metadata


def build_particleset(cfg: dict, fieldset: FieldSet, lons, lats, depths=None, metadata_dict=None):
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

    # Add grouped-release metadata to ParticleSet
    # NOTE: Custom fields are passed to Parcels, but output to Zarr must be verified
    if metadata_dict:
        kwargs.update(metadata_dict)

    start_time = sim_cfg.get("start_time", None)
    if start_time is not None:
        dt0 = parse_datetime_like(start_time)
        kwargs["time"] = np.full(len(lons), np.datetime64(dt0))

    pset = ParticleSet.from_list(**kwargs)
    print(f"ParticleSet created with {len(lons)} particles")

    if start_time is not None:
        print(f"Particle release start_time = {start_time}")

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

    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(DeleteErrorParticle)

    pset.execute(
        kernels,
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
    lons, lats, depths, metadata = build_release(cfg, fieldset)
    pset = build_particleset(cfg, fieldset, lons, lats, depths=depths, metadata_dict=metadata)
    run_simulation(cfg, pset)


if __name__ == "__main__":
    main()