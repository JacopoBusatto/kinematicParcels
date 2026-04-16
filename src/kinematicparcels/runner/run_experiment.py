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

from kinematicparcels.utilities.geographicalRegions import (
    get_region_by_label,
    make_regular_grid_in_region,
)
from kinematicparcels.utilities.init_checks import (
    summarize_initial_points,
    check_initial_points_in_domain,
    filter_inside_domain,
    mask_inside_domain,
)
from kinematicparcels.utilities.init_depths import (
    summarize_depth_axis,
    build_multilevel_release,
)
from kinematicparcels.utilities.circle_release import (
    parse_timedelta_like,
    infer_depth_convention,
    to_positive_down,
    from_positive_down,
    depth_axis_max_positive_down,
    sample_circle_or_sphere,
)
from kinematicparcels.utilities.group_expansion import expand_groups
from kinematicparcels.utilities.lkm import build_lkm_modes
from kinematicparcels.utilities.group_dynamics import update_group_centers_and_relative_coords
from kinematicparcels.runner.kernels_lkm_inline import make_AdvectionRK4_with_LKM
from kinematicparcels.runner.grouped_kernels import make_grouped_rk4_lkm_kernel


# ============================================================================
# Custom Particle Classes with Grouped-Release Metadata Variables
# ============================================================================
class ScipyParticleGrouped(ScipyParticle):
    """ScipyParticle with group metadata and relative coordinates for LKM."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)

    # LKM-related variables: relative coordinates (kernel computes velocities)
    x_rel_m = Variable('x_rel_m', dtype=np.float32, initial=0.0)
    y_rel_m = Variable('y_rel_m', dtype=np.float32, initial=0.0)
    center_lon = Variable('center_lon', dtype=np.float32, initial=0.0)
    center_lat = Variable('center_lat', dtype=np.float32, initial=0.0)


class JITParticleGrouped(JITParticle):
    """JITParticle with group metadata and relative coordinates for LKM."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)

    # LKM-related variables: relative coordinates (kernel computes velocities)
    x_rel_m = Variable('x_rel_m', dtype=np.float32, initial=0.0)
    y_rel_m = Variable('y_rel_m', dtype=np.float32, initial=0.0)
    center_lon = Variable('center_lon', dtype=np.float32, initial=0.0)
    center_lat = Variable('center_lat', dtype=np.float32, initial=0.0)


class ScipyGroupEntityParticle(ScipyParticle):
    """One Parcels particle that stores all members of one fixed-size group."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_size = Variable('group_size', dtype=np.int32, initial=1)
    center_lon = Variable('center_lon', dtype=np.float32, initial=0.0)
    center_lat = Variable('center_lat', dtype=np.float32, initial=0.0)

    lon_1 = Variable('lon_1', dtype=np.float32, initial=0.0)
    lat_1 = Variable('lat_1', dtype=np.float32, initial=0.0)
    lon_2 = Variable('lon_2', dtype=np.float32, initial=0.0)
    lat_2 = Variable('lat_2', dtype=np.float32, initial=0.0)
    lon_3 = Variable('lon_3', dtype=np.float32, initial=0.0)
    lat_3 = Variable('lat_3', dtype=np.float32, initial=0.0)
    lon_4 = Variable('lon_4', dtype=np.float32, initial=0.0)
    lat_4 = Variable('lat_4', dtype=np.float32, initial=0.0)


# ============================================================================
# Helpers
# ============================================================================
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_group_entity_mode(cfg: dict) -> bool:
    group_size = int(cfg.get("release", {}).get("group", {}).get("size", 1))
    return group_size > 1


def get_particle_class(name: str, grouped_entity_mode: bool = False):
    name = name.lower()

    if grouped_entity_mode:
        if name == "scipy":
            return ScipyGroupEntityParticle
        if name == "jit":
            raise NotImplementedError(
                "Grouped-entity mode is currently implemented for scipy particles only. "
                "Keep singleton JIT runs unchanged."
            )
        raise ValueError(f"Unsupported particle_type: {name}")

    if name == "scipy":
        return ScipyParticleGrouped
    if name == "jit":
        return JITParticleGrouped
    raise ValueError(f"Unsupported particle_type: {name}")


def _build_group_entity_release(
    lons_base: np.ndarray,
    lats_base: np.ndarray,
    fieldset: FieldSet,
    group_cfg: dict,
):
    group_size = int(group_cfg.get("size", 1))
    if group_size < 2 or group_size > 4:
        raise ValueError(
            f"Grouped-entity mode supports group.size in [2, 4], got {group_size}"
        )

    lons_grouped, lats_grouped, group_id, group_member, _ = expand_groups(
        lons_base=lons_base,
        lats_base=lats_base,
        fieldset=fieldset,
        group_size=group_size,
        radius_km=group_cfg.get("radius_km", 0.1),
        placement=group_cfg.get("placement", "random"),
    )
    summarize_initial_points(lons_grouped, lats_grouped, name="grouped release points")

    unique_group_ids = np.unique(group_id)
    n_groups = len(unique_group_ids)

    center_lons = np.zeros(n_groups, dtype=float)
    center_lats = np.zeros(n_groups, dtype=float)
    lon_members = np.zeros((n_groups, 4), dtype=float)
    lat_members = np.zeros((n_groups, 4), dtype=float)

    for ig, gid in enumerate(unique_group_ids):
        idx = np.where(group_id == gid)[0]
        if len(idx) != group_size:
            raise ValueError(
                f"Expected {group_size} members for group_id={gid}, found {len(idx)}"
            )

        order = np.argsort(group_member[idx])
        idx_sorted = idx[order]
        member_lons = lons_grouped[idx_sorted]
        member_lats = lats_grouped[idx_sorted]

        lon_members[ig, :group_size] = member_lons
        lat_members[ig, :group_size] = member_lats
        center_lons[ig] = np.mean(member_lons)
        center_lats[ig] = np.mean(member_lats)

    metadata = {
        "group_id": unique_group_ids.astype(int),
        "group_size": np.full(n_groups, group_size, dtype=int),
        "center_lon": center_lons,
        "center_lat": center_lats,
        "lon_1": lon_members[:, 0],
        "lat_1": lat_members[:, 0],
        "lon_2": lon_members[:, 1],
        "lat_2": lat_members[:, 1],
        "lon_3": lon_members[:, 2],
        "lat_3": lat_members[:, 2],
        "lon_4": lon_members[:, 3],
        "lat_4": lat_members[:, 3],
    }

    print(f"Grouped-entity mode: {n_groups} groups, size={group_size}")
    return center_lons, center_lats, metadata


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


def _tile_metadata(metadata: dict, repeat_factor: int) -> dict:
    if repeat_factor <= 1:
        return metadata

    out = {}
    for key, arr in metadata.items():
        arr_np = np.asarray(arr)
        out[key] = np.tile(arr_np, repeat_factor)
    return out


def _build_circle_release(
    rel_cfg: dict,
    sim_cfg: dict,
    fieldset: FieldSet,
):
    circle_cfg = rel_cfg.get("circle", {})

    for key in ("lat", "lon", "dimension", "release_interval", "release_period"):
        if key not in circle_cfg:
            raise ValueError(f"release.circle.{key} is required for mode='circle'")

    radius_km = circle_cfg.get("radius_km", circle_cfg.get("radius"))
    if radius_km is None:
        radius_km = circle_cfg.get("radious_km", circle_cfg.get("radious"))
    if radius_km is None:
        raise ValueError("release.circle.radius_km (or radius) is required")
    radius_km = float(radius_km)

    count_per_timestep = int(circle_cfg.get("count_per_timestep", 0))
    if count_per_timestep <= 0:
        raise ValueError("release.circle.count_per_timestep must be > 0")

    dimension = str(circle_cfg["dimension"]).strip().lower()
    if dimension not in {"2d", "3d"}:
        raise ValueError("release.circle.dimension must be '2D' or '3D'")

    sampling = str(circle_cfg.get("sampling", "uniform")).strip().lower()
    if sampling not in {"uniform", "gaussian"}:
        raise ValueError("release.circle.sampling must be 'uniform' or 'gaussian'")

    out_policy = str(circle_cfg.get("out_of_domain_policy", "retry")).strip().lower()
    if out_policy not in {"retry", "drop", "error"}:
        raise ValueError("release.circle.out_of_domain_policy must be retry, drop, or error")

    bath_policy = str(circle_cfg.get("bathymetry_policy", "drop")).strip().lower()
    if bath_policy not in {"drop", "clip_to_depth_axis", "ignore"}:
        raise ValueError(
            "release.circle.bathymetry_policy must be drop, clip_to_depth_axis, or ignore"
        )

    if "start_time" not in sim_cfg:
        raise ValueError(
            "simulation.start_time is required for time-dependent circle release"
        )

    t0 = parse_datetime_like(sim_cfg["start_time"])
    dt_release = parse_timedelta_like(circle_cfg["release_interval"])
    release_period = parse_timedelta_like(circle_cfg["release_period"])
    if dt_release.total_seconds() <= 0:
        raise ValueError("release.circle.release_interval must be > 0")
    if release_period.total_seconds() < 0:
        raise ValueError("release.circle.release_period must be >= 0")

    release_times_dt = []
    t = t0
    t_end = t0 + release_period
    while t <= t_end:
        release_times_dt.append(t)
        t += dt_release

    center_lon = float(circle_cfg["lon"])
    center_lat = float(circle_cfg["lat"])
    seed = circle_cfg.get("seed", None)
    rng = np.random.default_rng(None if seed is None else int(seed))

    depth_convention = infer_depth_convention(fieldset)
    depth_max_pd = depth_axis_max_positive_down(fieldset)

    center_depth_pd = None
    if dimension == "3d":
        if depth_max_pd is None:
            raise ValueError(
                "release.circle.dimension='3D' requires a fieldset depth axis"
            )
        if "depth" not in circle_cfg:
            raise ValueError("release.circle.depth is required in 3D mode")
        center_depth_raw = float(circle_cfg["depth"])
        center_depth_pd = float(to_positive_down(center_depth_raw, depth_convention))
        if center_depth_pd < 0:
            raise ValueError(
                "release.circle.depth must be below the surface in inferred depth convention"
            )

    lons_all = []
    lats_all = []
    depths_pd_all = []
    times_all = []

    for t_step in release_times_dt:
        if out_policy in {"drop", "error"}:
            candidate_lons, candidate_lats, candidate_depths_pd = sample_circle_or_sphere(
                center_lon=center_lon,
                center_lat=center_lat,
                center_depth_pd=center_depth_pd,
                radius_km=radius_km,
                count=count_per_timestep,
                dimension=dimension,
                sampling=sampling,
                rng=rng,
            )

            mask_domain = mask_inside_domain(candidate_lons, candidate_lats, fieldset)
            mask_depth = np.ones(len(candidate_lons), dtype=bool)

            if dimension == "3d":
                mask_depth &= candidate_depths_pd >= 0.0
                if bath_policy == "clip_to_depth_axis" and depth_max_pd is not None:
                    candidate_depths_pd = np.clip(candidate_depths_pd, 0.0, depth_max_pd)
                elif bath_policy == "drop" and depth_max_pd is not None:
                    mask_depth &= candidate_depths_pd <= depth_max_pd

            mask_ok = mask_domain & mask_depth

            if out_policy == "error" and not np.all(mask_ok):
                raise ValueError(
                    "circle release produced out-of-domain or invalid-depth points "
                    "with out_of_domain_policy='error'"
                )

            keep_lons = candidate_lons[mask_ok]
            keep_lats = candidate_lats[mask_ok]
            if dimension == "3d":
                keep_depths_pd = candidate_depths_pd[mask_ok]
            else:
                keep_depths_pd = None

        else:  # retry
            keep_lons = []
            keep_lats = []
            keep_depths_pd = []
            attempts = 0
            max_attempts = 100

            while len(keep_lons) < count_per_timestep and attempts < max_attempts:
                attempts += 1
                batch_count = count_per_timestep - len(keep_lons)
                candidate_lons, candidate_lats, candidate_depths_pd = sample_circle_or_sphere(
                    center_lon=center_lon,
                    center_lat=center_lat,
                    center_depth_pd=center_depth_pd,
                    radius_km=radius_km,
                    count=batch_count,
                    dimension=dimension,
                    sampling=sampling,
                    rng=rng,
                )

                mask_domain = mask_inside_domain(candidate_lons, candidate_lats, fieldset)
                mask_depth = np.ones(len(candidate_lons), dtype=bool)

                if dimension == "3d":
                    mask_depth &= candidate_depths_pd >= 0.0
                    if bath_policy == "clip_to_depth_axis" and depth_max_pd is not None:
                        candidate_depths_pd = np.clip(candidate_depths_pd, 0.0, depth_max_pd)
                    elif bath_policy == "drop" and depth_max_pd is not None:
                        mask_depth &= candidate_depths_pd <= depth_max_pd

                mask_ok = mask_domain & mask_depth

                keep_lons.extend(candidate_lons[mask_ok].tolist())
                keep_lats.extend(candidate_lats[mask_ok].tolist())
                if dimension == "3d":
                    keep_depths_pd.extend(candidate_depths_pd[mask_ok].tolist())

            if len(keep_lons) < count_per_timestep:
                raise ValueError(
                    "circle release could not collect enough valid points with "
                    "out_of_domain_policy='retry'. Increase radius, reduce count, "
                    "or switch policy."
                )

            keep_lons = np.asarray(keep_lons[:count_per_timestep], dtype=float)
            keep_lats = np.asarray(keep_lats[:count_per_timestep], dtype=float)
            if dimension == "3d":
                keep_depths_pd = np.asarray(keep_depths_pd[:count_per_timestep], dtype=float)
            else:
                keep_depths_pd = None

        keep_n = len(keep_lons)
        if keep_n == 0:
            continue

        lons_all.append(np.asarray(keep_lons, dtype=float))
        lats_all.append(np.asarray(keep_lats, dtype=float))
        times_all.append(np.full(keep_n, np.datetime64(t_step), dtype="datetime64[ns]"))
        if dimension == "3d":
            depths_pd_all.append(np.asarray(keep_depths_pd, dtype=float))

    if len(lons_all) == 0:
        raise ValueError("circle release generated zero valid points")

    lons = np.concatenate(lons_all)
    lats = np.concatenate(lats_all)
    release_times = np.concatenate(times_all)

    if dimension == "3d":
        depths_pd = np.concatenate(depths_pd_all)
        depths = from_positive_down(depths_pd, depth_convention)
    else:
        depths = None

    print(
        f"Circle release: steps={len(release_times_dt)}, generated={len(lons)} points, "
        f"dimension={dimension.upper()}, sampling={sampling}, policy={out_policy}"
    )

    return lons, lats, depths, release_times, dimension


def parse_lkm_config(cfg: dict) -> dict | None:
    """Parse and validate LKM configuration from YAML."""
    lkm_cfg = cfg.get('lkm', {})

    if not lkm_cfg.get('enabled', False):
        return None

    # Required parameters
    required_keys = ['L_min_km', 'L_max_km', 'increment_factor', 'epsilon_tke', 'c0']
    for key in required_keys:
        if key not in lkm_cfg:
            raise ValueError(f"LKM config missing required key: {key}")

    # Type conversions and defaults
    parsed_cfg = {
        'enabled': True,
        'mode': lkm_cfg.get('mode', 'group_center_of_mass'),
        'L_min_km': float(lkm_cfg['L_min_km']),
        'L_max_km': float(lkm_cfg['L_max_km']),
        'increment_factor': float(lkm_cfg['increment_factor']),
        'epsilon_tke': float(lkm_cfg['epsilon_tke']),
        'c0': float(lkm_cfg['c0']),
        'phi_mode': lkm_cfg.get('phi_mode', 'constant'),
        'phi_value': float(lkm_cfg.get('phi_value', np.pi/4)),
        'update_every_steps': int(lkm_cfg.get('update_every_steps', 1)),
        'apply_to_group_size_min': int(lkm_cfg.get('apply_to_group_size_min', 2)),
        'debug_output': bool(lkm_cfg.get('debug_output', False)),
    }

    return parsed_cfg


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
    sim_cfg = cfg["simulation"]
    release_mode = rel_cfg.get("mode", "region_grid")
    release_times = None
    release_dimension = "2d"
    depths_from_circle_3d = None

    if release_mode == "region_grid":
        lons_raw, lats_raw = _build_release_points_from_region(rel_cfg)

    elif release_mode == "point_list":
        lons_raw, lats_raw = _build_release_points_from_list(rel_cfg)

    elif release_mode == "circle":
        lons_raw, lats_raw, depths_from_circle_3d, release_times, release_dimension = _build_circle_release(
            rel_cfg=rel_cfg,
            sim_cfg=sim_cfg,
            fieldset=fieldset,
        )

    else:
        raise ValueError(
            f"Unsupported release.mode: {release_mode}. "
            "Use 'region_grid', 'point_list', or 'circle'."
        )

    summarize_initial_points(lons_raw, lats_raw, name="raw release points")

    if release_mode != "circle" and rel_cfg.get("filter_domain", True):
        check_initial_points_in_domain(lons_raw, lats_raw, fieldset, verbose=True)
        lons_raw, lats_raw = filter_inside_domain(lons_raw, lats_raw, fieldset)
        summarize_initial_points(lons_raw, lats_raw, name="filtered release points")

    # =========================================================================
    # GROUPED/SINGLETON RELEASE BUILDING
    # =========================================================================
    group_cfg = rel_cfg.get("group", {})
    group_size = int(group_cfg.get("size", 1))

    depth_cfg = rel_cfg.get("depth", {})
    use_depth = depth_cfg.get("enabled", False)

    if _is_group_entity_mode(cfg):
        if use_depth:
            raise NotImplementedError(
                "Grouped-entity mode currently supports depth.enabled=false only."
            )

        center_lons, center_lats, metadata = _build_group_entity_release(
            lons_base=lons_raw,
            lats_base=lats_raw,
            fieldset=fieldset,
            group_cfg=group_cfg,
        )
        return center_lons, center_lats, None, metadata, release_times

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

        if release_times is not None:
            release_times = np.repeat(release_times, group_size)
    else:
        # Single mode: trivial metadata (one particle per base center)
        metadata = {
            "group_id": np.arange(len(lons_raw), dtype=int),
            "group_member": np.ones(len(lons_raw), dtype=int),
            "group_size": np.ones(len(lons_raw), dtype=int),
        }

    # =========================================================================
    # DEPTH EXPANSION (singleton/member-based mode)
    # =========================================================================

    if release_mode == "circle" and release_dimension == "3d":
        return lons_raw, lats_raw, depths_from_circle_3d, metadata, release_times

    if use_depth:
        summarize_depth_axis(fieldset)

        n_before_depth = len(lons_raw)

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

        repeat_factor = int(len(lons) / n_before_depth) if n_before_depth > 0 else 1
        metadata = _tile_metadata(metadata, repeat_factor)
        if release_times is not None and repeat_factor > 1:
            release_times = np.tile(release_times, repeat_factor)

        return lons, lats, depths, metadata, release_times

    return lons_raw, lats_raw, None, metadata, release_times


def build_particleset(
    cfg: dict,
    fieldset: FieldSet,
    lons,
    lats,
    depths=None,
    metadata_dict=None,
    release_times=None,
):
    sim_cfg = cfg["simulation"]
    pclass = get_particle_class(
        sim_cfg.get("particle_type", "scipy"),
        grouped_entity_mode=_is_group_entity_mode(cfg),
    )

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
    if release_times is not None:
        kwargs["time"] = np.asarray(release_times)
    elif start_time is not None:
        dt0 = parse_datetime_like(start_time)
        kwargs["time"] = np.full(len(lons), np.datetime64(dt0))

    pset = ParticleSet.from_list(**kwargs)
    print(f"ParticleSet created with {len(lons)} particles")

    if release_times is not None:
        print(
            f"Particle release schedule: {release_times.min()} -> {release_times.max()} "
            f"({len(np.unique(release_times))} timesteps)"
        )
    elif start_time is not None:
        print(f"Particle release start_time = {start_time}")

    return pset

def run_simulation(cfg: dict, pset: ParticleSet, fieldset: FieldSet, lkm_modes=None):
    """Run simulation in singleton/member mode or grouped-entity mode."""
    sim_cfg = cfg["simulation"]
    out_cfg = cfg["output"]
    exp_cfg = cfg["experiment"]
    lkm_cfg = cfg.get('lkm', {})

    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = output_dir / out_cfg["zarr_name"]

    # Simulation parameters
    dt_integration_hours = sim_cfg["dt_hours"]
    dt_output_hours = sim_cfg["outputdt_hours"]
    runtime_days = sim_cfg["runtime_days"]

    grouped_entity_mode = _is_group_entity_mode(cfg)

    output_file = pset.ParticleFile(
        name=str(zarr_path),
        outputdt=timedelta(hours=dt_output_hours),
    )

    # Choose kernel and execution strategy.
    if grouped_entity_mode:
        group_size = int(cfg["release"].get("group", {}).get("size", 1))
        if lkm_modes is not None:
            fieldset.lkm_modes = lkm_modes
            print(f"LKM enabled in grouped-entity mode: {lkm_modes.n_modes} modes")
        else:
            print("LKM disabled in grouped-entity mode")

        grouped_kernel = make_grouped_rk4_lkm_kernel(group_size)
        kernels = pset.Kernel(grouped_kernel)

        pset.execute(
            kernels,
            runtime=timedelta(days=runtime_days),
            dt=timedelta(hours=dt_integration_hours),
            output_file=output_file,
        )
    elif lkm_modes is not None:
        update_freq = lkm_cfg.get('update_every_steps', 1)
        dt_sync_hours = dt_integration_hours * update_freq

        # LKM enabled: Attach lkm_modes to fieldset so kernel can access it
        fieldset.lkm_modes = lkm_modes

        # Member-based custom kernel with synchronized group updates
        kernel_func = make_AdvectionRK4_with_LKM(lkm_modes)
        kernels = pset.Kernel(kernel_func)
        print(f"LKM enabled: {lkm_modes.n_modes} modes, update every {update_freq} steps")

        total_steps = int(runtime_days * 24 / dt_sync_hours)
        for _ in range(total_steps):
            update_group_centers_and_relative_coords(
                pset=pset,
                apply_to_group_size_min=lkm_cfg.get('apply_to_group_size_min', 2),
            )
            pset.execute(
                kernels,
                runtime=timedelta(hours=dt_sync_hours),
                dt=timedelta(hours=dt_integration_hours),
                output_file=output_file,
            )
    else:
        kernels = pset.Kernel(AdvectionRK4)
        print("LKM disabled: using standard advection")
        pset.execute(
            kernels,
            runtime=timedelta(days=runtime_days),
            dt=timedelta(hours=dt_integration_hours),
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

    # Parse LKM configuration
    lkm_cfg = parse_lkm_config(cfg)
    lkm_modes = None

    if lkm_cfg is not None:
        print("Building LKM modes...")
        lkm_modes = build_lkm_modes(
            L_min_m=lkm_cfg['L_min_km'] * 1000,
            L_max_m=lkm_cfg['L_max_km'] * 1000,
            increment_factor=lkm_cfg['increment_factor'],
            epsilon_tke=lkm_cfg['epsilon_tke'],
            c0=lkm_cfg['c0'],
            phi_spec=(lkm_cfg['phi_mode'], lkm_cfg.get('phi_value')),
        )
        print(f"LKM modes built: {lkm_modes.n_modes} modes from {lkm_cfg['L_min_km']} to {lkm_cfg['L_max_km']} km")

    fieldset = build_fieldset(cfg)
    lons, lats, depths, metadata, release_times = build_release(cfg, fieldset)
    pset = build_particleset(
        cfg,
        fieldset,
        lons,
        lats,
        depths=depths,
        metadata_dict=metadata,
        release_times=release_times,
    )
    run_simulation(cfg, pset, fieldset, lkm_modes=lkm_modes)


if __name__ == "__main__":
    main()