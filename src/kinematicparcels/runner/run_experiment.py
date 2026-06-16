from __future__ import annotations

import argparse
from datetime import timedelta, datetime
from glob import glob
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

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

from kinematicparcels.regions import (
    get_region_by_label,
    make_regular_grid_in_region,
)
from kinematicparcels.utilities.init_checks import (
    summarize_initial_points,
    check_initial_points_in_domain,
    filter_inside_domain,
    filter_inside_ocean,
    mask_inside_domain,
    mask_inside_ocean,
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
from kinematicparcels.runner.kernels import (
    BoundaryHaloKill,
    DeleteParticleIfTooOld,
    WrapLongitudePeriodic,
)
from kinematicparcels.runner.grouped_kernels import (
    make_grouped_rk4_lkm_kernel,
    BoundaryHaloKill_GroupedEntity,
    WrapLongitudePeriodic_GroupedEntity,
)


# ============================================================================
# Custom Particle Classes with Grouped-Release Metadata Variables
# ============================================================================
class ScipyParticleGrouped(ScipyParticle):
    """ScipyParticle with group metadata and relative coordinates for LKM."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)
    circle_id = Variable('circle_id', dtype=np.int32, initial=1)
    release_time = Variable('release_time', dtype=np.float64, initial=0.0, to_write=False)

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
    circle_id = Variable('circle_id', dtype=np.int32, initial=1)
    release_time = Variable('release_time', dtype=np.float64, initial=0.0, to_write=False)

    # LKM-related variables: relative coordinates (kernel computes velocities)
    x_rel_m = Variable('x_rel_m', dtype=np.float32, initial=0.0)
    y_rel_m = Variable('y_rel_m', dtype=np.float32, initial=0.0)
    center_lon = Variable('center_lon', dtype=np.float32, initial=0.0)
    center_lat = Variable('center_lat', dtype=np.float32, initial=0.0)


class ScipyGroupEntityParticle(ScipyParticle):
    """One Parcels particle that stores all members of one fixed-size group."""
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_size = Variable('group_size', dtype=np.int32, initial=1)
    circle_id = Variable('circle_id', dtype=np.int32, initial=1)
    release_time = Variable('release_time', dtype=np.float64, initial=0.0, to_write=False)
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
    lon_5 = Variable('lon_5', dtype=np.float32, initial=0.0)
    lat_5 = Variable('lat_5', dtype=np.float32, initial=0.0)


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
    circle_ids_base: np.ndarray | None = None,
    filter_land: bool = False,
):
    group_size = int(group_cfg.get("size", 1))
    if group_size < 2 or group_size > 5:
        raise ValueError(
            f"Grouped-entity mode supports group.size in [2, 5], got {group_size}"
        )

    lons_grouped, lats_grouped, group_id, group_member, _ = expand_groups(
        lons_base=lons_base,
        lats_base=lats_base,
        fieldset=fieldset,
        group_size=group_size,
        radius_km=group_cfg.get("radius_km", 0.1),
        placement=group_cfg.get("placement", "random"),
        filter_land=filter_land,
    )
    summarize_initial_points(lons_grouped, lats_grouped, name="grouped release points")

    unique_group_ids = np.unique(group_id)
    n_groups = len(unique_group_ids)

    center_lons = np.zeros(n_groups, dtype=float)
    center_lats = np.zeros(n_groups, dtype=float)
    lon_members = np.zeros((n_groups, 5), dtype=float)
    lat_members = np.zeros((n_groups, 5), dtype=float)

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
        "lon_5": lon_members[:, 4],
        "lat_5": lat_members[:, 4],
    }
    if circle_ids_base is not None:
        metadata["circle_id"] = np.asarray(circle_ids_base, dtype=int)[unique_group_ids]

    print(f"Grouped-entity mode: {n_groups} groups, size={group_size}")
    return center_lons, center_lats, metadata


def _needs_xarray_fieldset_fallback(sample_file: str, variables: dict, dims_cfg: dict) -> bool:
    """Detect whether xarray normalization is needed before building the fieldset."""
    lat_name = dims_cfg["lat"]
    lon_name = dims_cfg["lon"]

    with xr.open_dataset(sample_file) as ds:
        lon_coord = ds[lon_name]
        lat_coord = ds[lat_name]

        if lon_coord.ndim != 1 or lat_coord.ndim != 1:
            return True

        for var_name in variables.values():
            dims = tuple(ds[var_name].dims)
            if lat_name not in dims or lon_name not in dims:
                return True
            if dims[-2:] != (lat_name, lon_name):
                return True
    return False


def _normalize_coordinate_axes(lon_coord: xr.DataArray, lat_coord: xr.DataArray):
    """Return axes and spatial dimension order for regular or curvilinear coordinates."""
    lon_vals = np.asarray(lon_coord.values)
    lat_vals = np.asarray(lat_coord.values)

    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        return {
            "lon": lon_vals,
            "lat": lat_vals,
            "lon_dim": lon_coord.dims[0],
            "lat_dim": lat_coord.dims[0],
            "curvilinear": False,
        }

    if lon_vals.ndim != 2 or lat_vals.ndim != 2 or lon_vals.shape != lat_vals.shape:
        raise ValueError("Unsupported coordinate layout for fieldset lon/lat variables")

    dims2d = lon_coord.dims

    if np.allclose(lon_vals, lon_vals[:, [0]]) and np.allclose(lat_vals, lat_vals[[0], :]):
        return {
            "lon": lon_vals[:, 0],
            "lat": lat_vals[0, :],
            "lon_dim": dims2d[0],
            "lat_dim": dims2d[1],
            "curvilinear": False,
        }

    if np.allclose(lon_vals, lon_vals[[0], :]) and np.allclose(lat_vals, lat_vals[:, [0]]):
        return {
            "lon": lon_vals[0, :],
            "lat": lat_vals[:, 0],
            "lon_dim": dims2d[1],
            "lat_dim": dims2d[0],
            "curvilinear": False,
        }

    return {
        "lon": lon_vals,
        "lat": lat_vals,
        "lon_dim": dims2d[1],
        "lat_dim": dims2d[0],
        "curvilinear": True,
    }


def _build_fieldset_via_xarray(files: list[str], variables: dict, dims_cfg: dict, mesh: str) -> FieldSet:
    """Build a Parcels fieldset from xarray after normalizing coordinates and data order."""
    ds = xr.open_mfdataset(files, combine="by_coords")
    try:
        lon_name = dims_cfg["lon"]
        lat_name = dims_cfg["lat"]
        time_dim = dims_cfg["time"]
        depth_dim = dims_cfg.get("depth")

        coord_info = _normalize_coordinate_axes(ds[lon_name], ds[lat_name])
        lon_dim = coord_info["lon_dim"]
        lat_dim = coord_info["lat_dim"]

        u = ds[variables["U"]]
        v = ds[variables["V"]]

        target_dims = []
        if time_dim and time_dim in u.dims:
            target_dims.append(time_dim)
        if depth_dim and depth_dim in u.dims:
            target_dims.append(depth_dim)
        target_dims.extend([lat_dim, lon_dim])

        u = u.transpose(*[d for d in target_dims if d in u.dims])
        v = v.transpose(*[d for d in target_dims if d in v.dims])

        data = {
            "U": np.asarray(u.values),
            "V": np.asarray(v.values),
        }
        dimensions = {
            "lon": np.asarray(coord_info["lon"]),
            "lat": np.asarray(coord_info["lat"]),
            "time": np.asarray(ds[time_dim].values),
        }
        if depth_dim and depth_dim in ds:
            dimensions["depth"] = np.asarray(ds[depth_dim].values)

        if coord_info["curvilinear"]:
            print("Detected true 2D coordinates; building curvilinear fieldset via xarray")
        else:
            print("Detected regular grid stored with 2D coordinates; reducing to 1D axes via xarray")

        fieldset = FieldSet.from_data(data=data, dimensions=dimensions, mesh=mesh)
    finally:
        ds.close()

    return fieldset


def _apply_periodic_halo(fieldset: FieldSet, cfg: dict) -> None:
    """Apply a zonal periodic halo when requested in the fieldset config."""
    fs_cfg = cfg["fieldset"]
    if not bool(fs_cfg.get("periodic_halo", False)):
        return

    lon_axis = np.asarray(fieldset.U.grid.lon)
    if len(lon_axis) < 2:
        raise ValueError("fieldset.periodic_halo requires at least two longitude grid points")

    halo_size = int(fs_cfg.get("periodic_halo_size", 5))
    if halo_size <= 0:
        raise ValueError("fieldset.periodic_halo_size must be > 0 when periodic_halo=true")

    dlon = float(np.mean(np.diff(lon_axis)))
    wrap_west = float(lon_axis[0])
    wrap_span = float(lon_axis[-1] - lon_axis[0] + abs(dlon))
    fieldset.add_constant("periodic_lon_west", wrap_west)
    fieldset.add_constant("periodic_lon_span", wrap_span)

    fieldset.add_periodic_halo(zonal=True, halosize=halo_size)
    print(f"Applied periodic halo: zonal=True, halosize={halo_size}")


def _is_boundary_halo_enabled(cfg: dict) -> bool:
    """Return whether the boundary-halo kill guard is enabled."""
    return bool(cfg.get("simulation", {}).get("boundary_halo", {}).get("enabled", True))


def _needs_periodic_wrap(cfg: dict) -> bool:
    """Return whether runtime longitude wrapping is active."""
    return bool(cfg.get("fieldset", {}).get("periodic_halo", False))


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

    mesh = fs_cfg.get("mesh", "spherical")

    if _needs_xarray_fieldset_fallback(files[0], variables, dims_cfg):
        print("Detected non-standard variable dimension order; reordering via xarray")
        fieldset = _build_fieldset_via_xarray(files, variables, dims_cfg, mesh)
    else:
        fieldset = FieldSet.from_netcdf(
            filenames=filenames,
            variables=variables,
            dimensions=dimensions,
            mesh=mesh,
        )

    _apply_periodic_halo(fieldset, cfg)

    # Cache source metadata for release-time ocean/land filtering.
    fieldset._kp_source_files = files
    fieldset._kp_variables = variables.copy()
    fieldset._kp_dimensions = dims_cfg.copy()

    _attach_boundary_halo_constants(fieldset, cfg)

    print(fieldset)
    return fieldset


def _attach_boundary_halo_constants(fieldset: FieldSet, cfg: dict) -> None:
    """Attach runtime constants needed by periodic wrapping and boundary-halo kill."""
    fs_cfg = cfg["fieldset"]
    bh_cfg = cfg.get("simulation", {}).get("boundary_halo", {})
    enabled = _is_boundary_halo_enabled(cfg)
    n_cells = int(bh_cfg.get("n_cells", 1))

    periodic = bool(fs_cfg.get("periodic_halo", False))
    fieldset.add_constant("bh_periodic", 1.0 if periodic else 0.0)

    if not enabled:
        print("Boundary halo guard disabled")
        return

    lon_axis = np.asarray(fieldset.U.grid.lon)
    lat_axis = np.asarray(fieldset.U.grid.lat)

    if lon_axis.ndim != 1 or lat_axis.ndim != 1:
        raise ValueError(
            "simulation.boundary_halo currently supports regular 1D lon/lat axes only. "
            "Disable boundary_halo or normalize the fieldset to a regular grid."
        )

    dlon = float(np.mean(np.diff(lon_axis))) if len(lon_axis) > 1 else 0.0
    dlat = float(np.mean(np.diff(lat_axis))) if len(lat_axis) > 1 else 0.0

    if n_cells > 0:
        lat_min = float(lat_axis[0])  + n_cells * abs(dlat)
        lat_max = float(lat_axis[-1]) - n_cells * abs(dlat)
        lon_min = float(lon_axis[0])  + n_cells * abs(dlon)
        lon_max = float(lon_axis[-1]) - n_cells * abs(dlon)
        print(
            f"Boundary halo guard: n_cells={n_cells}, "
            f"lat=[{lat_min:.4f}, {lat_max:.4f}], "
            f"lon=[{lon_min:.4f}, {lon_max:.4f}]"
            + (" (lon check skipped – periodic)" if periodic else "")
        )
    else:
        # Disabled: set bounds to the actual field edges so the check never fires.
        lat_min = float(lat_axis[0])
        lat_max = float(lat_axis[-1])
        lon_min = float(lon_axis[0])
        lon_max = float(lon_axis[-1])

    fieldset.add_constant("bh_lat_min", lat_min)
    fieldset.add_constant("bh_lat_max", lat_max)
    fieldset.add_constant("bh_lon_min", lon_min)
    fieldset.add_constant("bh_lon_max", lon_max)


def _build_release_points_from_region(rel_cfg: dict):
    region = get_region_by_label(rel_cfg["region_label"])
    lons2d, lats2d = make_regular_grid_in_region(
        region,
        dlon=rel_cfg["dlon"],
        dlat=rel_cfg["dlat"],
    )
    return lons2d, lats2d


def _build_release_points_from_list(rel_cfg: dict):
    def _read_points_table(path_like: str | Path) -> pd.DataFrame:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"release.points_file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            table = pd.read_csv(path)
        elif suffix == ".parquet":
            table = pd.read_parquet(path)
        else:
            raise ValueError(
                "release.points_file must have extension .csv or .parquet"
            )

        if table.empty:
            raise ValueError("release.points_file contains no rows")

        return table

    def _parse_release_times(table: pd.DataFrame) -> np.ndarray | None:
        if "time" not in table.columns:
            return None
        return pd.to_datetime(table["time"], errors="raise").to_numpy(dtype="datetime64[ns]")

    def _build_explicit_group_entity_from_table(
        table: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None, int]:
        member_indices = []
        for idx in range(1, 6):
            lon_col = f"lon_{idx}"
            lat_col = f"lat_{idx}"
            has_lon = lon_col in table.columns
            has_lat = lat_col in table.columns
            if has_lon != has_lat:
                raise ValueError(
                    f"release.points_file must contain both '{lon_col}' and '{lat_col}'"
                )
            if has_lon:
                member_indices.append(idx)

        if len(member_indices) < 2:
            raise ValueError(
                "release.points_file grouped input requires lon_i/lat_i columns for i in [1, 5]"
            )

        expected = list(range(1, max(member_indices) + 1))
        if member_indices != expected:
            raise ValueError(
                "release.points_file grouped member columns must be contiguous from 1. "
                f"Found indices: {member_indices}"
            )

        group_size = len(member_indices)
        n_groups = len(table)

        lon_members = np.zeros((n_groups, 5), dtype=float)
        lat_members = np.zeros((n_groups, 5), dtype=float)

        for idx in member_indices:
            lon_col = f"lon_{idx}"
            lat_col = f"lat_{idx}"

            lon_vals = pd.to_numeric(table[lon_col], errors="coerce").to_numpy(dtype=float)
            lat_vals = pd.to_numeric(table[lat_col], errors="coerce").to_numpy(dtype=float)
            if np.any(~np.isfinite(lon_vals)) or np.any(~np.isfinite(lat_vals)):
                raise ValueError(
                    f"release.points_file contains non-finite values in '{lon_col}'/'{lat_col}'"
                )

            lon_members[:, idx - 1] = lon_vals
            lat_members[:, idx - 1] = lat_vals

        center_lons = np.mean(lon_members[:, :group_size], axis=1)
        center_lats = np.mean(lat_members[:, :group_size], axis=1)

        metadata = {
            "group_id": np.arange(n_groups, dtype=int),
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
            "lon_5": lon_members[:, 4],
            "lat_5": lat_members[:, 4],
        }

        if "circle_id" in table.columns:
            circle_vals = pd.to_numeric(table["circle_id"], errors="coerce").to_numpy(dtype=float)
            if np.any(~np.isfinite(circle_vals)):
                raise ValueError("release.points_file contains non-finite values in 'circle_id'")
            metadata["circle_id"] = circle_vals.astype(int)

        return center_lons, center_lats, metadata, _parse_release_times(table), group_size

    points_file = rel_cfg.get("points_file")
    points = rel_cfg.get("points", [])

    if points_file is not None and len(points) > 0:
        raise ValueError("Provide only one of release.points or release.points_file")

    if points_file is not None:
        table = _read_points_table(points_file)

        has_center_cols = ("lon" in table.columns) and ("lat" in table.columns)
        has_group_cols = any((f"lon_{idx}" in table.columns) or (f"lat_{idx}" in table.columns) for idx in range(1, 6))

        if has_center_cols and has_group_cols:
            raise ValueError(
                "release.points_file is ambiguous: provide either lon/lat columns or lon_i/lat_i columns"
            )

        if has_group_cols:
            lons, lats, metadata, release_times, group_size = _build_explicit_group_entity_from_table(table)
            print(f"Loaded grouped points_file: {len(lons)} groups, group_size={group_size}")
            return lons, lats, metadata, release_times, True, group_size

        if not has_center_cols:
            raise ValueError(
                "release.points_file must contain lon/lat columns or grouped lon_i/lat_i columns"
            )

        lons = pd.to_numeric(table["lon"], errors="coerce").to_numpy(dtype=float)
        lats = pd.to_numeric(table["lat"], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(lons)) or np.any(~np.isfinite(lats)):
            raise ValueError("release.points_file contains non-finite values in lon/lat columns")

        print(f"Loaded points_file: {len(lons)} points")
        return lons, lats, None, _parse_release_times(table), False, 1

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

    return np.asarray(lons), np.asarray(lats), None, None, False, 1


def _tile_metadata(metadata: dict, repeat_factor: int) -> dict:
    if repeat_factor <= 1:
        return metadata

    out = {}
    for key, arr in metadata.items():
        arr_np = np.asarray(arr)
        out[key] = np.tile(arr_np, repeat_factor)
    return out


def _is_backward_simulation(sim_cfg: dict) -> bool:
    return float(sim_cfg.get("dt_hours", 1.0)) < 0.0


def _build_release_schedule(
    *,
    start_time: datetime,
    release_interval: timedelta,
    release_period: timedelta,
    backward: bool,
) -> np.ndarray:
    if release_interval.total_seconds() <= 0:
        raise ValueError("release interval must be > 0")
    if release_period.total_seconds() < 0:
        raise ValueError("release period must be >= 0")

    release_times_dt = []
    t = start_time
    t_end = start_time - release_period if backward else start_time + release_period

    if backward:
        while t >= t_end:
            release_times_dt.append(t)
            t -= release_interval
    else:
        while t <= t_end:
            release_times_dt.append(t)
            t += release_interval

    return np.asarray(release_times_dt, dtype="datetime64[ns]")


def _build_continuous_release_schedule(sim_cfg: dict, continuous_cfg: dict) -> np.ndarray:
    if not continuous_cfg.get("enabled", False):
        return None

    for key in ("release_interval", "release_period"):
        if key not in continuous_cfg:
            raise ValueError(f"release.continuous.{key} is required when continuous.enabled=true")

    if "start_time" not in sim_cfg:
        raise ValueError(
            "simulation.start_time is required for continuous release scheduling"
        )

    t0 = parse_datetime_like(sim_cfg["start_time"])
    dt_release = parse_timedelta_like(continuous_cfg["release_interval"])
    release_period = parse_timedelta_like(continuous_cfg["release_period"])
    return _build_release_schedule(
        start_time=t0,
        release_interval=dt_release,
        release_period=release_period,
        backward=_is_backward_simulation(sim_cfg),
    )


def _get_scheduled_release_max_age_seconds(
    cfg: dict,
    *,
    has_release_schedule: bool | None = None,
) -> float | None:
    rel_cfg = cfg["release"]
    continuous_cfg = rel_cfg.get("continuous", {})
    raw_max_age = continuous_cfg.get("max_age")

    if raw_max_age is None:
        return None

    release_mode = rel_cfg.get("mode", "region_grid")
    if release_mode != "circle" and not continuous_cfg.get("enabled", False):
        raise ValueError(
            "release.continuous.max_age requires release.continuous.enabled=true "
            "for non-circle releases"
        )

    max_age = parse_timedelta_like(raw_max_age)
    if max_age.total_seconds() <= 0:
        raise ValueError("release.continuous.max_age must be > 0")

    if has_release_schedule is False:
        raise ValueError(
            "release.continuous.max_age requires a scheduled release with per-particle times"
        )

    return float(max_age.total_seconds())


def _build_circle_release(
    rel_cfg: dict,
    sim_cfg: dict,
    fieldset: FieldSet,
):
    circle_cfg = rel_cfg.get("circle", {})

    # Detect radius_km key (support legacy aliases)
    _radius_key = next(
        (k for k in ("radius_km", "radius", "radious_km", "radious") if k in circle_cfg),
        None,
    )

    for key in ("lat", "lon", "dimension", "release_interval", "release_period"):
        if key not in circle_cfg:
            raise ValueError(f"release.circle.{key} is required for mode='circle'")
    if _radius_key is None:
        raise ValueError("release.circle.radius_km (or radius) is required")

    # --- Detect multi-circle mode ---
    # The 6 list-able fields must be either ALL scalars or ALL lists of the same length.
    _LIST_ABLE = ("lat", "lon", _radius_key, "count_per_timestep", "release_interval", "release_period")
    _is_list = {k: isinstance(circle_cfg.get(k), list) for k in _LIST_ABLE}

    if any(_is_list.values()) and not all(_is_list.values()):
        scalar_keys = [k for k, v in _is_list.items() if not v]
        list_keys = [k for k, v in _is_list.items() if v]
        raise ValueError(
            "release.circle: in multi-circle mode all of lat, lon, radius_km, "
            "count_per_timestep, release_interval, and release_period must be lists. "
            f"Got lists for {list_keys} but scalars for {scalar_keys}."
        )

    multi_circle = any(_is_list.values())

    if multi_circle:
        if "start_time" in circle_cfg and not isinstance(circle_cfg["start_time"], list):
            raise ValueError(
                "release.circle.start_time must be a list in multi-circle mode"
            )
        lengths = {k: len(circle_cfg[k]) for k in _LIST_ABLE}
        if "start_time" in circle_cfg:
            lengths["start_time"] = len(circle_cfg["start_time"])
        if len(set(lengths.values())) > 1:
            raise ValueError(
                "release.circle: all list fields must have the same length. "
                f"Got lengths: {dict(lengths)}"
            )
        n_circles = next(iter(lengths.values()))
        per_circle = [
            {
                "lat": float(circle_cfg["lat"][i]),
                "lon": float(circle_cfg["lon"][i]),
                "radius_km": float(circle_cfg[_radius_key][i]),
                "count_per_timestep": int(circle_cfg["count_per_timestep"][i]),
                "release_interval": circle_cfg["release_interval"][i],
                "release_period": circle_cfg["release_period"][i],
                "start_time": circle_cfg.get("start_time", [None] * n_circles)[i],
            }
            for i in range(n_circles)
        ]
    else:
        if isinstance(circle_cfg.get("start_time"), list):
            raise ValueError(
                "release.circle.start_time must be a scalar in single-circle mode"
            )
        count_per_timestep = int(circle_cfg.get("count_per_timestep", 0))
        if count_per_timestep <= 0:
            raise ValueError("release.circle.count_per_timestep must be > 0")
        per_circle = [
            {
                "lat": float(circle_cfg["lat"]),
                "lon": float(circle_cfg["lon"]),
                "radius_km": float(circle_cfg[_radius_key]),
                "count_per_timestep": count_per_timestep,
                "release_interval": circle_cfg["release_interval"],
                "release_period": circle_cfg["release_period"],
                "start_time": circle_cfg.get("start_time"),
            }
        ]

    # --- Shared params (scalar only) ---
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

    if "start_time" not in sim_cfg and "start_time" not in circle_cfg:
        raise ValueError(
            "simulation.start_time is required for time-dependent circle release "
            "unless release.circle.start_time is provided"
        )
    seed = circle_cfg.get("seed", None)
    rng = np.random.default_rng(None if seed is None else int(seed))
    backward = _is_backward_simulation(sim_cfg)

    depth_convention = infer_depth_convention(fieldset)
    depth_max_pd = depth_axis_max_positive_down(fieldset)

    # --- Per-circle depth and depth_radius (supports scalar or list) ---
    center_depths_pd: list[float | None] = [None] * len(per_circle)
    depth_radii_m: list[float | None] = [None] * len(per_circle)

    if dimension == "3d":
        if depth_max_pd is None:
            raise ValueError(
                "release.circle.dimension='3D' requires a fieldset depth axis"
            )
        if "depth" not in circle_cfg:
            raise ValueError("release.circle.depth is required in 3D mode")
        if "depth_radius" not in circle_cfg:
            raise ValueError("release.circle.depth_radius is required in 3D mode")

        raw_depth = circle_cfg["depth"]
        raw_radius = circle_cfg["depth_radius"]
        depth_vals = raw_depth if isinstance(raw_depth, list) else [raw_depth] * len(per_circle)
        radius_vals = raw_radius if isinstance(raw_radius, list) else [raw_radius] * len(per_circle)

        if len(depth_vals) != len(per_circle):
            raise ValueError(
                f"release.circle.depth must have {len(per_circle)} values in multi-circle mode"
            )
        if len(radius_vals) != len(per_circle):
            raise ValueError(
                f"release.circle.depth_radius must have {len(per_circle)} values in multi-circle mode"
            )

        for i, (d, r) in enumerate(zip(depth_vals, radius_vals)):
            label_i = f"circle[{i + 1}]" if multi_circle else "circle"
            cdp = float(to_positive_down(float(d), depth_convention))
            if cdp < 0:
                raise ValueError(
                    f"release.{label_i}.depth must be below the surface in inferred depth convention"
                )
            dr = float(r)
            if dr <= 0:
                raise ValueError(f"release.{label_i}.depth_radius must be > 0")
            center_depths_pd[i] = cdp
            depth_radii_m[i] = dr

    elif dimension == "2d" and "depth" in circle_cfg:
        raw_depth = circle_cfg["depth"]
        depth_vals = raw_depth if isinstance(raw_depth, list) else [raw_depth] * len(per_circle)
        if len(depth_vals) != len(per_circle):
            raise ValueError(
                f"release.circle.depth must have {len(per_circle)} values in multi-circle mode"
            )
        for i, d in enumerate(depth_vals):
            center_depths_pd[i] = float(to_positive_down(float(d), depth_convention))

    lons_all = []
    lats_all = []
    depths_pd_all = []
    times_all = []
    circle_ids_all = []
    total_steps = 0

    for circle_idx, cp in enumerate(per_circle):
        circle_id = circle_idx + 1
        label = f"circle[{circle_id}]" if multi_circle else "circle"

        start_time_value = cp["start_time"]
        if start_time_value is None:
            if "start_time" not in sim_cfg:
                raise ValueError(f"release.{label}.start_time is required")
            start_time_value = sim_cfg["start_time"]

        circle_start_time = parse_datetime_like(start_time_value)
        dt_release = parse_timedelta_like(cp["release_interval"])
        rp = parse_timedelta_like(cp["release_period"])
        if cp["count_per_timestep"] <= 0:
            raise ValueError(f"release.{label}.count_per_timestep must be > 0")

        try:
            release_times_dt = _build_release_schedule(
                start_time=circle_start_time,
                release_interval=dt_release,
                release_period=rp,
                backward=backward,
            )
        except ValueError as exc:
            raise ValueError(f"release.{label}.{str(exc)}") from exc

        total_steps += len(release_times_dt)
        center_lon = cp["lon"]
        center_lat = cp["lat"]
        radius_km = cp["radius_km"]
        count_per_timestep = cp["count_per_timestep"]

        circle_center_depth_pd = center_depths_pd[circle_idx]
        circle_depth_radius_m = depth_radii_m[circle_idx]

        for t_step in release_times_dt:
            if out_policy in {"drop", "error"}:
                candidate_lons, candidate_lats, candidate_depths_pd = sample_circle_or_sphere(
                    center_lon=center_lon,
                    center_lat=center_lat,
                    center_depth_pd=circle_center_depth_pd,
                    radius_km=radius_km,
                    depth_radius_m=circle_depth_radius_m,
                    count=count_per_timestep,
                    dimension=dimension,
                    sampling=sampling,
                    rng=rng,
                )

                mask_domain = mask_inside_domain(candidate_lons, candidate_lats, fieldset)
                mask_depth = np.ones(len(candidate_lons), dtype=bool)

                if dimension == "3d":
                    assert candidate_depths_pd is not None
                    mask_depth &= candidate_depths_pd >= 0.0
                    if bath_policy == "clip_to_depth_axis" and depth_max_pd is not None:
                        candidate_depths_pd = np.clip(candidate_depths_pd, 0.0, depth_max_pd)
                    elif bath_policy == "drop" and depth_max_pd is not None:
                        mask_depth &= candidate_depths_pd <= depth_max_pd

                mask_ok = mask_domain & mask_depth

                if out_policy == "error" and not np.all(mask_ok):
                    raise ValueError(
                        f"{label} release produced out-of-domain or invalid-depth points "
                        "with out_of_domain_policy='error'"
                    )

                keep_lons = candidate_lons[mask_ok]
                keep_lats = candidate_lats[mask_ok]
                if dimension == "3d":
                    assert candidate_depths_pd is not None
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
                        center_depth_pd=circle_center_depth_pd,
                        radius_km=radius_km,
                        depth_radius_m=circle_depth_radius_m,
                        count=batch_count,
                        dimension=dimension,
                        sampling=sampling,
                        rng=rng,
                    )

                    mask_domain = mask_inside_domain(candidate_lons, candidate_lats, fieldset)
                    mask_depth = np.ones(len(candidate_lons), dtype=bool)

                    if dimension == "3d":
                        assert candidate_depths_pd is not None
                        mask_depth &= candidate_depths_pd >= 0.0
                        if bath_policy == "clip_to_depth_axis" and depth_max_pd is not None:
                            candidate_depths_pd = np.clip(candidate_depths_pd, 0.0, depth_max_pd)
                        elif bath_policy == "drop" and depth_max_pd is not None:
                            mask_depth &= candidate_depths_pd <= depth_max_pd

                    mask_ok = mask_domain & mask_depth

                    keep_lons.extend(candidate_lons[mask_ok].tolist())
                    keep_lats.extend(candidate_lats[mask_ok].tolist())
                    if dimension == "3d":
                        assert candidate_depths_pd is not None
                        keep_depths_pd.extend(candidate_depths_pd[mask_ok].tolist())

                if len(keep_lons) < count_per_timestep:
                    raise ValueError(
                        f"{label} release could not collect enough valid points with "
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
            circle_ids_all.append(np.full(keep_n, circle_id, dtype=int))
            if dimension == "3d":
                depths_pd_all.append(np.asarray(keep_depths_pd, dtype=float))
            elif center_depths_pd[circle_idx] is not None:
                depths_pd_all.append(np.full(keep_n, center_depths_pd[circle_idx], dtype=float))

    if len(lons_all) == 0:
        raise ValueError("circle release generated zero valid points")

    lons = np.concatenate(lons_all)
    lats = np.concatenate(lats_all)
    release_times = np.concatenate(times_all)
    circle_ids = np.concatenate(circle_ids_all)

    if dimension == "3d" or any(d is not None for d in center_depths_pd):
        depths_pd = np.concatenate(depths_pd_all)
        depths = from_positive_down(depths_pd, depth_convention)
    else:
        depths = None

    circle_label = f"{len(per_circle)} circles" if multi_circle else "circle"
    print(
        f"Circle release: {circle_label}, steps={total_steps}, generated={len(lons)} points, "
        f"dimension={dimension.upper()}, sampling={sampling}, policy={out_policy}"
    )

    return lons, lats, depths, release_times, dimension, circle_ids


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
    circle_ids_raw = None
    explicit_group_entity_metadata = None
    explicit_group_entity_size = None
    is_explicit_group_entity = False

    if release_mode == "region_grid":
        lons_raw, lats_raw = _build_release_points_from_region(rel_cfg)

    elif release_mode == "point_list":
        (
            lons_raw,
            lats_raw,
            explicit_group_entity_metadata,
            explicit_release_times,
            is_explicit_group_entity,
            explicit_group_entity_size,
        ) = _build_release_points_from_list(rel_cfg)
        if explicit_release_times is not None:
            release_times = np.asarray(explicit_release_times)

    elif release_mode == "circle":
        lons_raw, lats_raw, depths_from_circle_3d, release_times, release_dimension, circle_ids_raw = _build_circle_release(
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

    continuous_cfg = rel_cfg.get("continuous", {})

    if release_mode == "point_list" and is_explicit_group_entity:
        if continuous_cfg.get("enabled", False):
            raise ValueError(
                "release.continuous.enabled=true is not supported when release.points_file "
                "provides explicit lon_i/lat_i grouped coordinates"
            )

        group_cfg = rel_cfg.get("group", {})
        configured_group_size = int(group_cfg.get("size", 1))
        if configured_group_size != int(explicit_group_entity_size):
            raise ValueError(
                "release.group.size must match the grouped points_file member count. "
                f"Configured size={configured_group_size}, file size={explicit_group_entity_size}"
            )

        depth_cfg = rel_cfg.get("depth", {})
        use_depth = depth_cfg.get("enabled", False)

        metadata = explicit_group_entity_metadata
        if metadata is None:
            raise RuntimeError("Internal error: missing grouped metadata for points_file release")

        keep_groups = np.ones(len(lons_raw), dtype=bool)
        if rel_cfg.get("filter_domain", True):
            member_ok_domain = np.ones(len(lons_raw), dtype=bool)
            for member_idx in range(1, configured_group_size + 1):
                member_ok_domain &= mask_inside_domain(
                    np.asarray(metadata[f"lon_{member_idx}"]),
                    np.asarray(metadata[f"lat_{member_idx}"]),
                    fieldset,
                )
            keep_groups &= member_ok_domain

        if rel_cfg.get("filter_land", False):
            member_ok_ocean = np.ones(len(lons_raw), dtype=bool)
            for member_idx in range(1, configured_group_size + 1):
                member_ok_ocean &= mask_inside_ocean(
                    np.asarray(metadata[f"lon_{member_idx}"]),
                    np.asarray(metadata[f"lat_{member_idx}"]),
                    fieldset,
                )
            n_removed = int((~member_ok_ocean).sum())
            if n_removed > 0:
                print(f"[land filter] removed {n_removed} grouped release rows from points_file")
            keep_groups &= member_ok_ocean

        if not np.all(keep_groups):
            lons_raw = np.asarray(lons_raw)[keep_groups]
            lats_raw = np.asarray(lats_raw)[keep_groups]
            metadata = {k: np.asarray(v)[keep_groups] for k, v in metadata.items()}
            if release_times is not None:
                release_times = np.asarray(release_times)[keep_groups]

        summarize_initial_points(lons_raw, lats_raw, name="grouped points_file release points")

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

    if release_mode != "circle" and rel_cfg.get("filter_domain", True):
        check_initial_points_in_domain(lons_raw, lats_raw, fieldset, verbose=True)
        lons_raw, lats_raw = filter_inside_domain(lons_raw, lats_raw, fieldset)
        summarize_initial_points(lons_raw, lats_raw, name="filtered release points")

    if release_mode != "circle" and rel_cfg.get("filter_land", False):
        ocean_mask = mask_inside_ocean(lons_raw, lats_raw, fieldset)
        n_land = int((~ocean_mask).sum())
        if n_land > 0:
            print(f"[land filter] removed {n_land} release points on masked land cells")
        lons_raw, lats_raw = filter_inside_ocean(lons_raw, lats_raw, fieldset)
        summarize_initial_points(lons_raw, lats_raw, name="ocean-filtered release points")

    if release_mode != "circle" and continuous_cfg.get("enabled", False):
        if release_times is not None:
            raise ValueError(
                "release.continuous.enabled=true cannot be combined with per-row 'time' "
                "values from release.points_file"
            )

        release_steps = _build_continuous_release_schedule(sim_cfg, continuous_cfg)
        n_base_points = len(lons_raw)

        if n_base_points == 0:
            raise ValueError("continuous release has zero valid base points after filtering")

        release_times = np.repeat(release_steps, n_base_points)
        lons_raw = np.tile(lons_raw, len(release_steps))
        lats_raw = np.tile(lats_raw, len(release_steps))

        print(
            f"Continuous release: steps={len(release_steps)}, "
            f"base_points={n_base_points}, generated={len(lons_raw)} points"
        )

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
            circle_ids_base=circle_ids_raw,
            filter_land=rel_cfg.get("filter_land", False),
        )
        if release_times is not None:
            release_times = np.asarray(release_times)[metadata["group_id"]]
        depths = np.asarray(depths_from_circle_3d)[metadata["group_id"]] if depths_from_circle_3d is not None else None
        return center_lons, center_lats, depths, metadata, release_times

    if group_size > 1:
        lons_raw, lats_raw, group_id, group_member, group_size_arr = expand_groups(
            lons_base=lons_raw,
            lats_base=lats_raw,
            fieldset=fieldset,
            group_size=group_size,
            radius_km=group_cfg.get("radius_km", 0.1),
            placement=group_cfg.get("placement", "random"),
            filter_land=rel_cfg.get("filter_land", False),
        )
        summarize_initial_points(lons_raw, lats_raw, name="grouped release points")

        metadata = {
            "group_id": group_id,
            "group_member": group_member,
            "group_size": group_size_arr,
        }
        if circle_ids_raw is not None:
            metadata["circle_id"] = np.asarray(circle_ids_raw, dtype=int)[group_id]

        if release_times is not None:
            release_times = np.repeat(release_times, group_size)
    else:
        # Single mode: trivial metadata (one particle per base center)
        metadata = {
            "group_id": np.arange(len(lons_raw), dtype=int),
            "group_member": np.ones(len(lons_raw), dtype=int),
            "group_size": np.ones(len(lons_raw), dtype=int),
        }
        if circle_ids_raw is not None:
            metadata["circle_id"] = np.asarray(circle_ids_raw, dtype=int)

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

    return lons_raw, lats_raw, depths_from_circle_3d, metadata, release_times


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
    max_age_seconds = _get_scheduled_release_max_age_seconds(
        cfg,
        has_release_schedule=release_times is not None,
    )
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

    if max_age_seconds is not None:
        for particle in pset:
            particle.release_time = particle.time

    print(f"ParticleSet created with {len(lons)} particles")

    if release_times is not None:
        print(
            f"Particle release schedule: {release_times.min()} -> {release_times.max()} "
            f"({len(np.unique(release_times))} timesteps)"
        )
    elif start_time is not None:
        print(f"Particle release start_time = {start_time}")

    return pset


def _ordered_unique_particle_times(pset: ParticleSet, *, forward_time: bool) -> np.ndarray:
    particledata = getattr(pset, "particledata", None)
    if particledata is None:
        return np.empty(0, dtype=float)

    particle_times = np.asarray(particledata.getvardata("time"), dtype=float)
    particle_times = particle_times[np.isfinite(particle_times)]
    if particle_times.size == 0:
        return np.empty(0, dtype=float)

    unique_times = np.unique(particle_times)
    return unique_times if forward_time else unique_times[::-1]


def _write_initial_release_snapshots(pset: ParticleSet, output_file, *, forward_time: bool) -> None:
    if not hasattr(output_file, "write"):
        return

    release_times = _ordered_unique_particle_times(pset, forward_time=forward_time)
    if release_times.size == 0:
        return

    for release_time in release_times:
        output_file.write(pset, time=float(release_time))

    print(
        "Wrote explicit initial release snapshot(s) for "
        f"{len(release_times)} release time(s) before integration"
    )


def _zarr_fill_value(dtype: np.dtype) -> float | int | bool:
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    if np.issubdtype(dtype, np.bool_):
        return False
    raise TypeError(f"Unsupported zarr dtype for row compaction: {dtype}")


def _compact_duplicate_initial_zarr_records(zarr_path: Path) -> None:
    """
    Remove Parcels' duplicated first record after an explicit release snapshot.

    Parcels writes the state at `time_at_startofloop` after the first integration
    step, so when the runner writes an explicit release snapshot up front the
    output file contains a duplicated first timestamp. Compact each affected
    trajectory row in place so `obs=0` remains the true release state and later
    observations stay contiguous.
    """
    z = zarr.open(str(zarr_path), mode="r+")
    if "time" not in z:
        return

    time_values = z["time"][:]
    if time_values.ndim != 2:
        return

    duplicate_rows: dict[int, np.ndarray] = {}
    for row_index, row in enumerate(time_values):
        valid = np.flatnonzero(np.isfinite(row))
        if len(valid) < 2:
            continue
        if np.isclose(row[valid[0]], row[valid[1]], atol=1.0e-6, rtol=0.0):
            duplicate_rows[row_index] = np.concatenate(([valid[0]], valid[2:]))

    if not duplicate_rows:
        return

    vars_to_compact = [
        name
        for name in z.array_keys()
        if z[name].ndim == 2 and z[name].shape == time_values.shape
    ]

    for var_name in vars_to_compact:
        arr = z[var_name]
        try:
            fill_value = _zarr_fill_value(arr.dtype)
        except TypeError:
            continue

        for row_index, keep_indices in duplicate_rows.items():
            row = arr[row_index, :]
            compacted = np.full(row.shape, fill_value, dtype=arr.dtype)
            compacted[: len(keep_indices)] = row[keep_indices]
            arr[row_index, :] = compacted

    print(
        f"[zarr repair] Compacted duplicate initial record(s) for {len(duplicate_rows)} "
        f"trajectory row(s) across {len(vars_to_compact)} variable(s)."
    )

def _nullify_off_grid_zarr_records(zarr_path: Path, *, outputdt_s: float) -> None:
    """
    Nullify zarr records whose timestamps are not aligned to the outputdt grid.

    When Parcels deletes a particle (beaching, boundary kill) it writes the final
    state at the deletion time, which is a dt-multiple but not necessarily an
    outputdt-multiple.  This produces isolated off-grid records.

    This function sets lon, lat, time (and z / member-position variables if present)
    to NaN at those cells.  The standard `truncate_at_first_invalid` step in the
    postprocessing pipeline then drops them, so the particle's last recorded
    position is always the last on-grid outputdt snapshot before deletion.
    """
    z = zarr.open(str(zarr_path), mode="r+")
    if "time" not in z:
        return

    t = z["time"][:]           # float64 (n_traj, n_obs) — seconds since Parcels epoch
    valid = ~np.isnan(t)
    if not valid.any():
        return

    # A record is off-grid if its time value is not within 1 s of an outputdt multiple.
    # Use a fixed 1-second tolerance (outputdt is at least several minutes).
    remainder = t[valid] % outputdt_s
    off_grid_flat = (remainder > 1.0) & (remainder < outputdt_s - 1.0)

    off_grid_2d = np.zeros(t.shape, dtype=bool)
    off_grid_2d[valid] = off_grid_flat

    if not off_grid_2d.any():
        return

    n_off = int(off_grid_2d.sum())

    # Variables to nullify: all 2D float arrays (skip 1D index arrays trajectory/obs
    # and integer metadata that cannot hold NaN).
    _FLOAT_VARS = ["time", "lon", "lat", "z",
                   "center_lon", "center_lat",
                   "lon_1", "lat_1", "lon_2", "lat_2",
                   "lon_3", "lat_3", "lon_4", "lat_4"]
    vars_to_nullify = [
        v for v in _FLOAT_VARS
        if v in z and z[v].ndim == 2 and np.issubdtype(z[v].dtype, np.floating)
    ]

    for var in vars_to_nullify:
        arr = z[var][:]
        arr[off_grid_2d] = np.nan
        z[var][:] = arr

    print(
        f"[zarr repair] Nullified {n_off} off-grid record(s) in {len(vars_to_nullify)} "
        f"variable(s) (outputdt = {outputdt_s:.0f} s = {outputdt_s / 3600:.4g} h). "
        "Their last on-grid snapshot is preserved."
    )


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
    max_age_seconds = _get_scheduled_release_max_age_seconds(cfg)

    grouped_entity_mode = _is_group_entity_mode(cfg)
    boundary_halo_enabled = _is_boundary_halo_enabled(cfg)
    periodic_wrap_enabled = _needs_periodic_wrap(cfg)

    if max_age_seconds is not None:
        fieldset.add_constant("kp_max_age_seconds", max_age_seconds)
        print(
            "Max particle age enabled: "
            f"{max_age_seconds / 86400.0:.4g} days ({max_age_seconds:.0f} s)"
        )

    output_file = pset.ParticleFile(
        name=str(zarr_path),
        outputdt=timedelta(hours=dt_output_hours),
    )
    _write_initial_release_snapshots(
        pset,
        output_file,
        forward_time=dt_integration_hours >= 0,
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
        kernels = None
        if periodic_wrap_enabled:
            kernels = pset.Kernel(WrapLongitudePeriodic_GroupedEntity)
        if max_age_seconds is not None:
            age_kernel = pset.Kernel(DeleteParticleIfTooOld)
            kernels = age_kernel if kernels is None else kernels + age_kernel
        if boundary_halo_enabled:
            boundary_kernel = pset.Kernel(BoundaryHaloKill_GroupedEntity)
            kernels = boundary_kernel if kernels is None else kernels + boundary_kernel
        advection_kernel = pset.Kernel(grouped_kernel)
        kernels = advection_kernel if kernels is None else kernels + advection_kernel

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
        kernels = None
        if periodic_wrap_enabled:
            kernels = pset.Kernel(WrapLongitudePeriodic)
        if max_age_seconds is not None:
            age_kernel = pset.Kernel(DeleteParticleIfTooOld)
            kernels = age_kernel if kernels is None else kernels + age_kernel
        if boundary_halo_enabled:
            boundary_kernel = pset.Kernel(BoundaryHaloKill)
            kernels = boundary_kernel if kernels is None else kernels + boundary_kernel
        advection_kernel = pset.Kernel(kernel_func)
        kernels = advection_kernel if kernels is None else kernels + advection_kernel
        print(f"LKM enabled: {lkm_modes.n_modes} modes, update every {update_freq} steps")

        total_steps = int(abs(runtime_days * 24 / dt_sync_hours))
        for _ in range(total_steps):
            update_group_centers_and_relative_coords(
                pset=pset,
                apply_to_group_size_min=lkm_cfg.get('apply_to_group_size_min', 2),
            )
            pset.execute(
                kernels,
                runtime=timedelta(hours=abs(dt_sync_hours)),
                dt=timedelta(hours=dt_integration_hours),
                output_file=output_file,
            )
    else:
        kernels = None
        if periodic_wrap_enabled:
            kernels = pset.Kernel(WrapLongitudePeriodic)
        if max_age_seconds is not None:
            age_kernel = pset.Kernel(DeleteParticleIfTooOld)
            kernels = age_kernel if kernels is None else kernels + age_kernel
        if boundary_halo_enabled:
            boundary_kernel = pset.Kernel(BoundaryHaloKill)
            kernels = boundary_kernel if kernels is None else kernels + boundary_kernel
        print("LKM disabled: using standard advection")
        if kernels is None:
            pset.execute(
                AdvectionRK4,
                runtime=timedelta(days=runtime_days),
                dt=timedelta(hours=dt_integration_hours),
                output_file=output_file,
            )
        else:
            kernels += pset.Kernel(AdvectionRK4)
            pset.execute(
                kernels,
                runtime=timedelta(days=runtime_days),
                dt=timedelta(hours=dt_integration_hours),
                output_file=output_file,
            )

    # Parcels appends ".zarr" to the output name if the path has no suffix.
    actual_zarr_path = zarr_path if zarr_path.exists() else zarr_path.parent / (zarr_path.name + ".zarr")
    if actual_zarr_path.exists():
        _compact_duplicate_initial_zarr_records(actual_zarr_path)
        _nullify_off_grid_zarr_records(
            actual_zarr_path,
            outputdt_s=timedelta(hours=dt_output_hours).total_seconds(),
        )


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
