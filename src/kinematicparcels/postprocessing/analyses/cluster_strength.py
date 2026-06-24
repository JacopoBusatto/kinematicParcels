from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ..core.distances import (
    EARTH_RADIUS_KM,
    haversine_km,
    local_equirectangular_xy_km,
    wrap_lon_delta_deg,
)
from ..core.gridding import RegularGrid

try:  # Optional: present in many scientific environments, but not required.
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only when scipy is absent.
    cKDTree = None


DEFAULT_CUTOFF_FACTOR = 4.0
SUPPORTED_DISTANCES = {"haversine", "euclidean"}


def _warn_scipy_missing() -> None:
    warnings.warn(
        "scipy.spatial.cKDTree is unavailable; cluster_strength is using a slower "
        "chunked fallback. Install scipy for efficient neighbor queries.",
        RuntimeWarning,
        stacklevel=3,
    )


class DistanceBackend(Protocol):
    name: str
    candidate_search: str

    def compute_strength(
        self,
        *,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        particle_lon: np.ndarray,
        particle_lat: np.ndarray,
        scale_km: float,
        cutoff_km: float,
        grid_chunk_size: int,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HaversineDistanceBackend:
    name: str = "haversine"
    candidate_search: str = "scipy.spatial.cKDTree local equirectangular query"

    def __post_init__(self) -> None:
        if cKDTree is None:
            object.__setattr__(
                self,
                "candidate_search",
                "chunked lon/lat bounding-box query; exact haversine final distance",
            )
            _warn_scipy_missing()

    def compute_strength(
        self,
        *,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        particle_lon: np.ndarray,
        particle_lat: np.ndarray,
        scale_km: float,
        cutoff_km: float,
        grid_chunk_size: int,
    ) -> np.ndarray:
        if target_lon.size == 0:
            return np.array([], dtype=float)

        out = np.zeros(target_lon.size, dtype=float)
        if particle_lon.size == 0:
            return out

        if cKDTree is None:
            return self._compute_strength_without_scipy(
                target_lon=target_lon,
                target_lat=target_lat,
                particle_lon=particle_lon,
                particle_lat=particle_lat,
                scale_km=scale_km,
                cutoff_km=cutoff_km,
                grid_chunk_size=grid_chunk_size,
            )

        lon0 = float(np.nanmean(np.concatenate([target_lon, particle_lon])))
        lat0 = float(np.nanmean(np.concatenate([target_lat, particle_lat])))
        particle_xy = local_equirectangular_xy_km(particle_lon, particle_lat, lon0=lon0, lat0=lat0)
        target_xy = local_equirectangular_xy_km(target_lon, target_lat, lon0=lon0, lat0=lat0)

        tree = cKDTree(particle_xy)
        query_radius_km = cutoff_km * 1.02 + 1.0e-9

        for start in range(0, target_lon.size, grid_chunk_size):
            stop = min(start + grid_chunk_size, target_lon.size)
            neighbor_lists = tree.query_ball_point(target_xy[start:stop], r=query_radius_km)

            for local_idx, particle_indices in enumerate(neighbor_lists):
                if len(particle_indices) == 0:
                    continue

                target_idx = start + local_idx
                candidate_idx = np.asarray(particle_indices, dtype=int)
                distances = haversine_km(
                    target_lon[target_idx],
                    target_lat[target_idx],
                    particle_lon[candidate_idx],
                    particle_lat[candidate_idx],
                )
                inside = distances <= cutoff_km
                if np.any(inside):
                    out[target_idx] = float(np.exp(-((distances[inside] / scale_km) ** 2)).sum())

        return out

    def _compute_strength_without_scipy(
        self,
        *,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        particle_lon: np.ndarray,
        particle_lat: np.ndarray,
        scale_km: float,
        cutoff_km: float,
        grid_chunk_size: int,
    ) -> np.ndarray:
        out = np.zeros(target_lon.size, dtype=float)
        lat_radius_deg = np.rad2deg(cutoff_km / EARTH_RADIUS_KM)

        for start in range(0, target_lon.size, grid_chunk_size):
            stop = min(start + grid_chunk_size, target_lon.size)
            for target_idx in range(start, stop):
                cos_lat = max(abs(np.cos(np.deg2rad(target_lat[target_idx]))), 1.0e-6)
                lon_radius_deg = np.rad2deg(cutoff_km / (EARTH_RADIUS_KM * cos_lat))
                candidate_mask = (
                    (np.abs(particle_lat - target_lat[target_idx]) <= lat_radius_deg)
                    & (np.abs(wrap_lon_delta_deg(particle_lon, target_lon[target_idx])) <= lon_radius_deg)
                )
                if not np.any(candidate_mask):
                    continue

                distances = haversine_km(
                    target_lon[target_idx],
                    target_lat[target_idx],
                    particle_lon[candidate_mask],
                    particle_lat[candidate_mask],
                )
                inside = distances <= cutoff_km
                if np.any(inside):
                    out[target_idx] = float(np.exp(-((distances[inside] / scale_km) ** 2)).sum())

        return out


@dataclass(frozen=True)
class EuclideanDistanceBackend:
    name: str = "euclidean"
    candidate_search: str = "scipy.spatial.cKDTree local equirectangular query"

    def __post_init__(self) -> None:
        if cKDTree is None:
            object.__setattr__(
                self,
                "candidate_search",
                "chunked local equirectangular Euclidean query",
            )
            _warn_scipy_missing()

    def compute_strength(
        self,
        *,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        particle_lon: np.ndarray,
        particle_lat: np.ndarray,
        scale_km: float,
        cutoff_km: float,
        grid_chunk_size: int,
    ) -> np.ndarray:
        if target_lon.size == 0:
            return np.array([], dtype=float)

        out = np.zeros(target_lon.size, dtype=float)
        if particle_lon.size == 0:
            return out

        lon0 = float(np.nanmean(np.concatenate([target_lon, particle_lon])))
        lat0 = float(np.nanmean(np.concatenate([target_lat, particle_lat])))
        particle_xy = local_equirectangular_xy_km(particle_lon, particle_lat, lon0=lon0, lat0=lat0)
        target_xy = local_equirectangular_xy_km(target_lon, target_lat, lon0=lon0, lat0=lat0)

        if cKDTree is not None:
            tree = cKDTree(particle_xy)
            for start in range(0, target_lon.size, grid_chunk_size):
                stop = min(start + grid_chunk_size, target_lon.size)
                neighbor_lists = tree.query_ball_point(target_xy[start:stop], r=cutoff_km)

                for local_idx, particle_indices in enumerate(neighbor_lists):
                    if len(particle_indices) == 0:
                        continue

                    target_idx = start + local_idx
                    candidate_idx = np.asarray(particle_indices, dtype=int)
                    distances = np.linalg.norm(particle_xy[candidate_idx] - target_xy[target_idx], axis=1)
                    out[target_idx] = float(np.exp(-((distances / scale_km) ** 2)).sum())
            return out

        for start in range(0, target_lon.size, grid_chunk_size):
            stop = min(start + grid_chunk_size, target_lon.size)
            for target_idx in range(start, stop):
                delta = particle_xy - target_xy[target_idx]
                candidate_mask = (np.abs(delta[:, 0]) <= cutoff_km) & (np.abs(delta[:, 1]) <= cutoff_km)
                if not np.any(candidate_mask):
                    continue

                distances = np.linalg.norm(delta[candidate_mask], axis=1)
                inside = distances <= cutoff_km
                if np.any(inside):
                    out[target_idx] = float(np.exp(-((distances[inside] / scale_km) ** 2)).sum())

        return out


def build_distance_backend(distance: str) -> DistanceBackend:
    if distance == "haversine":
        return HaversineDistanceBackend()
    if distance == "euclidean":
        return EuclideanDistanceBackend()
    raise ValueError(
        "cluster_strength.distance must be lowercase and one of: "
        f"{sorted(SUPPORTED_DISTANCES)}."
    )


def _particle_group_cols(df: pd.DataFrame) -> list[str]:
    cols = ["trajectory"]
    if "group_member" in df.columns:
        cols.append("group_member")
    return cols


def _elapsed_days_from_release(
    time_values: pd.Series,
    release_time_values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(time_values):
        delta = pd.to_datetime(time_values) - pd.to_datetime(release_time_values)
        return delta.dt.total_seconds() / 86400.0

    numeric_time = pd.to_numeric(time_values)
    numeric_release = pd.to_numeric(release_time_values)
    return numeric_time - numeric_release


def _infer_simulation_direction(work: pd.DataFrame, *, particle_cols: list[str], time_col: str) -> str:
    two_rows = (
        work.sort_values(particle_cols + ["obs"])
        .groupby(particle_cols, sort=False)
        .head(2)
    )
    if two_rows.empty:
        return "forward"

    deltas: list[float] = []
    for _, particle_data in two_rows.groupby(particle_cols, sort=False):
        if len(particle_data) < 2:
            continue
        first_time = particle_data.iloc[0][time_col]
        second_time = particle_data.iloc[1][time_col]
        delta = second_time - first_time
        if hasattr(delta, "total_seconds"):
            delta_value = float(delta.total_seconds())
        else:
            delta_value = float(delta)
        if delta_value != 0:
            deltas.append(delta_value)

    if not deltas:
        return "forward"

    has_forward = any(delta > 0 for delta in deltas)
    has_backward = any(delta < 0 for delta in deltas)
    if has_forward and has_backward:
        warnings.warn(
            "Cluster-strength input contains both forward and backward particle time ordering; "
            "using 'mixed' simulation_direction.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "mixed"
    return "backward" if has_backward else "forward"


def _attach_release_metadata(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, str]:
    required = ["trajectory", "obs", lon_col, lat_col, time_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.loc[work[time_col].notna()].copy()
    if work.empty:
        return work, "forward"

    particle_cols = _particle_group_cols(work)
    work = work.sort_values(particle_cols + ["obs"]).reset_index(drop=True)
    simulation_direction = _infer_simulation_direction(
        work,
        particle_cols=particle_cols,
        time_col=time_col,
    )

    release_meta = (
        work.groupby(particle_cols, sort=False)
        .first()
        .reset_index()[particle_cols + [time_col]]
        .rename(columns={time_col: "release_time"})
    )
    work = work.merge(release_meta, on=particle_cols, how="left")
    work["age_days"] = _elapsed_days_from_release(work[time_col], work["release_time"])
    return work, simulation_direction


def _empty_dataset(grid: RegularGrid, *, attrs: dict[str, object]) -> xr.Dataset:
    data = np.full((0, 0, grid.nlat, grid.nlon), np.nan, dtype=float)
    ds = xr.Dataset(
        data_vars={"cluster_strength": (("release_time", "age_days", "lat", "lon"), data)},
        coords={
            "release_time": np.array([], dtype="datetime64[ns]"),
            "age_days": np.array([], dtype=float),
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs=attrs,
    )
    return ds


def compute_cluster_strength(
    df: pd.DataFrame,
    *,
    grid: RegularGrid,
    scale_km: float,
    distance: str = "haversine",
    mask: bool = True,
    cutoff_factor: float = DEFAULT_CUTOFF_FACTOR,
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    max_group_member: int | None = 1,
    grid_chunk_size: int = 4096,
) -> xr.Dataset:
    """
    Compute Huntley et al. (2015) cluster strength on a regular lon/lat grid.

    C(x*, t) = sum_n exp(- (d(x*, x_n(t)) / L)^2 )
    """
    required = ["trajectory", "obs", lon_col, lat_col, time_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    if scale_km <= 0:
        raise ValueError("scale_km must be positive.")
    if cutoff_factor <= 0:
        raise ValueError("cutoff_factor must be positive.")
    if max_group_member is not None and max_group_member <= 0:
        raise ValueError("max_group_member must be an integer > 0 or None.")
    if grid_chunk_size < 1:
        raise ValueError("grid_chunk_size must be >= 1.")

    backend = build_distance_backend(distance)
    cutoff_km = float(cutoff_factor * scale_km)
    common_attrs: dict[str, object] = {
        "grid_type": "regular_lonlat",
        "lon_min": grid.lon_min,
        "lon_max": grid.lon_max,
        "lat_min": grid.lat_min,
        "lat_max": grid.lat_max,
        "dlon": grid.dlon,
        "dlat": grid.dlat,
        "scale_km": float(scale_km),
        "distance": backend.name,
        "formula": "C(x*, t) = sum_n exp(- (d(x*, x_n(t)) / L)^2 )",
        "cutoff_factor": float(cutoff_factor),
        "cutoff_km": cutoff_km,
        "mask": bool(mask),
        "max_group_member": "all" if max_group_member is None else int(max_group_member),
        "age_definition": "signed days since release_time",
    }

    if df.empty:
        ds = _empty_dataset(
            grid,
            attrs={
                **common_attrs,
                "candidate_search": backend.candidate_search,
                "simulation_direction": "forward",
            },
        )
        return ds

    work = df.copy()
    if max_group_member is not None and "group_member" in work.columns:
        work = work.loc[work["group_member"] <= max_group_member].copy()

    work, simulation_direction = _attach_release_metadata(
        work,
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
    )

    if work.empty:
        ds = _empty_dataset(
            grid,
            attrs={
                **common_attrs,
                "candidate_search": backend.candidate_search,
                "simulation_direction": simulation_direction,
            },
        )
        return ds

    release_time_values = pd.DatetimeIndex(work["release_time"].unique()).sort_values()
    age_values = np.array(sorted(float(age) for age in pd.unique(work["age_days"])), dtype=float)
    finite = work.loc[
        np.isfinite(work[lon_col].to_numpy(dtype=float))
        & np.isfinite(work[lat_col].to_numpy(dtype=float))
    ].copy()

    valid_cell_mask = np.ones((grid.nlat, grid.nlon), dtype=bool)
    if mask:
        valid_cell_mask[:, :] = False
        visited = grid.assign_bins(
            finite,
            lon_col=lon_col,
            lat_col=lat_col,
            drop_outside=True,
        )
        if not visited.empty:
            valid_cell_mask[
                visited["lat_bin"].to_numpy(dtype=int),
                visited["lon_bin"].to_numpy(dtype=int),
            ] = True

    valid_flat_indices = np.flatnonzero(valid_cell_mask.ravel())
    lon2d, lat2d = np.meshgrid(grid.lon_centers, grid.lat_centers)
    target_lon = lon2d.ravel()[valid_flat_indices]
    target_lat = lat2d.ravel()[valid_flat_indices]
    valid_lat_bins = valid_flat_indices // grid.nlon
    valid_lon_bins = valid_flat_indices % grid.nlon

    data = np.full((len(release_time_values), len(age_values), grid.nlat, grid.nlon), np.nan, dtype=float)
    release_lookup = {pd.Timestamp(value): idx for idx, value in enumerate(release_time_values)}
    age_lookup = {float(value): idx for idx, value in enumerate(age_values)}
    observed_release_ages = [
        (pd.Timestamp(release_time), float(age_days))
        for release_time, age_days in (
            work[["release_time", "age_days"]]
            .drop_duplicates()
            .sort_values(["release_time", "age_days"])
            .itertuples(index=False, name=None)
        )
    ]
    finite_by_release_age = {
        (pd.Timestamp(release_time), float(age_days)): group
        for (release_time, age_days), group in finite.groupby(
            ["release_time", "age_days"],
            sort=False,
            observed=False,
        )
    }

    for release_time, age_days in observed_release_ages:
        release_idx = release_lookup[pd.Timestamp(release_time)]
        age_idx = age_lookup[float(age_days)]
        step = finite_by_release_age.get((release_time, age_days))
        if step is None or step.empty:
            strengths = np.zeros(target_lon.size, dtype=float)
        else:
            strengths = backend.compute_strength(
                target_lon=target_lon,
                target_lat=target_lat,
                particle_lon=step[lon_col].to_numpy(dtype=float),
                particle_lat=step[lat_col].to_numpy(dtype=float),
                scale_km=scale_km,
                cutoff_km=cutoff_km,
                grid_chunk_size=grid_chunk_size,
            )

        if valid_flat_indices.size > 0:
            data[release_idx, age_idx, valid_lat_bins, valid_lon_bins] = strengths

    ds = xr.Dataset(
        data_vars={"cluster_strength": (("release_time", "age_days", "lat", "lon"), data)},
        coords={
            "release_time": release_time_values,
            "age_days": age_values,
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs={
            **common_attrs,
            "candidate_search": backend.candidate_search,
            "simulation_direction": simulation_direction,
        },
    )
    ds["cluster_strength"].attrs.update(
        {
            "long_name": "cluster strength",
            "units": "1",
            "scale_km": float(scale_km),
            "distance": backend.name,
            "cutoff_km": cutoff_km,
        }
    )
    ds["age_days"].attrs.update(
        {
            "long_name": "signed age since release",
            "unit_label": "days",
            "description": "signed elapsed time from release_time in days",
        }
    )

    return ds
