from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels import __version__

from ..config.models import GriddedTransitionMatrixConfig
from ..core.gridding import RegularGrid


_TRANSITION_TABLE_DTYPES = {
    "start_lon_bin": "int64",
    "start_lat_bin": "int64",
    "end_lon_bin": "int64",
    "end_lat_bin": "int64",
    "start_lon_center": "float64",
    "start_lat_center": "float64",
    "end_lon_center": "float64",
    "end_lat_center": "float64",
    "transition_count": "int64",
    "transition_probability": "float64",
}

_CARDINAL_DIRECTIONS = ("north", "east", "south", "west")
_DIRECTION_BOUNDARY_ATOL_DEGREES = 1.0e-10
_UNDEFINED_BEARING_ATOL = 1.0e-12


@dataclass(frozen=True)
class GriddedTransitionMatrixResult:
    transition_table: pd.DataFrame
    dataset: xr.Dataset


def _particle_group_cols(df: pd.DataFrame, trajectory_col: str) -> list[str]:
    cols = [trajectory_col]
    if "group_member" in df.columns:
        cols.append("group_member")
    return cols


def _empty_transition_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in _TRANSITION_TABLE_DTYPES.items()
        }
    )


def _longitude_convention_for_grid(grid: RegularGrid) -> str:
    if grid.lon_min >= 0.0 and grid.lon_max > 180.0:
        return "0_360"
    if grid.lon_min < 0.0 and grid.lon_max <= 180.0:
        return "-180_180"
    return "wrapped_to_grid_bounds"


def _normalize_longitudes_to_grid(lon: np.ndarray, grid: RegularGrid) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    out = lon.copy()
    width = float(grid.lon_max - grid.lon_min)
    finite = np.isfinite(out)
    if not np.isclose(width, 360.0, rtol=0.0, atol=1.0e-9):
        return out

    out[finite] = ((out[finite] - grid.lon_min) % 360.0) + grid.lon_min
    too_high = finite & (out >= grid.lon_max)
    out[too_high] = np.nextafter(grid.lon_max, grid.lon_min)
    return out


def _timestep_to_ns(value: float, unit: str) -> int:
    seconds_per_unit = {
        "seconds": 1.0,
        "hours": 3600.0,
        "days": 86400.0,
    }
    ns = int(round(value * seconds_per_unit[unit] * 1.0e9))
    if ns <= 0:
        raise ValueError("gridded_transition_matrix.timestep is too small.")
    return ns


def _prepare_trajectory(
    group: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    time_col: str,
    obs_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    work = group.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.loc[
        work[lon_col].notna() & work[lat_col].notna() & work[time_col].notna()
    ].copy()

    if work.empty:
        return None

    sort_cols = [time_col]
    if obs_col is not None and obs_col in work.columns:
        sort_cols.append(obs_col)
    work = work.sort_values(sort_cols, kind="stable").copy()
    work = work.drop_duplicates(subset=[time_col], keep="first")

    if len(work) < 2:
        return None

    lon = work[lon_col].to_numpy(dtype=float)
    lat = work[lat_col].to_numpy(dtype=float)
    time_ns = work[time_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    return lon, lat, time_ns


def _infer_source_timestep_ns(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    time_col: str,
    obs_col: str | None,
) -> int | None:
    gaps: list[np.ndarray] = []
    for _, group in df.groupby(group_cols, sort=False, observed=False):
        work = group.copy()
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.loc[work[time_col].notna()].copy()
        if len(work) < 2:
            continue
        sort_cols = [time_col]
        if obs_col is not None and obs_col in work.columns:
            sort_cols.append(obs_col)
        time_ns = (
            work.sort_values(sort_cols, kind="stable")
            .drop_duplicates(subset=[time_col], keep="first")[time_col]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )
        diff = np.diff(time_ns)
        diff = diff[diff > 0]
        if diff.size:
            gaps.append(diff)

    if not gaps:
        return None

    all_gaps = np.concatenate(gaps)
    values, counts = np.unique(all_gaps, return_counts=True)
    return int(values[np.argmax(counts)])


def _validate_requested_timestep(
    requested_ns: int,
    source_ns: int | None,
) -> None:
    if source_ns is None or requested_ns <= source_ns:
        return

    ratio = requested_ns / float(source_ns)
    nearest = round(ratio)
    if nearest < 1 or not np.isclose(ratio, nearest, rtol=1.0e-9, atol=1.0e-9):
        source_td = pd.to_timedelta(source_ns, unit="ns")
        requested_td = pd.to_timedelta(requested_ns, unit="ns")
        raise ValueError(
            "gridded_transition_matrix.timestep must be an integer multiple of "
            f"the inferred source timestep when it is larger than source. "
            f"Requested {requested_td}; inferred source {source_td}."
        )


def _interpolate_endpoint(
    lon: np.ndarray,
    lat: np.ndarray,
    time_ns: np.ndarray,
    *,
    start_idx: int,
    target_time_ns: int,
    grid: RegularGrid,
) -> tuple[float, float] | None:
    if target_time_ns > int(time_ns[-1]):
        return None

    right_idx = int(np.searchsorted(time_ns, target_time_ns, side="left"))
    if right_idx >= len(time_ns):
        return None

    if int(time_ns[right_idx]) == target_time_ns:
        return float(lon[right_idx]), float(lat[right_idx])

    left_idx = right_idx - 1
    if left_idx < start_idx or left_idx < 0:
        return None

    t0 = int(time_ns[left_idx])
    t1 = int(time_ns[right_idx])
    if t1 <= t0:
        return None

    lon_pair = np.rad2deg(
        np.unwrap(np.deg2rad(lon[[left_idx, right_idx]].astype(float)))
    )
    frac = (target_time_ns - t0) / float(t1 - t0)
    end_lon = lon_pair[0] + frac * (lon_pair[1] - lon_pair[0])
    end_lat = float(lat[left_idx]) + frac * (float(lat[right_idx]) - float(lat[left_idx]))
    return float(_normalize_longitudes_to_grid(np.asarray([end_lon]), grid)[0]), end_lat


def _iter_segment_points(
    lon: np.ndarray,
    lat: np.ndarray,
    time_ns: np.ndarray,
    *,
    timestep_ns: int | None,
    resample: bool,
    grid: RegularGrid,
) -> list[tuple[float, float, float, float]]:
    if timestep_ns is None:
        return [
            (float(lon[i]), float(lat[i]), float(lon[i + 1]), float(lat[i + 1]))
            for i in range(len(lon) - 1)
        ]

    if resample:
        n_segments = int((int(time_ns[-1]) - int(time_ns[0])) // timestep_ns)
        segments: list[tuple[float, float, float, float]] = []
        start_lon = float(lon[0])
        start_lat = float(lat[0])
        for segment_idx in range(n_segments):
            target_time_ns = int(time_ns[0]) + (segment_idx + 1) * timestep_ns
            endpoint = _interpolate_endpoint(
                lon,
                lat,
                time_ns,
                start_idx=0,
                target_time_ns=target_time_ns,
                grid=grid,
            )
            if endpoint is None:
                break
            end_lon, end_lat = endpoint
            segments.append((start_lon, start_lat, end_lon, end_lat))
            start_lon, start_lat = end_lon, end_lat
        return segments

    segments: list[tuple[float, float, float, float]] = []
    for start_idx in range(len(lon) - 1):
        target_time_ns = int(time_ns[start_idx]) + timestep_ns
        endpoint = _interpolate_endpoint(
            lon,
            lat,
            time_ns,
            start_idx=start_idx,
            target_time_ns=target_time_ns,
            grid=grid,
        )
        if endpoint is None:
            continue
        end_lon, end_lat = endpoint
        segments.append((float(lon[start_idx]), float(lat[start_idx]), end_lon, end_lat))
    return segments


def _assign_bins(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    grid: RegularGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_norm = _normalize_longitudes_to_grid(lon, grid)
    lat = np.asarray(lat, dtype=float)
    lon_bin = np.floor((lon_norm - grid.lon_min) / grid.dlon).astype(int)
    lat_bin = np.floor((lat - grid.lat_min) / grid.dlat).astype(int)
    valid = (
        np.isfinite(lon_norm)
        & np.isfinite(lat)
        & (lon_norm >= grid.lon_min)
        & (lon_norm < grid.lon_max)
        & (lat >= grid.lat_min)
        & (lat < grid.lat_max)
        & (lon_bin >= 0)
        & (lon_bin < grid.nlon)
        & (lat_bin >= 0)
        & (lat_bin < grid.nlat)
    )
    return lon_bin, lat_bin, valid


def _build_transition_table(
    grid: RegularGrid,
    *,
    start_lon_bin: np.ndarray,
    start_lat_bin: np.ndarray,
    end_lon_bin: np.ndarray,
    end_lat_bin: np.ndarray,
    transition_count: np.ndarray,
    start_counts_flat: np.ndarray,
) -> pd.DataFrame:
    if transition_count.size == 0:
        return _empty_transition_table()

    start_state = start_lat_bin * grid.nlon + start_lon_bin
    denominator = start_counts_flat[start_state].astype(float)
    probability = transition_count.astype(float) / denominator

    return pd.DataFrame(
        {
            "start_lon_bin": start_lon_bin.astype(int),
            "start_lat_bin": start_lat_bin.astype(int),
            "end_lon_bin": end_lon_bin.astype(int),
            "end_lat_bin": end_lat_bin.astype(int),
            "start_lon_center": grid.lon_min + (start_lon_bin + 0.5) * grid.dlon,
            "start_lat_center": grid.lat_min + (start_lat_bin + 0.5) * grid.dlat,
            "end_lon_center": grid.lon_min + (end_lon_bin + 0.5) * grid.dlon,
            "end_lat_center": grid.lat_min + (end_lat_bin + 0.5) * grid.dlat,
            "transition_count": transition_count.astype(np.int64),
            "transition_probability": probability,
        }
    )


def _spherical_initial_bearing_degrees(
    start_lon: np.ndarray,
    start_lat: np.ndarray,
    end_lon: np.ndarray,
    end_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial great-circle bearings and an undefined-bearing mask."""
    start_lon = np.asarray(start_lon, dtype=float)
    start_lat = np.asarray(start_lat, dtype=float)
    end_lon = np.asarray(end_lon, dtype=float)
    end_lat = np.asarray(end_lat, dtype=float)

    phi1 = np.deg2rad(start_lat)
    phi2 = np.deg2rad(end_lat)
    delta_lon = ((end_lon - start_lon + 180.0) % 360.0) - 180.0
    delta_lambda = np.deg2rad(delta_lon)

    x = np.sin(delta_lambda) * np.cos(phi2)
    y = (
        np.cos(phi1) * np.sin(phi2)
        - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    )
    undefined = np.hypot(x, y) <= _UNDEFINED_BEARING_ATOL
    bearing = (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0
    bearing[undefined] = np.nan
    return bearing, undefined


def _cardinal_sector_weights(
    bearing: np.ndarray,
    *,
    undefined: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Partition bearings into cardinal sectors, sharing exact boundaries."""
    bearing = np.asarray(bearing, dtype=float)
    if undefined is None:
        undefined = np.zeros(bearing.shape, dtype=bool)
    else:
        undefined = np.asarray(undefined, dtype=bool)

    weights = {
        direction: np.zeros(bearing.shape, dtype=float)
        for direction in _CARDINAL_DIRECTIONS
    }
    finite = np.isfinite(bearing) & ~undefined
    boundary_45 = finite & np.isclose(
        bearing, 45.0, rtol=0.0, atol=_DIRECTION_BOUNDARY_ATOL_DEGREES
    )
    boundary_135 = finite & np.isclose(
        bearing, 135.0, rtol=0.0, atol=_DIRECTION_BOUNDARY_ATOL_DEGREES
    )
    boundary_225 = finite & np.isclose(
        bearing, 225.0, rtol=0.0, atol=_DIRECTION_BOUNDARY_ATOL_DEGREES
    )
    boundary_315 = finite & np.isclose(
        bearing, 315.0, rtol=0.0, atol=_DIRECTION_BOUNDARY_ATOL_DEGREES
    )
    on_boundary = boundary_45 | boundary_135 | boundary_225 | boundary_315
    interior = finite & ~on_boundary

    weights["north"][interior & ((bearing < 45.0) | (bearing > 315.0))] = 1.0
    weights["east"][interior & (bearing > 45.0) & (bearing < 135.0)] = 1.0
    weights["south"][interior & (bearing > 135.0) & (bearing < 225.0)] = 1.0
    weights["west"][interior & (bearing > 225.0) & (bearing < 315.0)] = 1.0

    for boundary, first, second in (
        (boundary_45, "north", "east"),
        (boundary_135, "east", "south"),
        (boundary_225, "south", "west"),
        (boundary_315, "west", "north"),
    ):
        weights[first][boundary] = 0.5
        weights[second][boundary] = 0.5

    for values in weights.values():
        values[undefined] = 0.25
    return weights


def _direction_probability_maps(
    grid: RegularGrid,
    table: pd.DataFrame,
    *,
    start_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    maps = {
        "probability_north": np.full((grid.nlat, grid.nlon), np.nan, dtype=float),
        "probability_south": np.full((grid.nlat, grid.nlon), np.nan, dtype=float),
        "probability_east": np.full((grid.nlat, grid.nlon), np.nan, dtype=float),
        "probability_west": np.full((grid.nlat, grid.nlon), np.nan, dtype=float),
        "probability_stay": np.full((grid.nlat, grid.nlon), np.nan, dtype=float),
    }

    has_starts = start_counts > 0
    for values in maps.values():
        values[has_starts] = 0.0

    if table.empty:
        return maps

    probability = table["transition_probability"].to_numpy(dtype=float)
    start_lat = table["start_lat_bin"].to_numpy(dtype=int)
    start_lon = table["start_lon_bin"].to_numpy(dtype=int)
    end_lat = table["end_lat_bin"].to_numpy(dtype=int)
    end_lon = table["end_lon_bin"].to_numpy(dtype=int)
    stay = (end_lat == start_lat) & (end_lon == start_lon)
    np.add.at(
        maps["probability_stay"],
        (start_lat[stay], start_lon[stay]),
        probability[stay],
    )

    moving = ~stay
    if np.any(moving):
        bearing, undefined = _spherical_initial_bearing_degrees(
            table["start_lon_center"].to_numpy(dtype=float)[moving],
            table["start_lat_center"].to_numpy(dtype=float)[moving],
            table["end_lon_center"].to_numpy(dtype=float)[moving],
            table["end_lat_center"].to_numpy(dtype=float)[moving],
        )
        weights = _cardinal_sector_weights(bearing, undefined=undefined)
        moving_probability = probability[moving]
        moving_start_lat = start_lat[moving]
        moving_start_lon = start_lon[moving]
        for direction, direction_weights in weights.items():
            np.add.at(
                maps[f"probability_{direction}"],
                (moving_start_lat, moving_start_lon),
                moving_probability * direction_weights,
            )

    return maps


def _entropy_map(
    grid: RegularGrid,
    table: pd.DataFrame,
    *,
    start_counts: np.ndarray,
    log_base: str | int,
) -> np.ndarray:
    entropy = np.full((grid.nlat, grid.nlon), np.nan, dtype=float)
    entropy[start_counts > 0] = 0.0
    if table.empty:
        return entropy

    probability = table["transition_probability"].to_numpy(dtype=float)
    positive = probability > 0.0
    if not np.any(positive):
        return entropy

    log_probability = np.log(probability[positive])
    if log_base != "e":
        log_probability = log_probability / np.log(float(log_base))
    contribution = -probability[positive] * log_probability
    np.add.at(
        entropy,
        (
            table["start_lat_bin"].to_numpy(dtype=int)[positive],
            table["start_lon_bin"].to_numpy(dtype=int)[positive],
        ),
        contribution,
    )
    return entropy


def _entropy_units(log_base: str | int) -> str:
    if log_base == 2:
        return "bits"
    if log_base == 10:
        return "hartleys"
    return "nats"


def _build_dataset(
    grid: RegularGrid,
    table: pd.DataFrame,
    *,
    start_counts: np.ndarray,
    timestep_ns: int | None,
    source_timestep_ns: int | None,
    resample: bool,
    entropy_log_base: str | int,
) -> xr.Dataset:
    n_transition = len(table)
    transition_coord = np.arange(n_transition, dtype=np.int64)
    direction_maps = _direction_probability_maps(grid, table, start_counts=start_counts)
    entropy = _entropy_map(
        grid,
        table,
        start_counts=start_counts,
        log_base=entropy_log_base,
    )

    data_vars = {
        "n_segments_start": (("lat", "lon"), start_counts.astype(np.int64)),
        "probability_north": (("lat", "lon"), direction_maps["probability_north"]),
        "probability_south": (("lat", "lon"), direction_maps["probability_south"]),
        "probability_east": (("lat", "lon"), direction_maps["probability_east"]),
        "probability_west": (("lat", "lon"), direction_maps["probability_west"]),
        "probability_stay": (("lat", "lon"), direction_maps["probability_stay"]),
        "entropy": (("lat", "lon"), entropy),
    }

    sparse_columns = [
        "start_lon_bin",
        "start_lat_bin",
        "end_lon_bin",
        "end_lat_bin",
        "start_lon_center",
        "start_lat_center",
        "end_lon_center",
        "end_lat_center",
        "transition_count",
        "transition_probability",
    ]
    for column in sparse_columns:
        if column in table:
            data_vars[column] = (("transition",), table[column].to_numpy())
        else:
            data_vars[column] = (("transition",), np.asarray([], dtype=float))

    attrs = {
        "title": "Sparse gridded transition matrix",
        "summary": (
            "Empirical transition probabilities between regular lon/lat grid cells, "
            "stored as occupied sparse transitions plus start-cell cardinal-sector "
            "summaries and Shannon entropy."
        ),
        "grid_type": "regular_lonlat",
        "lon_min": grid.lon_min,
        "lon_max": grid.lon_max,
        "lat_min": grid.lat_min,
        "lat_max": grid.lat_max,
        "dlon": grid.dlon,
        "dlat": grid.dlat,
        "nlon": grid.nlon,
        "nlat": grid.nlat,
        "n_states": grid.nlon * grid.nlat,
        "n_sparse_transitions": n_transition,
        "timestep_seconds": "native" if timestep_ns is None else timestep_ns / 1.0e9,
        "source_timestep_seconds": (
            "unknown" if source_timestep_ns is None else source_timestep_ns / 1.0e9
        ),
        "segment_start_policy": (
            "regular_non_overlapping" if resample else "every_observation"
        ),
        "trajectory_resampling": "enabled" if resample else "disabled",
        "segment_inclusion": (
            "both start and end points must lie inside the analysis grid"
        ),
        "normalization": (
            "transition_count / n_valid_in_domain_segments_start(start_cell)"
        ),
        "direction_classification": (
            "initial spherical great-circle bearing between start/end grid-cell centers"
        ),
        "direction_bearing_convention": (
            "degrees clockwise from geographic north in [0, 360)"
        ),
        "direction_sector_boundaries_degrees": "45, 135, 225, 315",
        "direction_boundary_rule": (
            "split probability equally between adjacent cardinal sectors"
        ),
        "direction_boundary_tolerance_degrees": _DIRECTION_BOUNDARY_ATOL_DEGREES,
        "undefined_bearing_rule": (
            "split probability equally among all four cardinal sectors"
        ),
        "longitude_convention": _longitude_convention_for_grid(grid),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "software_version": __version__,
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "transition": transition_coord,
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs=attrs,
    )

    ds["n_segments_start"].attrs.update(
        long_name=(
            "number of valid segments with both endpoints inside the analysis domain "
            "starting in grid cell"
        ),
        units="count",
    )
    for direction in _CARDINAL_DIRECTIONS:
        ds[f"probability_{direction}"].attrs.update(
            long_name=(
                f"probability of moving into the {direction} cardinal bearing sector "
                "from start cell"
            ),
            units="1",
        )
    ds["probability_stay"].attrs.update(
        long_name="probability of remaining in start cell",
        units="1",
    )
    ds["entropy"].attrs.update(
        long_name="Shannon entropy of in-domain destination probabilities",
        units=_entropy_units(entropy_log_base),
        log_base=entropy_log_base,
        formula="-sum_j P_ij log_b(P_ij)",
        normalization="not normalized by destination count",
        conditioning="both segment endpoints lie inside the analysis grid",
    )
    ds["transition_count"].attrs.update(long_name="transition count", units="count")
    ds["transition_probability"].attrs.update(long_name="transition probability", units="1")
    return ds


def compute_gridded_transition_matrix(
    df: pd.DataFrame,
    *,
    grid: RegularGrid,
    cfg: GriddedTransitionMatrixConfig,
    trajectory_col: str = "trajectory",
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    obs_col: str = "obs",
) -> GriddedTransitionMatrixResult:
    """
    Compute a sparse transition matrix between regular lon/lat grid cells.

    Only displacement segments whose starting and ending points both lie inside
    the analysis grid contribute to counts and probabilities.
    """
    required = [trajectory_col, lon_col, lat_col, time_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    group_cols = _particle_group_cols(df, trajectory_col)
    obs_col_or_none = obs_col if obs_col in df.columns else None
    source_timestep_ns = _infer_source_timestep_ns(
        df,
        group_cols=group_cols,
        time_col=time_col,
        obs_col=obs_col_or_none,
    )
    timestep_ns = None
    if cfg.timestep is not None:
        timestep_ns = _timestep_to_ns(cfg.timestep, cfg.timestep_unit)
        _validate_requested_timestep(timestep_ns, source_timestep_ns)

    start_lons: list[float] = []
    start_lats: list[float] = []
    end_lons: list[float] = []
    end_lats: list[float] = []

    for _, group in df.groupby(group_cols, sort=False, observed=False):
        prepared = _prepare_trajectory(
            group,
            lon_col=lon_col,
            lat_col=lat_col,
            time_col=time_col,
            obs_col=obs_col_or_none,
        )
        if prepared is None:
            continue

        lon, lat, time_ns = prepared
        segments = _iter_segment_points(
            lon,
            lat,
            time_ns,
            timestep_ns=timestep_ns,
            resample=cfg.resample,
            grid=grid,
        )
        for start_lon, start_lat, end_lon, end_lat in segments:
            start_lons.append(start_lon)
            start_lats.append(start_lat)
            end_lons.append(end_lon)
            end_lats.append(end_lat)

    start_counts = np.zeros((grid.nlat, grid.nlon), dtype=np.int64)
    if not start_lons:
        table = _empty_transition_table()
        ds = _build_dataset(
            grid,
            table,
            start_counts=start_counts,
            timestep_ns=timestep_ns,
            source_timestep_ns=source_timestep_ns,
            resample=cfg.resample,
            entropy_log_base=cfg.plotting.entropy.log_base,
        )
        return GriddedTransitionMatrixResult(transition_table=table, dataset=ds)

    start_lon_bin, start_lat_bin, valid_start = _assign_bins(
        np.asarray(start_lons),
        np.asarray(start_lats),
        grid=grid,
    )
    end_lon_bin, end_lat_bin, valid_end = _assign_bins(
        np.asarray(end_lons),
        np.asarray(end_lats),
        grid=grid,
    )

    valid_transition = valid_start & valid_end
    start_state_all = (
        start_lat_bin[valid_transition] * grid.nlon
        + start_lon_bin[valid_transition]
    )
    start_counts_flat = np.bincount(
        start_state_all,
        minlength=grid.nlat * grid.nlon,
    ).astype(np.int64)
    start_counts = start_counts_flat.reshape((grid.nlat, grid.nlon))

    if not np.any(valid_transition):
        table = _empty_transition_table()
        ds = _build_dataset(
            grid,
            table,
            start_counts=start_counts,
            timestep_ns=timestep_ns,
            source_timestep_ns=source_timestep_ns,
            resample=cfg.resample,
            entropy_log_base=cfg.plotting.entropy.log_base,
        )
        return GriddedTransitionMatrixResult(transition_table=table, dataset=ds)

    start_state = (
        start_lat_bin[valid_transition] * grid.nlon
        + start_lon_bin[valid_transition]
    )
    end_state = (
        end_lat_bin[valid_transition] * grid.nlon
        + end_lon_bin[valid_transition]
    )
    flat_transition = start_state * (grid.nlat * grid.nlon) + end_state
    transition_ids, transition_count = np.unique(flat_transition, return_counts=True)
    decoded_start_state = transition_ids // (grid.nlat * grid.nlon)
    decoded_end_state = transition_ids % (grid.nlat * grid.nlon)

    table = _build_transition_table(
        grid,
        start_lon_bin=(decoded_start_state % grid.nlon).astype(int),
        start_lat_bin=(decoded_start_state // grid.nlon).astype(int),
        end_lon_bin=(decoded_end_state % grid.nlon).astype(int),
        end_lat_bin=(decoded_end_state // grid.nlon).astype(int),
        transition_count=transition_count.astype(np.int64),
        start_counts_flat=start_counts_flat,
    )
    ds = _build_dataset(
        grid,
        table,
        start_counts=start_counts,
        timestep_ns=timestep_ns,
        source_timestep_ns=source_timestep_ns,
        resample=cfg.resample,
        entropy_log_base=cfg.plotting.entropy.log_base,
    )
    return GriddedTransitionMatrixResult(transition_table=table, dataset=ds)
