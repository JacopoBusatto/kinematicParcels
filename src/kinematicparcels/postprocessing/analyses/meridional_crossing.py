from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from kinematicparcels import __version__

from ..config.models import MeridionalCrossingConfig
from ..core.gridding import RegularGrid


@dataclass(frozen=True)
class MeridionalCrossingResult:
    grid_table: pd.DataFrame
    dataset: xr.Dataset


def _empty_grid_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lon_bin",
            "lat_bin",
            "lon_center",
            "lat_center",
            "crossing_count_northward",
            "crossing_probability_northward",
            "crossing_count_southward",
            "crossing_probability_southward",
        ]
    )


def _resolve_threshold(value: float | str, *, auto_value: float) -> float:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"Unsupported automatic threshold specifier: {value!r}")
        return float(auto_value)
    return float(value)


def _longitude_convention_for_grid(grid: RegularGrid) -> str:
    if grid.lon_min >= 0.0 and grid.lon_max > 180.0:
        return "0_360"
    if grid.lon_min < 0.0 and grid.lon_max <= 180.0:
        return "-180_180"
    return "wrapped_to_grid_bounds"


def _normalize_longitude_to_grid(lon: float, grid: RegularGrid) -> float:
    width = float(grid.lon_max - grid.lon_min)
    if not np.isfinite(lon):
        return np.nan
    if width <= 0.0 or width > 360.0 + 1.0e-9:
        return lon

    wrapped = ((lon - grid.lon_min) % 360.0) + grid.lon_min
    if wrapped >= grid.lon_max:
        wrapped = np.nextafter(grid.lon_max, grid.lon_min)
    return wrapped


def _apply_latitude_filter(lat: np.ndarray, *, method: str, window: int) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    if method == "none" or window <= 1:
        return lat.copy()

    rolling = pd.Series(lat).rolling(window=window, center=True, min_periods=1)
    if method == "rolling_mean":
        return rolling.mean().to_numpy(dtype=float)
    if method == "rolling_median":
        return rolling.median().to_numpy(dtype=float)

    raise ValueError(f"Unsupported latitude filter: {method!r}")


def _classify_directional_steps(filtered_lat: np.ndarray, *, threshold_deg: float) -> np.ndarray:
    dlat = np.diff(filtered_lat)
    direction = np.zeros(dlat.shape, dtype=np.int8)
    direction[dlat > threshold_deg] = 1
    direction[dlat < -threshold_deg] = -1
    return direction


def _extract_directional_segments(step_direction: np.ndarray) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    start_idx: int | None = None
    current_sign = 0

    for step_idx, sign in enumerate(step_direction):
        if sign == 0:
            if current_sign != 0 and start_idx is not None:
                direction = "northward" if current_sign > 0 else "southward"
                segments.append((direction, start_idx, step_idx - 1))
            start_idx = None
            current_sign = 0
            continue

        if current_sign == 0:
            start_idx = step_idx
            current_sign = int(sign)
            continue

        if sign != current_sign:
            if start_idx is not None:
                direction = "northward" if current_sign > 0 else "southward"
                segments.append((direction, start_idx, step_idx - 1))
            start_idx = step_idx
            current_sign = int(sign)

    if current_sign != 0 and start_idx is not None:
        direction = "northward" if current_sign > 0 else "southward"
        segments.append((direction, start_idx, len(step_direction) - 1))

    return segments


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
    time = work[time_col].to_numpy(dtype="datetime64[ns]")
    return lon, lat, time


def _segment_is_valid(
    segment_lat: np.ndarray,
    segment_time: np.ndarray,
    *,
    min_duration_days: float,
    min_displacement_deg: float,
) -> bool:
    duration_days = float((segment_time[-1] - segment_time[0]) / np.timedelta64(1, "D"))
    net_lat_displacement = float(np.abs(segment_lat[-1] - segment_lat[0]))
    return (
        duration_days >= min_duration_days
        or net_lat_displacement >= min_displacement_deg
    )


def _build_crossing_targets(
    segment_lat: np.ndarray,
    *,
    direction: str,
    grid: RegularGrid,
    crossing_reference: str,
) -> list[tuple[float, int]]:
    start_lat = float(segment_lat[0])
    min_lat = float(np.nanmin(segment_lat))
    max_lat = float(np.nanmax(segment_lat))
    tol = 1.0e-12

    targets: list[tuple[float, int]] = []

    if crossing_reference == "center":
        if direction == "northward":
            for lat_bin, center in enumerate(grid.lat_centers):
                if center > start_lat + tol and center <= max_lat + tol:
                    targets.append((float(center), lat_bin))
        else:
            for lat_bin in range(grid.nlat - 1, -1, -1):
                center = float(grid.lat_centers[lat_bin])
                if center < start_lat - tol and center >= min_lat - tol:
                    targets.append((center, lat_bin))
        return targets

    if crossing_reference != "edge":
        raise ValueError(f"Unsupported crossing latitude reference: {crossing_reference!r}")

    if direction == "northward":
        for edge_idx in range(grid.nlat):
            edge = float(grid.lat_edges[edge_idx])
            if edge > start_lat + tol and edge <= max_lat + tol:
                targets.append((edge, edge_idx))
    else:
        for edge_idx in range(grid.nlat, 0, -1):
            edge = float(grid.lat_edges[edge_idx])
            if edge < start_lat - tol and edge >= min_lat - tol:
                targets.append((edge, edge_idx - 1))

    return targets


def _interpolate_crossing_longitude(
    lon_unwrapped: np.ndarray,
    lon_original: np.ndarray,
    lat: np.ndarray,
    *,
    pair_idx: int,
    target_lat: float,
    grid: RegularGrid,
) -> float:
    lat0 = float(lat[pair_idx])
    lat1 = float(lat[pair_idx + 1])
    lon0 = float(lon_unwrapped[pair_idx])
    lon1 = float(lon_unwrapped[pair_idx + 1])

    if not np.isfinite(lat0) or not np.isfinite(lat1):
        nearest = pair_idx
        if np.abs(lat1 - target_lat) < np.abs(lat0 - target_lat):
            nearest = pair_idx + 1
        return _normalize_longitude_to_grid(float(lon_original[nearest]), grid)

    if np.isclose(lat1, lat0, atol=1.0e-12):
        nearest = pair_idx
        if np.abs(lat1 - target_lat) < np.abs(lat0 - target_lat):
            nearest = pair_idx + 1
        return _normalize_longitude_to_grid(float(lon_original[nearest]), grid)

    frac = (target_lat - lat0) / (lat1 - lat0)
    frac = float(np.clip(frac, 0.0, 1.0))
    lon_cross = lon0 + frac * (lon1 - lon0)
    return _normalize_longitude_to_grid(lon_cross, grid)


def _iter_segment_crossings(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    direction: str,
    grid: RegularGrid,
    crossing_reference: str,
) -> list[tuple[int, float]]:
    targets = _build_crossing_targets(
        lat,
        direction=direction,
        grid=grid,
        crossing_reference=crossing_reference,
    )
    if not targets:
        return []

    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon.astype(float))))
    crossings: list[tuple[int, float]] = []
    search_start = 0

    for target_lat, lat_bin in targets:
        for pair_idx in range(search_start, len(lat) - 1):
            lat0 = float(lat[pair_idx])
            lat1 = float(lat[pair_idx + 1])
            if np.isclose(lat1, lat0, atol=1.0e-12):
                continue
            low = min(lat0, lat1)
            high = max(lat0, lat1)
            if low - 1.0e-12 <= target_lat <= high + 1.0e-12:
                lon_cross = _interpolate_crossing_longitude(
                    lon_unwrapped,
                    lon,
                    lat,
                    pair_idx=pair_idx,
                    target_lat=target_lat,
                    grid=grid,
                )
                crossings.append((lat_bin, lon_cross))
                search_start = pair_idx
                break

    return crossings


def _accumulate_crossings(
    crossings: list[tuple[int, float]],
    *,
    grid: RegularGrid,
    counts: np.ndarray,
) -> None:
    for lat_bin, lon_cross in crossings:
        if not np.isfinite(lon_cross):
            continue
        if lat_bin < 0 or lat_bin >= grid.nlat:
            continue

        lon_bin = int(np.floor((lon_cross - grid.lon_min) / grid.dlon))
        if lon_bin < 0 or lon_bin >= grid.nlon:
            continue

        counts[lat_bin, lon_bin] += 1.0


def _masked_probability(counts: np.ndarray, n_segments: int) -> np.ndarray:
    out = np.full(counts.shape, np.nan, dtype=float)
    if n_segments <= 0:
        return out
    nonzero = counts > 0.0
    out[nonzero] = counts[nonzero] / float(n_segments)
    return out


def _masked_counts(counts: np.ndarray) -> np.ndarray:
    out = np.full(counts.shape, np.nan, dtype=float)
    nonzero = counts > 0.0
    out[nonzero] = counts[nonzero]
    return out


def _build_grid_table(
    grid: RegularGrid,
    north_counts: np.ndarray,
    north_probability: np.ndarray,
    south_counts: np.ndarray,
    south_probability: np.ndarray,
) -> pd.DataFrame:
    occupied = (
        np.isfinite(north_counts)
        | np.isfinite(north_probability)
        | np.isfinite(south_counts)
        | np.isfinite(south_probability)
    )
    lat_bins, lon_bins = np.where(occupied)
    if len(lat_bins) == 0:
        return _empty_grid_table()

    return pd.DataFrame(
        {
            "lon_bin": lon_bins.astype(int),
            "lat_bin": lat_bins.astype(int),
            "lon_center": grid.lon_min + (lon_bins + 0.5) * grid.dlon,
            "lat_center": grid.lat_min + (lat_bins + 0.5) * grid.dlat,
            "crossing_count_northward": north_counts[lat_bins, lon_bins],
            "crossing_probability_northward": north_probability[lat_bins, lon_bins],
            "crossing_count_southward": south_counts[lat_bins, lon_bins],
            "crossing_probability_southward": south_probability[lat_bins, lon_bins],
        }
    )


def compute_meridional_crossing(
    df: pd.DataFrame,
    *,
    grid: RegularGrid,
    cfg: MeridionalCrossingConfig,
    trajectory_col: str = "trajectory",
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    obs_col: str = "obs",
) -> MeridionalCrossingResult:
    """
    Estimate directional meridional crossing frequency on a regular lon/lat grid.

    The result is an empirical, release-dependent crossing diagnostic derived from
    directional trajectory segments. It is not an intrinsic permeability field.
    """
    required = [trajectory_col, lon_col, lat_col, time_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    if not cfg.crossing.count_once_per_segment_per_lat_bin:
        raise ValueError(
            "meridional_crossing.crossing.count_once_per_segment_per_lat_bin=false "
            "is not supported in the first implementation."
        )

    direction_threshold_deg = _resolve_threshold(
        cfg.segmentation.direction_threshold_deg,
        auto_value=0.25 * grid.dlat,
    )
    min_segment_displacement_deg = _resolve_threshold(
        cfg.segmentation.min_segment_displacement_deg,
        auto_value=grid.dlat,
    )

    north_counts_raw = np.zeros((grid.nlat, grid.nlon), dtype=float)
    south_counts_raw = np.zeros((grid.nlat, grid.nlon), dtype=float)
    n_segments_northward = 0
    n_segments_southward = 0

    grouped = df.groupby(trajectory_col, sort=False, observed=False)
    for _, group in grouped:
        prepared = _prepare_trajectory(
            group,
            lon_col=lon_col,
            lat_col=lat_col,
            time_col=time_col,
            obs_col=obs_col if obs_col in df.columns else None,
        )
        if prepared is None:
            continue

        lon, lat, time = prepared
        filtered_lat = _apply_latitude_filter(
            lat,
            method=cfg.segmentation.lat_filter,
            window=cfg.segmentation.filter_window,
        )
        segments = _extract_directional_segments(
            _classify_directional_steps(
                filtered_lat,
                threshold_deg=direction_threshold_deg,
            )
        )

        for direction, step_start, step_end in segments:
            if cfg.direction != "both" and direction != cfg.direction:
                continue

            point_start = step_start
            point_end = step_end + 2
            segment_lon = lon[point_start:point_end]
            segment_lat = lat[point_start:point_end]
            segment_time = time[point_start:point_end]

            if len(segment_lat) < 2:
                continue

            if not _segment_is_valid(
                segment_lat,
                segment_time,
                min_duration_days=cfg.segmentation.min_segment_duration_days,
                min_displacement_deg=min_segment_displacement_deg,
            ):
                continue

            if direction == "northward":
                n_segments_northward += 1
            else:
                n_segments_southward += 1

            crossings = _iter_segment_crossings(
                segment_lon,
                segment_lat,
                direction=direction,
                grid=grid,
                crossing_reference=cfg.crossing.crossing_latitude_reference,
            )

            if direction == "northward":
                _accumulate_crossings(crossings, grid=grid, counts=north_counts_raw)
            else:
                _accumulate_crossings(crossings, grid=grid, counts=south_counts_raw)

    north_counts = _masked_counts(north_counts_raw)
    south_counts = _masked_counts(south_counts_raw)
    north_probability = _masked_probability(north_counts_raw, n_segments_northward)
    south_probability = _masked_probability(south_counts_raw, n_segments_southward)

    ds = xr.Dataset(
        data_vars={
            "crossing_count_northward": (("lat", "lon"), north_counts),
            "crossing_probability_northward": (("lat", "lon"), north_probability),
            "crossing_count_southward": (("lat", "lon"), south_counts),
            "crossing_probability_southward": (("lat", "lon"), south_probability),
            "n_segments_northward": ((), np.int64(n_segments_northward)),
            "n_segments_southward": ((), np.int64(n_segments_southward)),
        },
        coords={
            "lat": grid.lat_centers,
            "lon": grid.lon_centers,
        },
        attrs={
            "title": "Directional meridional crossing probability",
            "summary": (
                "Empirical, release-dependent crossing-frequency diagnostic derived from "
                "directional meridional trajectory segments. Not an intrinsic permeability field."
            ),
            "grid_type": "regular_lonlat",
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "dlon": grid.dlon,
            "dlat": grid.dlat,
            "direction": cfg.direction,
            "lat_filter": cfg.segmentation.lat_filter,
            "filter_window": cfg.segmentation.filter_window,
            "direction_threshold_deg": direction_threshold_deg,
            "min_segment_duration_days": cfg.segmentation.min_segment_duration_days,
            "min_segment_displacement_deg": min_segment_displacement_deg,
            "valid_if": cfg.segmentation.valid_if,
            "crossing_latitude_reference": cfg.crossing.crossing_latitude_reference,
            "count_once_per_segment_per_lat_bin": cfg.crossing.count_once_per_segment_per_lat_bin,
            "normalization": "crossing_count_direction / n_valid_directional_segments_direction",
            "longitude_convention": _longitude_convention_for_grid(grid),
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "software_version": __version__,
        },
    )

    ds["crossing_count_northward"].attrs.update(
        long_name="northward crossing count",
        units="count",
    )
    ds["crossing_probability_northward"].attrs.update(
        long_name="northward crossing probability",
        units="1",
    )
    ds["crossing_count_southward"].attrs.update(
        long_name="southward crossing count",
        units="count",
    )
    ds["crossing_probability_southward"].attrs.update(
        long_name="southward crossing probability",
        units="1",
    )
    ds["n_segments_northward"].attrs.update(units="count")
    ds["n_segments_southward"].attrs.update(units="count")

    grid_table = _build_grid_table(
        grid,
        north_counts,
        north_probability,
        south_counts,
        south_probability,
    )
    return MeridionalCrossingResult(grid_table=grid_table, dataset=ds)