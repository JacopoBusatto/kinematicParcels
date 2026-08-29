from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..config.models import AliveLatitudeFractionConfig

_DAY_NS = 86_400_000_000_000


def _particle_columns(df: pd.DataFrame) -> list[str]:
    columns = ["trajectory"]
    if "group_member" in df.columns:
        columns.append("group_member")
    return columns


def _iter_particles(df: pd.DataFrame, particle_columns: list[str]):
    grouper = particle_columns[0] if len(particle_columns) == 1 else particle_columns
    return df.groupby(grouper, sort=False, dropna=False)


def _prepare_observations(
    df: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> pd.DataFrame:
    required = ["trajectory", "obs", "time", "lat"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    work = df.copy()
    if cfg.max_group_member is not None and "group_member" in work.columns:
        work = work.loc[work["group_member"] <= cfg.max_group_member].copy()

    work["time"] = pd.to_datetime(work["time"])
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    finite_latitude = np.isfinite(work["lat"].to_numpy(dtype=float))
    work = work.loc[work["time"].notna() & finite_latitude].copy()
    if work.empty:
        return work

    particle_columns = _particle_columns(work)
    work = work.sort_values(particle_columns + ["obs"]).reset_index(drop=True)

    duplicate_columns = particle_columns + ["time"]
    duplicates = work.loc[work.duplicated(duplicate_columns, keep=False)]
    for key, group in duplicates.groupby(
        duplicate_columns, sort=False, dropna=False
    ):
        if group["lat"].nunique(dropna=False) > 1:
            if not isinstance(key, tuple):
                key = (key,)
            key_text = ", ".join(
                f"{column}={value!r}"
                for column, value in zip(duplicate_columns, key)
            )
            raise ValueError(
                "alive_latitude_fraction input has conflicting latitudes for "
                f"the same particle and timestamp ({key_text})."
            )

    work = (
        work.drop_duplicates(duplicate_columns, keep="last")
        .sort_values(particle_columns + ["obs"])
        .reset_index(drop=True)
    )

    for particle_key, group in _iter_particles(work, particle_columns):
        time_ns = pd.DatetimeIndex(group["time"]).asi8
        deltas = np.diff(time_ns)
        if deltas.size > 1 and not (np.all(deltas > 0) or np.all(deltas < 0)):
            raise ValueError(
                "alive_latitude_fraction requires monotonic time ordering within "
                f"each particle; particle {particle_key!r} changes direction."
            )

    work["release_time"] = work.groupby(
        particle_columns, sort=False, dropna=False
    )["time"].transform("first")
    work["age_days"] = (
        (work["time"] - work["release_time"]).dt.total_seconds() / 86400.0
    )
    return work


def _latitude_edges(cfg: AliveLatitudeFractionConfig) -> np.ndarray:
    span = float(cfg.lat_max - cfg.lat_min)
    ratio = span / float(cfg.bin_width_deg)
    n_bins = max(1, int(np.ceil(ratio - 1.0e-12)))
    edges = cfg.lat_min + np.arange(n_bins + 1, dtype=float) * cfg.bin_width_deg
    edges[-1] = cfg.lat_max
    return edges


def _cropped_native_observations(
    work: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> pd.DataFrame:
    if cfg.max_time_days is None or work.empty:
        return work

    if cfg.time_axis == "time":
        first_time = work["time"].min()
        end_time = first_time + pd.Timedelta(days=cfg.max_time_days)
        return work.loc[work["time"] <= end_time].copy()

    tolerance = 1.0e-12
    return work.loc[
        work["age_days"].abs() <= cfg.max_time_days + tolerance
    ].copy()


def _native_frames(
    work: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> tuple[np.ndarray | pd.DatetimeIndex, list[np.ndarray]]:
    work = _cropped_native_observations(work, cfg=cfg)
    if cfg.time_axis == "time":
        axis_values = pd.DatetimeIndex(work["time"].unique()).sort_values()
        grouped = {
            pd.Timestamp(value): group["lat"].to_numpy(dtype=float)
            for value, group in work.groupby("time", sort=False)
        }
        frames = [grouped[pd.Timestamp(value)] for value in axis_values]
        return axis_values, frames

    axis_values = np.array(
        sorted(float(value) for value in pd.unique(work["age_days"])), dtype=float
    )
    grouped = {
        float(value): group["lat"].to_numpy(dtype=float)
        for value, group in work.groupby("age_days", sort=False)
    }
    frames = [grouped[float(value)] for value in axis_values]
    return axis_values, frames


def _time_resample_axis(
    work: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(work["time"].min())
    end = pd.Timestamp(work["time"].max())
    if cfg.max_time_days is not None:
        end = min(end, start + pd.Timedelta(days=cfg.max_time_days))

    step = pd.Timedelta(days=cfg.resample_days)
    step_ns = int(step.value)
    if step_ns <= 0:
        raise ValueError(
            "alive_latitude_fraction.resample_days is too small for nanosecond resolution."
        )
    count = int((end.value - start.value) // step_ns) + 1
    values = start.value + np.arange(count, dtype=np.int64) * step_ns
    return pd.DatetimeIndex(values.astype("datetime64[ns]"))


def _age_resample_axis(
    work: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> np.ndarray:
    minimum = float(work["age_days"].min())
    maximum = float(work["age_days"].max())
    if cfg.max_time_days is not None:
        minimum = max(minimum, -cfg.max_time_days)
        maximum = min(maximum, cfg.max_time_days)

    step = float(cfg.resample_days)
    tolerance = 1.0e-12
    first_multiple = int(np.ceil(minimum / step - tolerance))
    last_multiple = int(np.floor(maximum / step + tolerance))
    if first_multiple > last_multiple:
        return np.array([], dtype=float)
    return np.arange(first_multiple, last_multiple + 1, dtype=float) * step


def _interpolate_frames(
    work: pd.DataFrame,
    axis_values: np.ndarray | pd.DatetimeIndex,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> list[np.ndarray]:
    frame_values: list[list[float]] = [[] for _ in range(len(axis_values))]
    if len(axis_values) == 0:
        return [np.array([], dtype=float) for _ in frame_values]

    particle_columns = _particle_columns(work)
    if cfg.time_axis == "time":
        target_values = pd.DatetimeIndex(axis_values).asi8
    else:
        target_values = np.asarray(axis_values, dtype=float)

    for _, group in _iter_particles(work, particle_columns):
        if cfg.time_axis == "time":
            ordered = group.sort_values("time")
            source_values = pd.DatetimeIndex(ordered["time"]).asi8
        else:
            ordered = group.sort_values("age_days")
            source_values = ordered["age_days"].to_numpy(dtype=float)
        source_latitudes = ordered["lat"].to_numpy(dtype=float)

        if len(source_values) == 1:
            if cfg.time_axis == "time":
                matched = np.flatnonzero(target_values == source_values[0])
            else:
                matched = np.flatnonzero(
                    np.isclose(
                        target_values,
                        source_values[0],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
            for frame_index in matched:
                frame_values[int(frame_index)].append(float(source_latitudes[0]))
            continue

        inside = (target_values >= source_values[0]) & (
            target_values <= source_values[-1]
        )
        target_indices = np.flatnonzero(inside)
        if target_indices.size == 0:
            continue

        if cfg.time_axis == "time":
            origin = int(source_values[0])
            source_numeric = (source_values - origin).astype(float) / _DAY_NS
            target_numeric = (
                target_values[target_indices] - origin
            ).astype(float) / _DAY_NS
        else:
            source_numeric = source_values
            target_numeric = target_values[target_indices]

        interpolated = np.interp(
            target_numeric,
            source_numeric,
            source_latitudes,
        )
        for frame_index, latitude in zip(target_indices, interpolated):
            frame_values[int(frame_index)].append(float(latitude))

    return [np.asarray(values, dtype=float) for values in frame_values]


def _resampled_frames(
    work: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> tuple[np.ndarray | pd.DatetimeIndex, list[np.ndarray]]:
    if cfg.time_axis == "time":
        axis_values = _time_resample_axis(work, cfg=cfg)
    else:
        axis_values = _age_resample_axis(work, cfg=cfg)
    return axis_values, _interpolate_frames(work, axis_values, cfg=cfg)


def _empty_axis(cfg: AliveLatitudeFractionConfig):
    if cfg.time_axis == "time":
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return np.array([], dtype=float)


def compute_alive_latitude_fraction(
    df: pd.DataFrame,
    *,
    cfg: AliveLatitudeFractionConfig,
) -> xr.Dataset:
    """Compute the fraction of alive tracers occupying each latitude bin."""
    work = _prepare_observations(df, cfg=cfg)
    edges = _latitude_edges(cfg)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if work.empty:
        axis_values = _empty_axis(cfg)
        frames: list[np.ndarray] = []
    elif cfg.resample_days is None:
        axis_values, frames = _native_frames(work, cfg=cfg)
    else:
        axis_values, frames = _resampled_frames(work, cfg=cfg)

    counts = np.zeros((len(axis_values), len(centers)), dtype=np.int64)
    alive_counts = np.zeros(len(axis_values), dtype=np.int64)
    for frame_index, latitudes in enumerate(frames):
        alive_counts[frame_index] = len(latitudes)
        counts[frame_index] = np.histogram(latitudes, bins=edges)[0]

    supported = alive_counts >= cfg.minimum_alive_tracers
    fractions = np.full(counts.shape, np.nan, dtype=float)
    if np.any(supported):
        fractions[supported] = (
            counts[supported].astype(float) / alive_counts[supported, np.newaxis]
        )

    axis_name = "time" if cfg.time_axis == "time" else "age_days"
    dataset = xr.Dataset(
        data_vars={
            "latitude_bin_count": (
                (axis_name, "latitude_bin"),
                counts,
            ),
            "alive_tracer_count": ((axis_name,), alive_counts),
            "alive_tracer_fraction": (
                (axis_name, "latitude_bin"),
                fractions,
            ),
            "meets_minimum_alive": ((axis_name,), supported),
        },
        coords={
            axis_name: axis_values,
            "latitude_bin": np.arange(len(centers), dtype=np.int64),
            "lat_lower": (("latitude_bin",), edges[:-1]),
            "lat_center": (("latitude_bin",), centers),
            "lat_upper": (("latitude_bin",), edges[1:]),
        },
        attrs={
            "analysis": "alive_latitude_fraction",
            "time_axis": cfg.time_axis,
            "native_or_resampled": (
                "native" if cfg.resample_days is None else "resampled"
            ),
            "resample_days": (
                "native" if cfg.resample_days is None else float(cfg.resample_days)
            ),
            "max_time_days": (
                "full" if cfg.max_time_days is None else float(cfg.max_time_days)
            ),
            "minimum_alive_tracers": int(cfg.minimum_alive_tracers),
            "max_group_member": (
                "all"
                if cfg.max_group_member is None
                else int(cfg.max_group_member)
            ),
            "normalization": (
                "latitude_bin_count divided by all selected tracers alive at the axis coordinate"
            ),
            "latitude_bin_boundary_rule": (
                "lower-inclusive upper-exclusive; final bin includes lat_max"
            ),
        },
    )
    dataset["latitude_bin_count"].attrs.update(
        {"long_name": "tracers in latitude bin", "units": "1"}
    )
    dataset["alive_tracer_count"].attrs.update(
        {"long_name": "all selected tracers alive", "units": "1"}
    )
    dataset["alive_tracer_fraction"].attrs.update(
        {
            "long_name": "fraction of alive tracers in latitude bin",
            "units": "1",
        }
    )
    dataset["lat_lower"].attrs["units"] = "degrees_north"
    dataset["lat_center"].attrs["units"] = "degrees_north"
    dataset["lat_upper"].attrs["units"] = "degrees_north"
    if axis_name == "age_days":
        dataset[axis_name].attrs.update(
            {"long_name": "signed age since release", "units": "days"}
        )
    return dataset
