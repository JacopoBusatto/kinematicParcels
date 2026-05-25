from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import glob

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from kinematicparcels.regions import ALL_REGIONS, Region, RegionManager


DEFAULT_COLUMNS = {
    "platform_code": "PLATFORM_CODE",
    "time": "DATE (YYYY-MM-DDTHH:MI:SSZ)",
    "lat": "LATITUDE (degree_north)",
    "lon": "LONGITUDE (degree_east)",
    "pressure": "PRES_ADJUSTED (decibar)",
}

DEFAULT_DATASET_ATTRS = {
    "Conventions": "CF-1.6/CF-1.7",
    "feature_type": "trajectory",
    "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
    "source": "ARGO CSV conversion",
}


@dataclass(frozen=True)
class SegmentConfig:
    mode: str = "ignore"
    max_gap_days: float = 10.0
    min_duration_days: float = 0.0
    max_speed_km_per_day: float | None = None


@dataclass(frozen=True)
class ResampleConfig:
    frequency: str | None = None
    interpolate: str = "time"


@dataclass(frozen=True)
class RegionFilterConfig:
    names_or_labels: tuple[str, ...] = ()
    cut_from_first_entry: bool = False
    input_lon_mode: str = "-180_180"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ARGO CSV files into a Parcels-compatible trajectory Zarr dataset."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the ARGO conversion YAML configuration file.",
    )
    return parser


def _resolve_columns(config: dict[str, Any]) -> dict[str, str]:
    columns = dict(DEFAULT_COLUMNS)
    columns.update(config.get("columns", {}))
    return columns


def _resolve_input_files(config: dict[str, Any]) -> list[Path]:
    input_cfg = config.get("input", {})
    files: list[Path] = []

    for item in input_cfg.get("csv_files", []) or []:
        files.append(Path(item))

    csv_glob = input_cfg.get("csv_glob")
    if csv_glob:
        files.extend(Path(path) for path in glob.glob(csv_glob))

    input_dir = input_cfg.get("csv_dir")
    pattern = input_cfg.get("pattern", "*.csv")
    if input_dir:
        files.extend(sorted(Path(input_dir).glob(pattern)))

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError("No input ARGO CSV files were found from the provided configuration.")

    missing = [path for path in unique_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"The following ARGO CSV files were not found: {missing_str}")

    return unique_files


def _resolve_optional_variables(config: dict[str, Any]) -> list[str]:
    variables_cfg = config.get("variables", {})
    optional = variables_cfg.get("optional", []) or []
    return list(optional)


def _resolve_output_path(config: dict[str, Any]) -> Path:
    output_cfg = config.get("output", {})
    output_path = output_cfg.get("path")
    if not output_path:
        raise ValueError("The conversion config must define output.path.")
    return Path(output_path)


def _resolve_parking_depth_value(config: dict[str, Any]) -> float:
    parking_cfg = config.get("processing", {}).get("parking_depth", {})
    mode = parking_cfg.get("mode", "fixed")
    if mode != "fixed":
        raise ValueError(
            "Only processing.parking_depth.mode='fixed' is supported in the first ARGO converter version."
        )

    value = parking_cfg.get("value", 1000.0)
    return float(value)


def _resolve_segment_config(config: dict[str, Any]) -> SegmentConfig:
    raw = config.get("processing", {}).get("segment", {})
    mode = raw.get("mode", "ignore")
    aliases = {
        "split": "split_as_new",
        "separate": "split_as_new",
        "irregular": "ignore",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"ignore", "longest", "split_as_new"}:
        raise ValueError("processing.segment.mode must be one of: ignore, longest, split_as_new")

    max_speed_raw = raw.get("max_speed_km_per_day", None)
    max_speed_km_per_day: float | None
    if max_speed_raw is None:
        max_speed_km_per_day = None
    else:
        max_speed_km_per_day = float(max_speed_raw)
        if max_speed_km_per_day <= 0.0:
            raise ValueError("processing.segment.max_speed_km_per_day must be > 0 when provided")

    return SegmentConfig(
        mode=mode,
        max_gap_days=float(raw.get("max_gap_days", 10.0)),
        min_duration_days=float(raw.get("min_duration_days", 0.0)),
        max_speed_km_per_day=max_speed_km_per_day,
    )


def _resolve_resample_config(config: dict[str, Any]) -> ResampleConfig:
    raw = config.get("processing", {}).get("resample", {})
    enabled = bool(raw.get("enabled", raw.get("frequency") is not None))
    frequency = raw.get("frequency") if enabled else None
    if isinstance(frequency, str):
        frequency = frequency.lower()
    return ResampleConfig(
        frequency=frequency,
        interpolate=str(raw.get("interpolate", "time")),
    )


def _resolve_region_filter_config(config: dict[str, Any]) -> RegionFilterConfig:
    raw = config.get("processing", {}).get("regions", {})
    names_or_labels = tuple(raw.get("names_or_labels", []) or [])
    return RegionFilterConfig(
        names_or_labels=names_or_labels,
        cut_from_first_entry=bool(raw.get("cut_from_first_entry", False)),
        input_lon_mode=str(raw.get("input_lon_mode", "-180_180")),
    )


def _build_selected_region_manager(region_names_or_labels: tuple[str, ...]) -> RegionManager | None:
    if not region_names_or_labels:
        return None

    requested = set(region_names_or_labels)
    selected_regions = [
        region for region in ALL_REGIONS
        if region.label in requested or region.name in requested
    ]

    found_tokens = {
        token
        for region in selected_regions
        for token in (region.label, region.name)
        if isinstance(token, str)
    }
    missing = sorted(requested - found_tokens)
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown region labels/names requested in processing.regions: {missing_str}")

    return RegionManager(selected_regions)


def _coerce_time_series(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series, utc=True, errors="raise")
    return times.dt.tz_convert(None)


def _choose_surface_rows(df: pd.DataFrame, *, group_cols: list[str], pressure_col: str | None) -> pd.DataFrame:
    if pressure_col is None or pressure_col not in df.columns:
        return df.drop_duplicates(subset=group_cols, keep="first").copy()

    work = df.copy()
    work[pressure_col] = pd.to_numeric(work[pressure_col], errors="coerce")
    work["_pressure_order"] = work[pressure_col].fillna(np.inf)
    work = work.sort_values(group_cols + ["_pressure_order"], kind="stable")
    work = work.drop_duplicates(subset=group_cols, keep="first")
    return work.drop(columns="_pressure_order")


def _read_surface_points(
    csv_path: Path,
    *,
    columns: dict[str, str],
    optional_variables: list[str],
    parking_depth_value: float,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        columns["platform_code"],
        columns["time"],
        columns["lat"],
        columns["lon"],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required ARGO columns in {csv_path}: {missing_str}")

    missing_optional = [column for column in optional_variables if column not in df.columns]
    if missing_optional:
        missing_str = ", ".join(missing_optional)
        raise KeyError(f"Missing requested optional ARGO columns in {csv_path}: {missing_str}")

    pressure_col = columns.get("pressure")
    if pressure_col not in df.columns:
        pressure_col = "PRES (decibar)" if "PRES (decibar)" in df.columns else None

    time_col = columns["time"]
    lat_col = columns["lat"]
    lon_col = columns["lon"]
    platform_col = columns["platform_code"]

    df = df.copy()
    df[time_col] = _coerce_time_series(df[time_col])
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    if pressure_col is not None:
        df[pressure_col] = pd.to_numeric(df[pressure_col], errors="coerce")

    group_cols = [platform_col, time_col, lat_col, lon_col]
    surface = _choose_surface_rows(df, group_cols=group_cols, pressure_col=pressure_col)
    surface = surface.sort_values([platform_col, time_col, lat_col, lon_col], kind="stable")

    rename_map = {
        platform_col: "platform_code",
        time_col: "time",
        lat_col: "lat",
        lon_col: "lon",
    }

    surface = surface.rename(columns=rename_map)
    surface["platform_code"] = pd.to_numeric(surface["platform_code"], errors="coerce")
    if surface["platform_code"].isna().any():
        raise ValueError(f"Non-numeric platform_code values found in {csv_path}")

    surface["z"] = float(parking_depth_value)

    keep_columns = [
        "platform_code",
        "time",
        "lat",
        "lon",
        "z",
    ]
    keep_columns.extend(optional_variables)

    return surface[keep_columns].reset_index(drop=True)


def _merge_platform_surface_points(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.sort_values(["platform_code", "time", "lat", "lon"], kind="stable").reset_index(drop=True)

    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.sort_values(["platform_code", "time", "lat", "lon"], kind="stable")
    merged = merged.drop_duplicates(subset=["platform_code", "time", "lat", "lon"], keep="last")
    return merged.reset_index(drop=True)


def _segment_labels(times: pd.Series, *, max_gap_days: float) -> pd.Series:
    gaps = times.diff().dt.total_seconds().div(86400.0)
    return gaps.gt(max_gap_days).fillna(False).cumsum().astype(int)


def _haversine_km(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _build_segment_break_mask(df: pd.DataFrame, *, config: SegmentConfig) -> pd.Series:
    ordered = df.sort_values("time", kind="stable").reset_index(drop=True)
    gap_breaks = _segment_labels(ordered["time"], max_gap_days=config.max_gap_days).diff().fillna(0).gt(0)

    if config.max_speed_km_per_day is None or len(ordered) <= 1:
        return gap_breaks.astype(bool)

    dt_days = ordered["time"].diff().dt.total_seconds().div(86400.0)
    lon_prev = ordered["lon"].shift(1).to_numpy(dtype=float)
    lat_prev = ordered["lat"].shift(1).to_numpy(dtype=float)
    lon_curr = ordered["lon"].to_numpy(dtype=float)
    lat_curr = ordered["lat"].to_numpy(dtype=float)

    distances_km = np.full(len(ordered), np.nan, dtype=float)
    valid_pairs = np.isfinite(lon_prev) & np.isfinite(lat_prev) & np.isfinite(lon_curr) & np.isfinite(lat_curr)
    distances_km[valid_pairs] = _haversine_km(
        lon_prev[valid_pairs],
        lat_prev[valid_pairs],
        lon_curr[valid_pairs],
        lat_curr[valid_pairs],
    )

    dt_values = dt_days.to_numpy(dtype=float)
    speed_breaks = np.zeros(len(ordered), dtype=bool)
    positive_dt = valid_pairs & np.isfinite(dt_values) & (dt_values > 0.0)
    speed_breaks[positive_dt] = (distances_km[positive_dt] / dt_values[positive_dt]) > config.max_speed_km_per_day

    zero_or_negative_dt = valid_pairs & np.isfinite(dt_values) & (dt_values <= 0.0)
    speed_breaks[zero_or_negative_dt] = distances_km[zero_or_negative_dt] > 0.0

    return (gap_breaks.to_numpy(dtype=bool) | speed_breaks)


def _segment_duration_days(df: pd.DataFrame) -> float:
    if len(df) <= 1:
        return 0.0
    delta = df["time"].iloc[-1] - df["time"].iloc[0]
    return float(delta.total_seconds() / 86400.0)


def _apply_segment_policy(df: pd.DataFrame, *, config: SegmentConfig) -> list[pd.DataFrame]:
    if df.empty:
        return []

    ordered = df.sort_values("time", kind="stable").reset_index(drop=True)
    if config.mode == "ignore":
        return [ordered]

    break_mask = _build_segment_break_mask(ordered, config=config)
    labels = pd.Series(break_mask, index=ordered.index).cumsum().astype(int)
    segments = [segment.reset_index(drop=True) for _, segment in ordered.groupby(labels, sort=True)]

    if config.mode == "longest":
        selected = max(
            segments,
            key=lambda segment: (len(segment), _segment_duration_days(segment)),
        ).copy()
        return [selected]

    kept_segments: list[pd.DataFrame] = []
    for segment in segments:
        if _segment_duration_days(segment) < config.min_duration_days:
            continue
        kept_segments.append(segment.copy())

    return kept_segments


def _point_in_regions(
    lon: float,
    lat: float,
    *,
    region_manager: RegionManager,
    input_lon_mode: str,
) -> bool:
    return bool(
        region_manager.find_regions(
            lon,
            lat,
            howMany="first",
            input_lon_mode=input_lon_mode,
        )
    )


def _apply_region_filter(
    trajectories: list[pd.DataFrame],
    *,
    config: RegionFilterConfig,
) -> list[pd.DataFrame]:
    region_manager = _build_selected_region_manager(config.names_or_labels)
    if region_manager is None:
        return trajectories

    filtered: list[pd.DataFrame] = []
    trajectory_iterator = (
        tqdm(trajectories, desc="Applying region filter", unit="traj")
        if len(trajectories) > 1
        else trajectories
    )
    for trajectory in trajectory_iterator:
        mask = trajectory.apply(
            lambda row: _point_in_regions(
                float(row["lon"]),
                float(row["lat"]),
                region_manager=region_manager,
                input_lon_mode=config.input_lon_mode,
            ),
            axis=1,
        )

        if not mask.any():
            continue

        selected = trajectory
        if config.cut_from_first_entry:
            first_hit_position = int(np.flatnonzero(mask.to_numpy())[0])
            selected = trajectory.iloc[first_hit_position:].reset_index(drop=True)

        filtered.append(selected.reset_index(drop=True))

    return filtered


def _collapse_duplicate_times(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("time", kind="stable").reset_index(drop=True)
    if not ordered["time"].duplicated().any():
        return ordered

    return ordered.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)


def _unwrap_longitudes(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=float)
    if values.size <= 1:
        return values.copy()

    unwrapped = values.copy()
    valid_idx = np.flatnonzero(np.isfinite(values))
    if valid_idx.size <= 1:
        return unwrapped

    split_points = np.flatnonzero(np.diff(valid_idx) > 1) + 1
    for chunk in np.split(valid_idx, split_points):
        if chunk.size <= 1:
            continue
        unwrapped[chunk] = np.rad2deg(np.unwrap(np.deg2rad(values[chunk])))

    return unwrapped


def _wrap_longitudes(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def _resample_single_trajectory(df: pd.DataFrame, *, config: ResampleConfig) -> pd.DataFrame:
    if not config.frequency or len(df) <= 1:
        return df.reset_index(drop=True)

    ordered = _collapse_duplicate_times(df).set_index("time").copy()
    if "lon" in ordered.columns:
        ordered.loc[:, "lon"] = _unwrap_longitudes(ordered["lon"].to_numpy(dtype=float))

    new_index = pd.date_range(ordered.index.min(), ordered.index.max(), freq=config.frequency)
    if len(new_index) == 0 or new_index[-1] != ordered.index.max():
        new_index = new_index.append(pd.DatetimeIndex([ordered.index.max()]))

    expanded = ordered.reindex(ordered.index.union(new_index)).sort_index()

    numeric_cols = expanded.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if numeric_cols:
        expanded.loc[:, numeric_cols] = expanded.loc[:, numeric_cols].interpolate(
            method=config.interpolate,
            limit_direction="both",
        )

    other_cols = [column for column in expanded.columns if column not in numeric_cols]
    if other_cols:
        expanded.loc[:, other_cols] = expanded.loc[:, other_cols].ffill().bfill()

    result = expanded.loc[new_index].reset_index().rename(columns={"index": "time"})
    if "lon" in result.columns:
        result.loc[:, "lon"] = _wrap_longitudes(result["lon"].to_numpy(dtype=float))
    return result.reset_index(drop=True)


def _apply_resampling(
    trajectories: list[pd.DataFrame],
    *,
    config: ResampleConfig,
) -> list[pd.DataFrame]:
    if not config.frequency or len(trajectories) <= 1:
        return [_resample_single_trajectory(trajectory, config=config) for trajectory in trajectories]

    return [
        _resample_single_trajectory(trajectory, config=config)
        for trajectory in tqdm(trajectories, desc="Resampling trajectories", unit="traj")
    ]


def _series_to_fixed_width_array(series: pd.Series, length: int) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        out = np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]")
        values = series.to_numpy(dtype="datetime64[ns]")
        out[: len(values)] = values
        return out

    if pd.api.types.is_numeric_dtype(series.dtype):
        out = np.full(length, np.nan, dtype=np.float64)
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        out[: len(values)] = values
        return out

    values = series.astype(str).to_numpy()
    width = max([1, *(len(value) for value in values)])
    out = np.full(length, "", dtype=f"<U{width}")
    out[: len(values)] = values
    return out


def build_dataset_from_trajectories(trajectories: list[pd.DataFrame]) -> xr.Dataset:
    if not trajectories:
        raise ValueError("The ARGO conversion produced no trajectories after filtering.")

    ordered = sorted(trajectories, key=lambda frame: int(frame["trajectory"].iloc[0]))
    trajectory_ids = np.asarray([int(frame["trajectory"].iloc[0]) for frame in ordered], dtype=np.int64)
    max_obs = max(len(frame) for frame in ordered)
    obs = np.arange(max_obs, dtype=np.int32)

    variable_names = []
    for frame in ordered:
        for column in frame.columns:
            if column not in {"trajectory", "obs"} and column not in variable_names:
                variable_names.append(column)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for name in variable_names:
        sample_series = ordered[0][name]
        rows = [_series_to_fixed_width_array(frame[name], max_obs) for frame in ordered]
        values = np.stack(rows, axis=0)

        if pd.api.types.is_datetime64_any_dtype(sample_series.dtype):
            values = values.astype("datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(sample_series.dtype):
            values = values.astype(np.float64)

        data_vars[name] = (("trajectory", "obs"), values)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "trajectory": trajectory_ids,
            "obs": obs,
        },
        attrs=dict(DEFAULT_DATASET_ATTRS),
    )

    return ds


def convert_argo_to_dataframe(config: dict[str, Any]) -> list[pd.DataFrame]:
    columns = _resolve_columns(config)
    files = _resolve_input_files(config)
    optional_variables = _resolve_optional_variables(config)
    parking_depth_value = _resolve_parking_depth_value(config)
    segment_config = _resolve_segment_config(config)
    region_config = _resolve_region_filter_config(config)
    resample_config = _resolve_resample_config(config)

    print(f"Resolved {len(files)} ARGO CSV file(s)")
    file_iterator = tqdm(files, desc="Reading ARGO CSV files", unit="file") if len(files) > 1 else files
    platform_buffers: dict[float, pd.DataFrame] = {}
    for csv_path in file_iterator:
        surface_points = _read_surface_points(
            csv_path,
            columns=columns,
            optional_variables=optional_variables,
            parking_depth_value=parking_depth_value,
        )

        for platform_code, platform_df in surface_points.groupby("platform_code", sort=False):
            code = float(platform_code)
            platform_buffers[code] = _merge_platform_surface_points(
                platform_buffers.get(code),
                platform_df.reset_index(drop=True),
            )

    n_platforms = len(platform_buffers)
    print(f"Buffered surface fixes for {n_platforms} platform(s)")
    print(f"Applying segmentation policy: {segment_config.mode}")
    trajectories: list[pd.DataFrame] = []
    platform_iterator = (
        tqdm(platform_buffers.items(), total=len(platform_buffers), desc="Segmenting platforms", unit="platform")
        if len(platform_buffers) > 1
        else platform_buffers.items()
    )
    for _, platform_df in platform_iterator:
        trajectories.extend(_apply_segment_policy(platform_df.reset_index(drop=True), config=segment_config))

    print(f"Built {len(trajectories)} trajectory segment(s) after segmentation")
    if region_config.names_or_labels:
        print(f"Filtering trajectories by region(s): {', '.join(region_config.names_or_labels)}")
    trajectories = _apply_region_filter(trajectories, config=region_config)

    print(f"Kept {len(trajectories)} trajectory segment(s) after region filtering")
    if resample_config.frequency:
        print(f"Resampling trajectories at frequency: {resample_config.frequency}")
    trajectories = _apply_resampling(trajectories, config=resample_config)

    normalized: list[pd.DataFrame] = []
    trajectory_iterator = (
        tqdm(enumerate(trajectories), total=len(trajectories), desc="Normalizing trajectories", unit="traj")
        if len(trajectories) > 1
        else enumerate(trajectories)
    )
    for trajectory_index, trajectory in trajectory_iterator:
        current = trajectory.sort_values("time", kind="stable").reset_index(drop=True).copy()
        current["trajectory"] = trajectory_index
        current["obs"] = np.arange(len(current), dtype=np.int32)
        normalized.append(current)

    return normalized


def convert_argo_to_zarr(config: dict[str, Any]) -> xr.Dataset:
    trajectories = convert_argo_to_dataframe(config)
    print("Building output dataset")
    ds = build_dataset_from_trajectories(trajectories)

    output_path = _resolve_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Zarr dataset to {output_path}")
    ds.to_zarr(output_path, mode="w")
    print("ARGO conversion completed")
    return ds


def run_conversion(config_path: str | Path) -> xr.Dataset:
    config = load_config(config_path)
    return convert_argo_to_zarr(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args.config)


if __name__ == "__main__":
    main()