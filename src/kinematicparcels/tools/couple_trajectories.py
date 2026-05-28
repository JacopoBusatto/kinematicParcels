from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from kinematicparcels.postprocessing.io.parcels import load_trajectory_table, open_parcels_dataset
from kinematicparcels.regions import ALL_REGIONS, RegionManager
from kinematicparcels.tools.zarr_writer import build_dataset_from_trajectories, build_zarr_encoding


TRAJECTORY_LEVEL_COLUMNS = {"group_id", "group_size", "platform_code_1", "platform_code_2"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build grouped-entity pair trajectories from a Parcels-compatible input Zarr dataset."
    )
    parser.add_argument("input_zarr", type=str, help="Path to the input trajectory Zarr dataset.")
    parser.add_argument("output_zarr", type=str, help="Path to the output grouped-entity Zarr dataset.")
    parser.add_argument(
        "--threshold-km",
        type=float,
        required=True,
        help="Maximum closest-approach distance required to accept a pair.",
    )
    parser.add_argument(
        "--minimum-life-days",
        type=float,
        default=None,
        help="Minimum duration in days after closest approach. Defaults to no duration filter.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional region labels or names. If provided, the accepted closest approach must occur inside one of these regions.",
    )
    return parser


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
        raise ValueError(f"Unknown region labels/names requested for trajectory coupling: {missing_str}")

    return RegionManager(selected_regions)


def _point_in_regions(
    lon: float,
    lat: float,
    *,
    region_manager: RegionManager,
    input_lon_mode: str = "-180_180",
) -> bool:
    return bool(
        region_manager.find_regions(
            lon,
            lat,
            howMany="first",
            input_lon_mode=input_lon_mode,
        )
    )


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


def _circular_mean_longitude(lon1: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    mean_rad = np.arctan2(
        np.sin(lon1_rad) + np.sin(lon2_rad),
        np.cos(lon1_rad) + np.cos(lon2_rad),
    )
    return ((np.degrees(mean_rad) + 180.0) % 360.0) - 180.0


def _resolve_extra_input_vars(path: str | Path) -> list[str]:
    ds = open_parcels_dataset(path)
    try:
        available = {str(name) for name in ds.variables}
    finally:
        ds.close()

    return [name for name in ("platform_code",) if name in available]


def _prepare_trajectory_groups(df: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    prepared: list[tuple[object, pd.DataFrame]] = []
    for trajectory_id, group in df.groupby("trajectory", sort=True):
        current = group.sort_values("time", kind="stable").reset_index(drop=True).copy()
        current = current.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
        if len(current) < 2:
            continue
        prepared.append((trajectory_id, current))
    return prepared


def _candidate_pair_segment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    threshold_km: float,
    minimum_life_days: float | None,
    region_manager: RegionManager | None,
    group_id: int,
) -> pd.DataFrame | None:
    left_time_min = left["time"].iloc[0]
    left_time_max = left["time"].iloc[-1]
    right_time_min = right["time"].iloc[0]
    right_time_max = right["time"].iloc[-1]

    if left_time_max < right_time_min or right_time_max < left_time_min:
        return None

    merge_columns = ["time", "lon", "lat"]
    if "z" in left.columns and "z" in right.columns:
        merge_columns.append("z")
    if "platform_code" in left.columns and "platform_code" in right.columns:
        merge_columns.append("platform_code")

    merged = left[merge_columns].merge(
        right[merge_columns],
        on="time",
        how="inner",
        suffixes=("_1", "_2"),
        sort=True,
    )
    if len(merged) < 2:
        return None

    distances = _haversine_km(
        merged["lon_1"].to_numpy(dtype=float),
        merged["lat_1"].to_numpy(dtype=float),
        merged["lon_2"].to_numpy(dtype=float),
        merged["lat_2"].to_numpy(dtype=float),
    )
    candidate_indices = np.flatnonzero(distances <= threshold_km)
    if candidate_indices.size == 0:
        return None

    if region_manager is None:
        min_index = int(candidate_indices[np.argmin(distances[candidate_indices])])
    else:
        center_lon = _circular_mean_longitude(
            merged["lon_1"].to_numpy(dtype=float),
            merged["lon_2"].to_numpy(dtype=float),
        )
        center_lat = (
            merged["lat_1"].to_numpy(dtype=float) + merged["lat_2"].to_numpy(dtype=float)
        ) / 2.0
        region_hits = np.array(
            [
                _point_in_regions(
                    float(center_lon[index]),
                    float(center_lat[index]),
                    region_manager=region_manager,
                )
                for index in candidate_indices
            ],
            dtype=bool,
        )
        if not region_hits.any():
            return None
        eligible_indices = candidate_indices[region_hits]
        min_index = int(eligible_indices[np.argmin(distances[eligible_indices])])

    pair = merged.iloc[min_index:].reset_index(drop=True).copy()
    if len(pair) < 2:
        return None

    pair_start_time = pair.loc[0, "time"]
    age_days = (pair["time"] - pair_start_time).dt.total_seconds() / 86400.0
    if minimum_life_days is not None and float(age_days.iloc[-1]) < minimum_life_days:
        return None

    pair["trajectory"] = group_id
    pair["obs"] = np.arange(len(pair), dtype=np.int32)
    pair["group_id"] = group_id
    pair["group_size"] = 2
    pair["center_lon"] = _circular_mean_longitude(
        pair["lon_1"].to_numpy(dtype=float),
        pair["lon_2"].to_numpy(dtype=float),
    )
    pair["center_lat"] = (
        pair["lat_1"].to_numpy(dtype=float) + pair["lat_2"].to_numpy(dtype=float)
    ) / 2.0
    pair["lon"] = pair["center_lon"]
    pair["lat"] = pair["center_lat"]
    if "z_1" in pair.columns and "z_2" in pair.columns:
        pair["z"] = np.nanmean(
            np.column_stack(
                [
                    pair["z_1"].to_numpy(dtype=float),
                    pair["z_2"].to_numpy(dtype=float),
                ]
            ),
            axis=1,
        )
    else:
        pair["z"] = 0.0

    output_columns = [
        "trajectory",
        "obs",
        "time",
        "lon",
        "lat",
        "z",
        "group_id",
        "group_size",
        "center_lon",
        "center_lat",
        "lon_1",
        "lat_1",
        "lon_2",
        "lat_2",
    ]

    if "platform_code_1" in pair.columns and "platform_code_2" in pair.columns:
        output_columns.extend(["platform_code_1", "platform_code_2"])

    return pair[output_columns]


def build_coupled_trajectories(
    input_path: str | Path,
    *,
    threshold_km: float,
    minimum_life_days: float | None = None,
    regions: tuple[str, ...] = (),
) -> list[pd.DataFrame]:
    if threshold_km <= 0.0:
        raise ValueError("threshold_km must be > 0")
    if minimum_life_days is not None and minimum_life_days < 0.0:
        raise ValueError("minimum_life_days must be >= 0 when provided")

    region_manager = _build_selected_region_manager(regions)
    extra_vars = _resolve_extra_input_vars(input_path)
    table = load_trajectory_table(input_path, extra_vars=extra_vars)
    trajectories = _prepare_trajectory_groups(table)

    coupled: list[pd.DataFrame] = []
    pair_iterator = combinations(trajectories, 2)
    total_pairs = len(trajectories) * (len(trajectories) - 1) // 2
    if total_pairs > 1:
        pair_iterator = tqdm(pair_iterator, total=total_pairs, desc="Evaluating pairs", unit="pair")

    next_group_id = 0
    for (_, left), (_, right) in pair_iterator:
        candidate = _candidate_pair_segment(
            left,
            right,
            threshold_km=threshold_km,
            minimum_life_days=minimum_life_days,
            region_manager=region_manager,
            group_id=next_group_id,
        )
        if candidate is None:
            continue
        coupled.append(candidate)
        next_group_id += 1

    return coupled


def couple_trajectories_to_zarr(
    input_path: str | Path,
    output_path: str | Path,
    *,
    threshold_km: float,
    minimum_life_days: float | None = None,
    regions: tuple[str, ...] = (),
) -> xr.Dataset:
    trajectories = build_coupled_trajectories(
        input_path,
        threshold_km=threshold_km,
        minimum_life_days=minimum_life_days,
        regions=regions,
    )
    if not trajectories:
        raise ValueError("No trajectory pairs satisfied the requested distance and lifetime filters.")

    ds = build_dataset_from_trajectories(
        trajectories,
        trajectory_level_columns=TRAJECTORY_LEVEL_COLUMNS,
    )
    ds.attrs["source"] = "Trajectory pair coupling"
    ds.attrs["pair_threshold_km"] = float(threshold_km)
    if minimum_life_days is not None:
        ds.attrs["minimum_life_days"] = float(minimum_life_days)
    if regions:
        ds.attrs["pair_regions"] = list(regions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(output_path, mode="w", encoding=build_zarr_encoding(ds))

    average_duration_days = float(
        np.mean(
            [
                (trajectory["time"].iloc[-1] - trajectory["time"].iloc[0]).total_seconds() / 86400.0
                for trajectory in trajectories
            ]
        )
    )
    print(
        f"Found {len(trajectories)} couples. Average length: {average_duration_days:.2f} days."
    )
    return ds


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    couple_trajectories_to_zarr(
        args.input_zarr,
        args.output_zarr,
        threshold_km=args.threshold_km,
        minimum_life_days=args.minimum_life_days,
        regions=tuple(args.regions or ()),
    )


if __name__ == "__main__":
    main()