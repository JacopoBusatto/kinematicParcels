from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class FSLEAnalysisResult:
    spectrum: pd.DataFrame
    crossing_events: pd.DataFrame


def _build_scales(min_scale: float, max_scale: float, rho_increment: float) -> list[float]:
    n = int((np.log(max_scale) - np.log(min_scale)) / np.log(rho_increment))
    scales = [float(min_scale * (rho_increment ** step)) for step in range(n + 1)]
    if len(scales) < 2:
        raise ValueError("FSLE requires at least two scales. Check min_scale, max_scale, and rho_increment.")
    return scales


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
    return EARTH_RADIUS_KM * c


def build_fsle_pair_trajectories(
    df: pd.DataFrame,
    *,
    pair_mode: str = "center_pairs",
) -> pd.DataFrame:
    required = {"group_id", "group_member", "group_size", "obs", "time", "lon", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Input dataframe missing required FSLE columns: {missing}")

    if "group_size" not in df.columns or int(df["group_size"].max()) <= 1:
        raise ValueError("FSLE is only defined for grouped outputs with group_size > 1.")

    if pair_mode not in {"center_pairs", "all_pairs"}:
        raise ValueError("pair_mode must be 'center_pairs' or 'all_pairs'.")

    base = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(base["time"]):
        try:
            base["time"] = pd.to_datetime(base["time"])
        except (TypeError, ValueError):
            pass

    pair_chunks: list[pd.DataFrame] = []
    for group_id, group_data in base.groupby("group_id", sort=True):
        members = sorted(int(member) for member in group_data["group_member"].dropna().unique())
        if len(members) < 2:
            continue

        if pair_mode == "center_pairs":
            if 1 not in members:
                raise ValueError(
                    f"center_pairs requires group_member=1 for every group. Missing in group_id={group_id}."
                )
            pair_members = [(1, member) for member in members if member != 1]
        else:
            pair_members = list(combinations(members, 2))

        for member_a, member_b in pair_members:
            left = group_data[group_data["group_member"] == member_a][
                ["group_id", "group_size", "obs", "time", "lon", "lat"]
            ].rename(columns={"lon": "lon_a", "lat": "lat_a"})
            right = group_data[group_data["group_member"] == member_b][
                ["group_id", "obs", "time", "lon", "lat"]
            ].rename(columns={"lon": "lon_b", "lat": "lat_b"})

            merged = left.merge(
                right,
                on=["group_id", "obs", "time"],
                how="inner",
                sort=True,
            )
            if merged.empty:
                continue

            merged["member_a"] = member_a
            merged["member_b"] = member_b
            merged["pair_id"] = merged["group_id"].astype(str) + f"_m{member_a}_m{member_b}"
            pair_chunks.append(merged)

    if not pair_chunks:
        raise ValueError("No valid FSLE pairs could be constructed from the grouped trajectory table.")

    pairs = pd.concat(pair_chunks, ignore_index=True)
    pairs = pairs.sort_values(["pair_id", "obs"]).reset_index(drop=True)
    pairs["distance_km"] = _haversine_km(
        pairs["lon_a"].to_numpy(),
        pairs["lat_a"].to_numpy(),
        pairs["lon_b"].to_numpy(),
        pairs["lat_b"].to_numpy(),
    )
    return pairs


def _add_elapsed_days(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_series = out["time"]

    if pd.api.types.is_datetime64_any_dtype(time_series):
        first_time = out.groupby("pair_id")["time"].transform("first")
        out["elapsed_days"] = (time_series - first_time).dt.total_seconds() / 86400.0
        return out

    numeric_time = pd.to_numeric(time_series)
    first_time = numeric_time.groupby(out["pair_id"]).transform("first")
    out["elapsed_days"] = numeric_time - first_time
    return out


def _collect_crossing_events(pair_df: pd.DataFrame, scales: list[float]) -> pd.DataFrame:
    pair_df = pair_df.copy()
    valid_entry = (
        pair_df[pair_df["distance_km"] > scales[0]]
        .sort_values(["pair_id", "elapsed_days"])
        .groupby("pair_id", sort=True)
        .first()["elapsed_days"]
        .rename("entry_elapsed_days")
    )

    if valid_entry.empty:
        return pd.DataFrame()

    filtered = pair_df.set_index("pair_id").join(valid_entry).reset_index()
    filtered = filtered[filtered["elapsed_days"] >= filtered["entry_elapsed_days"]].drop(
        columns=["entry_elapsed_days"]
    )

    event_chunks: list[pd.DataFrame] = []
    for scale_old, scale_new in zip(scales[:-1], scales[1:]):
        start = filtered[
            (filtered["distance_km"] >= scale_old) & (filtered["distance_km"] < scale_new)
        ]
        end = filtered[filtered["distance_km"] > scale_new]

        if start.empty or end.empty:
            continue

        start_first = start.sort_values(["pair_id", "elapsed_days"]).groupby("pair_id", sort=True).first()
        end_first = end.sort_values(["pair_id", "elapsed_days"]).groupby("pair_id", sort=True).first()
        if start_first.empty:
            continue

        joined = end_first.rename(
            columns={
                "elapsed_days": "elapsed_days_new",
                "distance_km": "distance_km_new",
                "time": "time_new",
            }
        ).join(
            start_first[
                [
                    "group_id",
                    "group_size",
                    "member_a",
                    "member_b",
                    "elapsed_days",
                    "distance_km",
                    "time",
                ]
            ].rename(
                columns={
                    "elapsed_days": "elapsed_days_old",
                    "distance_km": "distance_km_old",
                    "time": "time_old",
                }
            ),
            how="inner",
            rsuffix="_start",
        )
        if joined.empty:
            continue

        joined = joined.reset_index()
        joined["scale"] = float(scale_old)
        joined["scale_upper"] = float(scale_new)
        joined["time_delta_days"] = joined["elapsed_days_new"] - joined["elapsed_days_old"]
        joined["log_ratio"] = np.log(joined["distance_km_new"] / joined["distance_km_old"])
        event_chunks.append(
            joined[
                [
                    "pair_id",
                    "group_id",
                    "group_size",
                    "member_a",
                    "member_b",
                    "scale",
                    "scale_upper",
                    "distance_km_old",
                    "distance_km_new",
                    "time_old",
                    "time_new",
                    "time_delta_days",
                    "log_ratio",
                ]
            ]
        )

    if not event_chunks:
        return pd.DataFrame()

    events = pd.concat(event_chunks, ignore_index=True)
    events = events[events["time_delta_days"] > 0].reset_index(drop=True)
    return events


def _aggregate_fsle_spectrum(events: pd.DataFrame, rho_increment: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "scale",
                "fsle",
                "n_events",
                "mean_log_ratio",
                "mean_time_delta_days",
                "mean_inverse_time_delta_days",
                "sigma",
                "std",
            ]
        )

    work = events.copy()
    work["inverse_time_delta_days"] = 1.0 / work["time_delta_days"]

    grouped = work.groupby("scale", sort=True)
    mean_log_ratio = grouped["log_ratio"].mean().rename("mean_log_ratio")
    mean_time_delta = grouped["time_delta_days"].mean().rename("mean_time_delta_days")
    mean_inverse_time_delta = grouped["inverse_time_delta_days"].mean().rename(
        "mean_inverse_time_delta_days"
    )
    n_events = grouped.size().rename("n_events")

    spectrum = pd.concat(
        [mean_log_ratio, mean_time_delta, mean_inverse_time_delta, n_events],
        axis=1,
    ).reset_index()
    spectrum["fsle"] = spectrum["mean_log_ratio"] / spectrum["mean_time_delta_days"]

    log_increment = log(rho_increment)
    variance_factor = (
        spectrum["mean_inverse_time_delta_days"] * spectrum["mean_time_delta_days"] - 1.0
    ) / (spectrum["mean_time_delta_days"] ** 2)
    variance_factor = variance_factor.clip(lower=0.0)
    spectrum["sigma"] = log_increment * np.sqrt(variance_factor)
    spectrum["std"] = spectrum["sigma"] / np.sqrt(spectrum["n_events"])

    return spectrum[
        [
            "scale",
            "fsle",
            "n_events",
            "mean_log_ratio",
            "mean_time_delta_days",
            "mean_inverse_time_delta_days",
            "sigma",
            "std",
        ]
    ]


def compute_fsle(
    df: pd.DataFrame,
    *,
    pair_mode: str = "center_pairs",
    min_scale: float = 5.0e-3,
    max_scale: float = 1.0e4,
    rho_increment: float = 2 ** 0.5,
) -> FSLEAnalysisResult:
    pairs = build_fsle_pair_trajectories(df, pair_mode=pair_mode)
    pairs = _add_elapsed_days(pairs)
    scales = _build_scales(min_scale, max_scale, rho_increment)
    crossing_events = _collect_crossing_events(pairs, scales)
    spectrum = _aggregate_fsle_spectrum(crossing_events, rho_increment=rho_increment)
    return FSLEAnalysisResult(spectrum=spectrum, crossing_events=crossing_events)