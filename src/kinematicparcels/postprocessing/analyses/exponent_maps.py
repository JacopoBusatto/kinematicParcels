from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fsle import _haversine_km, _meridional_distance_km


@dataclass(frozen=True)
class ExponentMapsAnalysisResult:
    fsle_points: pd.DataFrame
    ftle_points: pd.DataFrame
    simulation_direction: str


def _require_grouped_columns(df: pd.DataFrame) -> None:
    required = {"group_id", "group_member", "group_size", "obs", "time", "lon", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Input dataframe missing required exponent-map columns: {missing}")

    if df.empty:
        raise ValueError("Exponent maps require a non-empty grouped trajectory table.")

    if int(df["group_size"].max()) <= 1:
        raise ValueError("Exponent maps require grouped outputs with group_size > 1.")


def _elapsed_days_from_release(
    time_values: pd.Series,
    release_time_values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(time_values):
        delta = pd.to_datetime(time_values) - pd.to_datetime(release_time_values)
        return delta.dt.total_seconds().abs() / 86400.0

    numeric_time = pd.to_numeric(time_values)
    numeric_release = pd.to_numeric(release_time_values)
    return (numeric_time - numeric_release).abs()


def _build_group_release_centers(df: pd.DataFrame) -> pd.DataFrame:
    centers: list[dict[str, object]] = []
    for group_id, group_data in df.groupby("group_id", sort=True):
        center_rows = group_data.loc[group_data["group_member"] == 1].sort_values("obs")
        if center_rows.empty:
            raise ValueError(f"Exponent maps require group_member=1 for every group. Missing in group_id={group_id}.")

        first = center_rows.iloc[0]
        centers.append(
            {
                "group_id": group_id,
                "time0": first["time"],
                "lon0": float(first["lon"]),
                "lat0": float(first["lat"]),
                "group_size": int(first["group_size"]),
            }
        )

    return pd.DataFrame(centers).sort_values("group_id").reset_index(drop=True)


def _build_center_member_pairs(
    df: pd.DataFrame,
    *,
    meridional_only: bool,
) -> pd.DataFrame:
    pair_chunks: list[pd.DataFrame] = []

    for group_id, group_data in df.groupby("group_id", sort=True):
        members = sorted(int(member) for member in group_data["group_member"].dropna().unique())
        if len(members) < 2:
            continue
        if 1 not in members:
            raise ValueError(f"Exponent maps require group_member=1 for every group. Missing in group_id={group_id}.")

        center = group_data.loc[group_data["group_member"] == 1, ["group_id", "group_size", "obs", "time", "lon", "lat"]]
        center = center.rename(
            columns={
                "time": "time_center",
                "lon": "lon_center",
                "lat": "lat_center",
            }
        )

        for member in members:
            if member == 1:
                continue

            partner = group_data.loc[group_data["group_member"] == member, ["group_id", "obs", "time", "lon", "lat"]]
            partner = partner.rename(
                columns={
                    "time": "time_member",
                    "lon": "lon_member",
                    "lat": "lat_member",
                }
            )

            merged = center.merge(partner, on=["group_id", "obs"], how="inner", sort=True)
            if merged.empty:
                continue

            merged["member_b"] = member
            merged["pair_id"] = merged["group_id"].astype(str) + f"_m1_m{member}"
            merged["time"] = merged["time_center"]

            if meridional_only:
                merged["distance_km"] = _meridional_distance_km(
                    merged["lat_center"].to_numpy(),
                    merged["lat_member"].to_numpy(),
                )
            else:
                merged["distance_km"] = _haversine_km(
                    merged["lon_center"].to_numpy(),
                    merged["lat_center"].to_numpy(),
                    merged["lon_member"].to_numpy(),
                    merged["lat_member"].to_numpy(),
                )

            pair_chunks.append(merged)

    if not pair_chunks:
        raise ValueError("No valid center-member pairs could be constructed for exponent maps.")

    pairs = pd.concat(pair_chunks, ignore_index=True)
    pairs = pairs.sort_values(["pair_id", "obs"]).reset_index(drop=True)

    pairs["time0"] = pairs.groupby("pair_id", sort=True)["time"].transform("first")
    pairs["lon0"] = pairs.groupby("pair_id", sort=True)["lon_center"].transform("first")
    pairs["lat0"] = pairs.groupby("pair_id", sort=True)["lat_center"].transform("first")
    pairs["initial_distance_km"] = pairs.groupby("pair_id", sort=True)["distance_km"].transform("first")
    pairs["age_days"] = _elapsed_days_from_release(pairs["time"], pairs["time0"])
    pairs = pairs.loc[pairs["initial_distance_km"] > 0].copy()

    if pairs.empty:
        raise ValueError("All exponent-map pairs have zero initial separation; cannot compute exponents.")

    return pairs.reset_index(drop=True)


def _infer_simulation_direction(pairs: pd.DataFrame) -> str:
    sample = pairs.groupby("pair_id", sort=True).head(2)
    if len(sample) < 2:
        return "forward"

    sample = sample.sort_values(["pair_id", "obs"]).reset_index(drop=True)
    t0 = sample.iloc[0]["time"]
    t1 = sample.iloc[1]["time"]
    return "backward" if t1 < t0 else "forward"


def _cross_join_with_scales(
    centers: pd.DataFrame,
    *,
    scale_values: tuple[float, ...],
    scale_col: str,
) -> pd.DataFrame:
    return (
        centers.assign(_merge_key=1)
        .merge(pd.DataFrame({scale_col: scale_values, "_merge_key": 1}), on="_merge_key", how="inner")
        .drop(columns="_merge_key")
    )


def _finalize_group_scale_points(
    centers: pd.DataFrame,
    *,
    scale_values: tuple[float, ...],
    scale_col: str,
    value_col: str,
    candidates: pd.DataFrame,
    mask_missing_as_nan: bool,
    mask_zeros: bool,
    simulation_direction: str,
) -> pd.DataFrame:
    base = _cross_join_with_scales(centers, scale_values=scale_values, scale_col=scale_col)
    if candidates.empty:
        merged = base.copy()
        merged[value_col] = np.nan
    else:
        merged = base.merge(candidates, on=["group_id", scale_col], how="left", suffixes=("", "_candidate"))
    merged["is_valid"] = merged[value_col].notna()

    if not mask_missing_as_nan:
        merged[value_col] = merged[value_col].fillna(0.0)

    if simulation_direction == "backward":
        merged[value_col] = -1.0 * merged[value_col]

    if mask_zeros:
        merged.loc[merged[value_col] == 0.0, value_col] = np.nan

    return merged.sort_values(["time0", scale_col, "group_id"]).reset_index(drop=True)


def _compute_fsle_points(
    pairs: pd.DataFrame,
    centers: pd.DataFrame,
    *,
    scales_km: tuple[float, ...],
    mask_zeros: bool,
    simulation_direction: str,
) -> pd.DataFrame:
    if len(scales_km) == 0:
        return pd.DataFrame()

    candidate_chunks: list[pd.DataFrame] = []
    positive_age = pairs.loc[pairs["age_days"] > 0].copy()

    for scale_km in scales_km:
        crossed = positive_age.loc[positive_age["distance_km"] >= scale_km].copy()
        if crossed.empty:
            continue

        first_cross = (
            crossed.sort_values(["pair_id", "obs"])
            .groupby("pair_id", sort=True)
            .first()
            .reset_index()
        )
        first_cross["scale_km"] = float(scale_km)
        first_cross["fsle"] = np.log(first_cross["distance_km"] / first_cross["initial_distance_km"]) / first_cross["age_days"]
        candidate_chunks.append(first_cross)

    candidates = pd.concat(candidate_chunks, ignore_index=True) if candidate_chunks else pd.DataFrame()
    if not candidates.empty:
        candidates.loc[candidates["distance_km"] <= candidates["initial_distance_km"], "fsle"] = 0.0
        candidates = (
            candidates.sort_values(["group_id", "scale_km", "age_days", "pair_id"])
            .groupby(["group_id", "scale_km"], sort=True)
            .first()
            .reset_index()
        )
        candidates = candidates[
            [
                "group_id",
                "scale_km",
                "pair_id",
                "member_b",
                "age_days",
                "distance_km",
                "initial_distance_km",
                "fsle",
            ]
        ]

    return _finalize_group_scale_points(
        centers,
        scale_values=scales_km,
        scale_col="scale_km",
        value_col="fsle",
        candidates=candidates,
        mask_missing_as_nan=mask_zeros,
        mask_zeros=mask_zeros,
        simulation_direction=simulation_direction,
    )


def _select_ftle_row(pair_data: pd.DataFrame, *, target_age_days: float, sampling_mode: str) -> pd.Series | None:
    if sampling_mode == "last_before_or_at":
        exact = pair_data.loc[np.isclose(pair_data["age_days"], target_age_days)].copy()
        if not exact.empty:
            return exact.sort_values("obs").iloc[-1]

        if not (pair_data["age_days"] > target_age_days).any():
            return None

        valid = pair_data.loc[(pair_data["age_days"] > 0) & (pair_data["age_days"] < target_age_days)].copy()
        if valid.empty:
            return None

        return valid.sort_values("obs").iloc[-1]

    valid = pair_data.loc[(pair_data["age_days"] > 0) & (pair_data["age_days"] <= target_age_days)].copy()
    if valid.empty:
        return None

    max_idx = valid["distance_km"].idxmax()
    return valid.loc[max_idx]


def _compute_ftle_points(
    pairs: pd.DataFrame,
    centers: pd.DataFrame,
    *,
    scales_days: tuple[float, ...],
    sampling_mode: str,
    mask_short_windows: bool,
    mask_zeros: bool,
    simulation_direction: str,
) -> pd.DataFrame:
    if len(scales_days) == 0:
        return pd.DataFrame()

    candidate_rows: list[dict[str, object]] = []
    for scale_days in scales_days:
        for pair_id, pair_data in pairs.groupby("pair_id", sort=True):
            pair_data = pair_data.sort_values("obs").reset_index(drop=True)
            selected = _select_ftle_row(
                pair_data,
                target_age_days=float(scale_days),
                sampling_mode=sampling_mode,
            )
            if selected is None:
                continue

            sampled_age_days = float(selected["age_days"])
            sampled_distance_km = float(selected["distance_km"])
            initial_distance_km = float(selected["initial_distance_km"])
            ftle_value = 0.0
            if sampled_distance_km > initial_distance_km:
                ftle_value = np.log(sampled_distance_km / initial_distance_km) / sampled_age_days

            candidate_rows.append(
                {
                    "group_id": selected["group_id"],
                    "scale_days": float(scale_days),
                    "pair_id": pair_id,
                    "member_b": int(selected["member_b"]),
                    "sampled_age_days": sampled_age_days,
                    "sampled_distance_km": sampled_distance_km,
                    "initial_distance_km": initial_distance_km,
                    "ftle": ftle_value,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = (
            candidates.sort_values(
                ["group_id", "scale_days", "sampled_distance_km", "ftle", "pair_id"],
                ascending=[True, True, False, False, True],
            )
            .groupby(["group_id", "scale_days"], sort=True)
            .first()
            .reset_index()
        )

    return _finalize_group_scale_points(
        centers,
        scale_values=scales_days,
        scale_col="scale_days",
        value_col="ftle",
        candidates=candidates,
        mask_missing_as_nan=mask_short_windows or mask_zeros,
        mask_zeros=mask_zeros,
        simulation_direction=simulation_direction,
    )


def compute_exponent_maps(
    df: pd.DataFrame,
    *,
    meridional_only: bool,
    fsle_scales_km: tuple[float, ...] = (),
    fsle_mask_zeros: bool = False,
    ftle_scales_days: tuple[float, ...] = (),
    ftle_sampling_mode: str = "last_before_or_at",
    ftle_mask_short_windows: bool = True,
    ftle_mask_zeros: bool = False,
) -> ExponentMapsAnalysisResult:
    _require_grouped_columns(df)

    centers = _build_group_release_centers(df)
    pairs = _build_center_member_pairs(df, meridional_only=meridional_only)
    simulation_direction = _infer_simulation_direction(pairs)

    fsle_points = _compute_fsle_points(
        pairs,
        centers,
        scales_km=fsle_scales_km,
        mask_zeros=fsle_mask_zeros,
        simulation_direction=simulation_direction,
    )
    ftle_points = _compute_ftle_points(
        pairs,
        centers,
        scales_days=ftle_scales_days,
        sampling_mode=ftle_sampling_mode,
        mask_short_windows=ftle_mask_short_windows,
        mask_zeros=ftle_mask_zeros,
        simulation_direction=simulation_direction,
    )

    return ExponentMapsAnalysisResult(
        fsle_points=fsle_points,
        ftle_points=ftle_points,
        simulation_direction=simulation_direction,
    )