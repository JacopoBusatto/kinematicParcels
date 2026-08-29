from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod

from .config import GeometryConfig, GridConfig, ModesConfig


@dataclass(frozen=True)
class GeometryResult:
    transitions: pd.DataFrame
    cell_diagnostics: pd.DataFrame
    dataset: xr.Dataset


def compute_transition_geometry(
    transitions: pd.DataFrame,
    grid: GridConfig,
    config: GeometryConfig,
    modes_config: ModesConfig | None = None,
) -> GeometryResult:
    df = transitions.copy()
    geod = Geod(ellps=config.ellipsoid)
    azimuth, _, distance_m = geod.inv(
        df.start_lon_center.to_numpy(float), df.start_lat_center.to_numpy(float),
        df.end_lon_center.to_numpy(float), df.end_lat_center.to_numpy(float),
    )
    bearing = azimuth % 360.0
    distance = distance_m / 1000.0
    is_stay = df.start_cell_id.eq(df.end_cell_id).to_numpy()
    distance[is_stay] = 0.0
    bearing[is_stay] = np.nan
    radians = np.deg2rad(np.nan_to_num(bearing))
    df["dx_km"] = distance * np.sin(radians)
    df["dy_km"] = distance * np.cos(radians)
    df["distance_km"] = distance
    df["bearing_deg"] = bearing
    df["is_stay"] = is_stay

    p_stay = df.loc[is_stay].groupby("start_cell_id").transition_probability.sum()
    df["P_stay"] = df.start_cell_id.map(p_stay).fillna(0.0)
    df["P_move"] = 1.0 - df.P_stay
    df["conditional_moving_probability"] = np.where(
        (~is_stay) & (df.P_move > 0), df.transition_probability / df.P_move, np.nan
    )

    records: list[dict] = []
    angular_bins = 72 if modes_config is None else modes_config.angular_bins
    for cell_id, group in df.groupby("start_cell_id", sort=True):
        first = group.iloc[0]
        moving = group.loc[~group.is_stay]
        n_i = int(group.transition_count.sum())
        rec = {
            "start_cell_id": int(cell_id),
            "start_lon_bin": int(first.start_lon_bin),
            "start_lat_bin": int(first.start_lat_bin),
            "lon": float(first.start_lon_center),
            "lat": float(first.start_lat_center),
            "N_i": n_i,
            "n_destinations": len(group),
            "P_stay": float(group.P_stay.iloc[0]),
            "P_move": float(group.P_move.iloc[0]),
            "moving_count": int(moving.transition_count.sum()),
        }
        if moving.empty or rec["P_move"] <= 0:
            rec.update({name: np.nan for name in (
                "mean_dx_km", "mean_dy_km", "mean_displacement_km", "mean_bearing_deg",
                "mean_transition_distance_km", "cov_xx_km2", "cov_xy_km2", "cov_yy_km2",
                "major_eigenvalue_km2", "minor_eigenvalue_km2", "ellipse_angle_deg",
                "ellipse_major_scale_km", "ellipse_minor_scale_km", "anisotropy",
                "angular_entropy", "circular_concentration",
            )})
            records.append(rec)
            continue
        w = moving.conditional_moving_probability.to_numpy(float)
        xy = moving[["dx_km", "dy_km"]].to_numpy(float)
        mean = np.sum(w[:, None] * xy, axis=0)
        centered = xy - mean
        cov = (centered * w[:, None]).T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        minor, major = np.maximum(eigenvalues, 0.0)
        major_vec = eigenvectors[:, 1]
        angle = np.degrees(np.arctan2(major_vec[0], major_vec[1])) % 180.0
        bearing_rad = np.deg2rad(moving.bearing_deg.to_numpy(float))
        c = float(np.sum(w * np.cos(bearing_rad)))
        s = float(np.sum(w * np.sin(bearing_rad)))
        hist, _ = np.histogram(moving.bearing_deg, bins=angular_bins, range=(0, 360), weights=w)
        positive = hist[hist > 0]
        entropy = float(-np.sum(positive * np.log(positive)) / np.log(angular_bins)) if len(positive) else np.nan
        rec.update({
            "mean_dx_km": float(mean[0]), "mean_dy_km": float(mean[1]),
            "mean_displacement_km": float(np.hypot(*mean)),
            "mean_bearing_deg": float(np.degrees(np.arctan2(mean[0], mean[1])) % 360),
            "mean_transition_distance_km": float(np.sum(w * moving.distance_km)),
            "cov_xx_km2": float(cov[0, 0]), "cov_xy_km2": float(cov[0, 1]),
            "cov_yy_km2": float(cov[1, 1]), "major_eigenvalue_km2": float(major),
            "minor_eigenvalue_km2": float(minor), "ellipse_angle_deg": float(angle),
            "ellipse_major_scale_km": float(np.sqrt(major)),
            "ellipse_minor_scale_km": float(np.sqrt(minor)),
            "anisotropy": float((major - minor) / (major + minor)) if major + minor > 0 else 0.0,
            "angular_entropy": entropy, "circular_concentration": float(np.hypot(c, s)),
        })
        records.append(rec)

    cells = pd.DataFrame(records)
    ds = diagnostics_dataset(cells, grid)
    return GeometryResult(df, cells, ds)


def diagnostics_dataset(cells: pd.DataFrame, grid: GridConfig) -> xr.Dataset:
    lat = grid.lat_min + (np.arange(grid.nlat) + 0.5) * grid.dlat
    lon = grid.lon_min + (np.arange(grid.nlon) + 0.5) * grid.dlon
    excluded = {"start_cell_id", "start_lon_bin", "start_lat_bin", "lon", "lat"}
    data_vars = {}
    for column in cells.columns:
        if column in excluded:
            continue
        values = np.full((grid.nlat, grid.nlon), np.nan, dtype=float)
        values[cells.start_lat_bin.to_numpy(int), cells.start_lon_bin.to_numpy(int)] = cells[column].to_numpy(float)
        data_vars[column] = (("lat", "lon"), values)
    return xr.Dataset(data_vars, coords={"lat": lat, "lon": lon}, attrs={
        "title": "Conditional transition displacement diagnostics",
        "geometry": "WGS84 geodesic inverse; covariance is not diffusivity",
    })
