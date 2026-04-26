from __future__ import annotations

from datetime import timedelta

import numpy as np


EARTH_METERS_PER_DEG_LAT = 111320.0


def parse_timedelta_like(value: str) -> timedelta:
    s = str(value).strip().lower()

    if s.endswith("min"):
        return timedelta(minutes=float(s[:-3]))
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))

    raise ValueError(
        f"Unsupported duration format: {value}. "
        "Use e.g. 30min, 12H, 1D, 2.5D"
    )


def infer_depth_convention(fieldset) -> str:
    depth_axis = getattr(fieldset.U.grid, "depth", None)
    if depth_axis is None:
        return "positive_down"

    depth_vals = np.asarray(depth_axis, dtype=float)
    if depth_vals.size == 0:
        return "positive_down"

    return "negative_down" if np.nanmedian(depth_vals) < 0 else "positive_down"


def to_positive_down(depth_values, convention: str):
    arr = np.asarray(depth_values, dtype=float)
    if convention == "positive_down":
        return arr
    if convention == "negative_down":
        return -arr
    raise ValueError(f"Unsupported depth convention: {convention}")


def from_positive_down(depth_values_pd, convention: str):
    arr = np.asarray(depth_values_pd, dtype=float)
    if convention == "positive_down":
        return arr
    if convention == "negative_down":
        return -arr
    raise ValueError(f"Unsupported depth convention: {convention}")


def depth_axis_max_positive_down(fieldset) -> float | None:
    depth_axis = getattr(fieldset.U.grid, "depth", None)
    if depth_axis is None:
        return None

    depth_vals = np.asarray(depth_axis, dtype=float)
    if depth_vals.size == 0:
        return None

    conv = infer_depth_convention(fieldset)
    depth_vals_pd = to_positive_down(depth_vals, conv)
    return float(np.nanmax(depth_vals_pd))


def _uniform_disk_offsets(rng: np.random.Generator, n: int, radius_m: float):
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rr = radius_m * np.sqrt(rng.random(size=n))
    return rr * np.cos(theta), rr * np.sin(theta)


def _uniform_sphere_offsets(
    rng: np.random.Generator, n: int, radius_m: float, depth_radius_m: float
):
    """Uniform distribution inside an ellipsoid with horizontal semi-axis radius_m
    and vertical semi-axis depth_radius_m."""
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    cos_phi = rng.uniform(-1.0, 1.0, size=n)
    sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
    # rr is the unit-sphere radius; axes are scaled independently after
    rr = np.cbrt(rng.random(size=n))

    dx = radius_m * rr * sin_phi * np.cos(theta)
    dy = radius_m * rr * sin_phi * np.sin(theta)
    dz = depth_radius_m * rr * cos_phi
    return dx, dy, dz


def _gaussian_inside_radius(
    rng: np.random.Generator,
    n: int,
    radius_m: float,
    dim3: bool,
    depth_radius_m: float = 0.0,
):
    out = []
    collected = 0

    while collected < n:
        batch = max(32, int((n - collected) * 2.5))
        if dim3:
            sigma_h = radius_m / 3.0
            sigma_v = depth_radius_m / 3.0
            xy = rng.normal(0.0, sigma_h, size=(batch, 2))
            z = rng.normal(0.0, sigma_v, size=(batch, 1))
            xyz = np.hstack([xy, z])
            # clip to ellipsoid: (x/r_h)^2 + (y/r_h)^2 + (z/r_v)^2 <= 1
            ellipsoid_r2 = (
                (xyz[:, 0] / radius_m) ** 2
                + (xyz[:, 1] / radius_m) ** 2
                + (xyz[:, 2] / depth_radius_m) ** 2
            )
            keep = xyz[ellipsoid_r2 <= 1.0]
            if keep.size == 0:
                continue
            out.append(keep)
            collected += keep.shape[0]
        else:
            sigma = radius_m / 3.0
            xy = rng.normal(0.0, sigma, size=(batch, 2))
            r2 = np.sum(xy * xy, axis=1)
            keep = xy[r2 <= radius_m * radius_m]
            if keep.size == 0:
                continue
            out.append(keep)
            collected += keep.shape[0]

    stacked = np.vstack(out)[:n]
    if dim3:
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]
    return stacked[:, 0], stacked[:, 1]


def sample_circle_or_sphere(
    *,
    center_lon: float,
    center_lat: float,
    center_depth_pd: float | None,
    radius_km: float,
    depth_radius_m: float | None,
    count: int,
    dimension: str,
    sampling: str,
    rng: np.random.Generator,
):
    if count <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), None

    radius_m = float(radius_km) * 1000.0
    dim = str(dimension).strip().lower()
    if dim not in {"2d", "3d"}:
        raise ValueError("release.circle.dimension must be '2D' or '3D'")

    sampling_mode = str(sampling).strip().lower()
    if sampling_mode not in {"uniform", "gaussian"}:
        raise ValueError("release.circle.sampling must be 'uniform' or 'gaussian'")

    if sampling_mode == "uniform":
        if dim == "2d":
            dx, dy = _uniform_disk_offsets(rng, count, radius_m)
            dz = None
        else:
            dx, dy, dz = _uniform_sphere_offsets(rng, count, radius_m, float(depth_radius_m))
    else:
        if dim == "2d":
            dx, dy = _gaussian_inside_radius(rng, count, radius_m, dim3=False)
            dz = None
        else:
            dx, dy, dz = _gaussian_inside_radius(
                rng, count, radius_m, dim3=True, depth_radius_m=float(depth_radius_m)
            )

    lat0_rad = np.radians(center_lat)
    meters_per_deg_lon = EARTH_METERS_PER_DEG_LAT * np.cos(lat0_rad)
    if np.abs(meters_per_deg_lon) < 1e-12:
        meters_per_deg_lon = 1e-12

    lons = center_lon + dx / meters_per_deg_lon
    lats = center_lat + dy / EARTH_METERS_PER_DEG_LAT

    if dim == "2d":
        return lons.astype(float), lats.astype(float), None

    if center_depth_pd is None:
        raise ValueError("3D release requires center depth")

    depths_pd = float(center_depth_pd) + dz
    return lons.astype(float), lats.astype(float), depths_pd.astype(float)
