from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


_SUPPORTED_AGGREGATIONS = {
    "mean",
    "count",
    "sum",
    "min",
    "max",
    "median",
    "std",
}


@dataclass(frozen=True)
class RegularGrid:
    """
    Regular lon/lat grid definition.

    The grid geometry is defined explicitly by bounds and spacing.
    Grid cell centers are always reconstructed from integer bin indices,
    so we do not rely on floating-point coordinates found in data tables.
    """
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    dlon: float
    dlat: float

    def __post_init__(self) -> None:
        if self.dlon <= 0 or self.dlat <= 0:
            raise ValueError("dlon and dlat must be positive.")

        if self.lon_max <= self.lon_min:
            raise ValueError("lon_max must be greater than lon_min.")

        if self.lat_max <= self.lat_min:
            raise ValueError("lat_max must be greater than lat_min.")

    @classmethod
    def from_point_centers(
        cls,
        lon: np.ndarray | pd.Series,
        lat: np.ndarray | pd.Series,
        *,
        dlon: float,
        dlat: float,
        eps: float = 1.0e-12,
    ) -> "RegularGrid":
        """
        Build a RegularGrid assuming the provided lon/lat values are
        pixel centers, not edges.

        Parameters
        ----------
        lon, lat
            Arrays of point coordinates interpreted as grid-cell centers.
        dlon, dlat
            Grid spacing.
        eps
            Small positive margin added to the upper edge to avoid losing
            the last cell because of floating point issues.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if lon.size == 0 or lat.size == 0:
            raise ValueError("Cannot build grid from empty lon/lat arrays.")

        lon_min = float(np.nanmin(lon)) - 0.5 * dlon
        lon_max = float(np.nanmax(lon)) + 0.5 * dlon + eps
        lat_min = float(np.nanmin(lat)) - 0.5 * dlat
        lat_max = float(np.nanmax(lat)) + 0.5 * dlat + eps

        return cls(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            dlon=dlon,
            dlat=dlat,
        )


    @classmethod
    def from_aligned_initial_centers(
        cls,
        lon: np.ndarray | pd.Series,
        lat: np.ndarray | pd.Series,
        *,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        dlon: float,
        dlat: float,
        eps: float = 1.0e-12,
    ) -> "RegularGrid":
        """
        Build a grid aligned to the phase of the provided initial particle centers,
        but extended to the requested analysis bounds.

        Parameters
        ----------
        lon, lat
            Initial particle positions interpreted as grid-cell centers.
        lon_min, lon_max, lat_min, lat_max
            Desired analysis-domain bounds.
        dlon, dlat
            Grid spacing.
        eps
            Small positive margin added to upper bounds.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if lon.size == 0 or lat.size == 0:
            raise ValueError("Cannot build grid from empty lon/lat arrays.")

        # Base edge implied by the initial centers
        base_lon_edge = float(np.nanmin(lon)) - 0.5 * dlon
        base_lat_edge = float(np.nanmin(lat)) - 0.5 * dlat

        # Snap requested bounds outward to the aligned grid
        aligned_lon_min = base_lon_edge + np.floor((lon_min - base_lon_edge) / dlon) * dlon
        aligned_lon_max = base_lon_edge + np.ceil((lon_max - base_lon_edge) / dlon) * dlon + eps

        aligned_lat_min = base_lat_edge + np.floor((lat_min - base_lat_edge) / dlat) * dlat
        aligned_lat_max = base_lat_edge + np.ceil((lat_max - base_lat_edge) / dlat) * dlat + eps

        return cls(
            lon_min=float(aligned_lon_min),
            lon_max=float(aligned_lon_max),
            lat_min=float(aligned_lat_min),
            lat_max=float(aligned_lat_max),
            dlon=dlon,
            dlat=dlat,
        )


    @property
    def nlon(self) -> int:
        return int(np.ceil((self.lon_max - self.lon_min) / self.dlon))

    @property
    def nlat(self) -> int:
        return int(np.ceil((self.lat_max - self.lat_min) / self.dlat))

    @property
    def lon_edges(self) -> np.ndarray:
        return self.lon_min + np.arange(self.nlon + 1) * self.dlon

    @property
    def lat_edges(self) -> np.ndarray:
        return self.lat_min + np.arange(self.nlat + 1) * self.dlat

    @property
    def lon_centers(self) -> np.ndarray:
        return self.lon_min + (np.arange(self.nlon) + 0.5) * self.dlon

    @property
    def lat_centers(self) -> np.ndarray:
        return self.lat_min + (np.arange(self.nlat) + 0.5) * self.dlat

    def contains_points(
        self,
        lon: np.ndarray | pd.Series,
        lat: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Return a boolean mask indicating whether points fall inside the grid.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        return (
            (lon >= self.lon_min)
            & (lon < self.lon_max)
            & (lat >= self.lat_min)
            & (lat < self.lat_max)
        )

    def assign_bins(
        self,
        df: pd.DataFrame,
        *,
        lon_col: str,
        lat_col: str,
        drop_outside: bool = True,
    ) -> pd.DataFrame:
        """
        Assign each row to a grid cell.

        Adds:
        - lon_bin
        - lat_bin
        - lon_center
        - lat_center
        """
        required = [lon_col, lat_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Input dataframe missing required columns: {missing}")

        out = df.copy()

        lon = out[lon_col].to_numpy()
        lat = out[lat_col].to_numpy()

        lon_bin = np.floor((lon - self.lon_min) / self.dlon).astype(float)
        lat_bin = np.floor((lat - self.lat_min) / self.dlat).astype(float)

        valid = self.contains_points(lon, lat)

        out["lon_bin"] = lon_bin
        out["lat_bin"] = lat_bin

        if drop_outside:
            out = out.loc[valid].copy()

        if out.empty:
            out["lon_bin"] = out["lon_bin"].astype(int)
            out["lat_bin"] = out["lat_bin"].astype(int)
            out["lon_center"] = pd.Series(dtype=float)
            out["lat_center"] = pd.Series(dtype=float)
            return out

        out["lon_bin"] = out["lon_bin"].astype(int)
        out["lat_bin"] = out["lat_bin"].astype(int)

        out["lon_center"] = self.lon_min + (out["lon_bin"] + 0.5) * self.dlon
        out["lat_center"] = self.lat_min + (out["lat_bin"] + 0.5) * self.dlat

        return out

    def aggregate(
        self,
        df: pd.DataFrame,
        *,
        value_col: str,
        agg: str,
        lon_col: str,
        lat_col: str,
        drop_outside: bool = True,
        output_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate a value on the regular grid.

        Returns one row per occupied grid cell.
        """
        if value_col not in df.columns:
            raise KeyError(f"Input dataframe missing required column: '{value_col}'")

        agg = agg.lower().strip()
        if agg not in _SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation '{agg}'. "
                f"Supported: {sorted(_SUPPORTED_AGGREGATIONS)}"
            )

        binned = self.assign_bins(
            df,
            lon_col=lon_col,
            lat_col=lat_col,
            drop_outside=drop_outside,
        )

        grouped = (
            binned.groupby(
                ["lon_bin", "lat_bin", "lon_center", "lat_center"],
                sort=True,
                observed=False,
            )[value_col]
            .agg(agg)
            .reset_index()
        )

        if output_col is None:
            output_col = f"{value_col}_{agg}"

        grouped = grouped.rename(columns={value_col: output_col})

        return grouped

    def to_xarray(
        self,
        grid_df: pd.DataFrame,
        *,
        value_col: str,
        lon_bin_col: str = "lon_bin",
        lat_bin_col: str = "lat_bin",
        dataset_name: str | None = None,
        fill_value: float = np.nan,
    ) -> xr.Dataset:
        """
        Convert an aggregated grid table to a regular xarray.Dataset.

        Parameters
        ----------
        grid_df
            Aggregated grid table with at least:
            lon_bin, lat_bin, and value_col.
        value_col
            Name of the value column to place on the grid.
        lon_bin_col
            Name of the longitude bin column.
        lat_bin_col
            Name of the latitude bin column.
        dataset_name
            Name of the output variable in the dataset.
            Defaults to value_col.
        fill_value
            Fill value for empty cells.

        Returns
        -------
        xr.Dataset
            Regular 2D dataset with dimensions:
            lat, lon
        """
        required = [lon_bin_col, lat_bin_col, value_col]
        missing = [c for c in required if c not in grid_df.columns]
        if missing:
            raise KeyError(
                f"Input grid dataframe missing required columns: {missing}"
            )

        data = np.full((self.nlat, self.nlon), fill_value, dtype=float)

        if not grid_df.empty:
            lon_bins = grid_df[lon_bin_col].to_numpy(dtype=int)
            lat_bins = grid_df[lat_bin_col].to_numpy(dtype=int)
            values = grid_df[value_col].to_numpy()

            valid = (
                (lon_bins >= 0)
                & (lon_bins < self.nlon)
                & (lat_bins >= 0)
                & (lat_bins < self.nlat)
            )

            lon_bins = lon_bins[valid]
            lat_bins = lat_bins[valid]
            values = values[valid]

            data[lat_bins, lon_bins] = values

        if dataset_name is None:
            dataset_name = value_col

        ds = xr.Dataset(
            data_vars={
                dataset_name: (("lat", "lon"), data)
            },
            coords={
                "lon": self.lon_centers,
                "lat": self.lat_centers,
            },
            attrs={
                "grid_type": "regular_lonlat",
                "lon_min": self.lon_min,
                "lon_max": self.lon_max,
                "lat_min": self.lat_min,
                "lat_max": self.lat_max,
                "dlon": self.dlon,
                "dlat": self.dlat,
            },
        )

        return ds


def assign_regular_grid_bins(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    dlon: float,
    dlat: float,
    drop_outside: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper around RegularGrid.assign_bins().
    """
    grid = RegularGrid(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        dlon=dlon,
        dlat=dlat,
    )
    return grid.assign_bins(
        df,
        lon_col=lon_col,
        lat_col=lat_col,
        drop_outside=drop_outside,
    )


def aggregate_on_regular_grid(
    df: pd.DataFrame,
    *,
    value_col: str,
    agg: str,
    lon_col: str,
    lat_col: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    dlon: float,
    dlat: float,
    drop_outside: bool = True,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper around RegularGrid.aggregate().
    """
    grid = RegularGrid(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        dlon=dlon,
        dlat=dlat,
    )
    return grid.aggregate(
        df,
        value_col=value_col,
        agg=agg,
        lon_col=lon_col,
        lat_col=lat_col,
        drop_outside=drop_outside,
        output_col=output_col,
    )


def infer_regular_spacing_from_centers(
    values: np.ndarray | pd.Series,
    *,
    round_decimals: int = 6,
) -> float:
    """
    Infer the regular spacing of a 1D grid from center coordinates.

    Parameters
    ----------
    values
        1D array of center coordinates.
    round_decimals
        Number of decimals used to collapse small floating-point differences.

    Returns
    -------
    float
        Inferred grid spacing.

    Raises
    ------
    ValueError
        If spacing cannot be inferred.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("Cannot infer spacing from an empty array.")

    unique_vals = np.unique(values[~np.isnan(values)])
    if unique_vals.size < 2:
        raise ValueError("Need at least two unique points to infer spacing.")

    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 0]

    if diffs.size == 0:
        raise ValueError("Could not infer spacing from repeated coordinates only.")

    diffs_rounded = np.round(diffs, round_decimals)
    positive = diffs_rounded[diffs_rounded > 0]

    if positive.size == 0:
        raise ValueError("Could not infer positive spacing after rounding.")

    spacing = float(np.min(positive))
    return spacing


def _select_release_centers(
    summary_df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    time_col: str | None = "time0",
    primary_group_member: int | None = 1,
) -> pd.DataFrame:
    """
    Select native release centers from a particle summary.

    For grouped releases, only the reference member is used so partner offsets do
    not contaminate grid-spacing inference. If deployment time is available, it is
    included in the duplicate filter so repeated releases at the same grid point
    are handled cleanly.
    """
    required = [lon_col, lat_col]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")

    keep_cols = [lon_col, lat_col]
    if time_col is not None and time_col in summary_df.columns:
        keep_cols = [time_col] + keep_cols
    if "group_member" in summary_df.columns:
        keep_cols.append("group_member")

    centers = summary_df[keep_cols].copy()

    if primary_group_member is not None and "group_member" in centers.columns:
        centers_primary = centers.loc[centers["group_member"] == primary_group_member].copy()
        if not centers_primary.empty:
            centers = centers_primary

    dedup_subset = [lon_col, lat_col]
    if time_col is not None and time_col in centers.columns:
        dedup_subset = [time_col] + dedup_subset

    centers = centers.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    return centers


def build_release_grid_from_summary(
    summary_df: pd.DataFrame,
    *,
    lon_col: str = "lon0",
    lat_col: str = "lat0",
    time_col: str | None = "time0",
    primary_group_member: int | None = 1,
    round_decimals: int = 6,
) -> RegularGrid:
    """
    Build the release grid from particle summary initial positions.

    The initial positions are interpreted as grid-cell centers.
    For grouped releases, only member 1 is used to recover the native center grid.
    """
    centers = _select_release_centers(
        summary_df,
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
        primary_group_member=primary_group_member,
    )

    dlon = infer_regular_spacing_from_centers(
        centers[lon_col].to_numpy(),
        round_decimals=round_decimals,
    )
    dlat = infer_regular_spacing_from_centers(
        centers[lat_col].to_numpy(),
        round_decimals=round_decimals,
    )

    return RegularGrid.from_point_centers(
        centers[lon_col].to_numpy(),
        centers[lat_col].to_numpy(),
        dlon=dlon,
        dlat=dlat,
    )


def build_grid_from_config(
    cfg,
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    time_col: str | None = None,
    round_decimals: int = 6,
) -> RegularGrid:
    """
    Build a RegularGrid from the postprocessing config and an input table.

    If a grid section is provided, that configuration is respected. For
    from_initial_centers mode, only the first deployment time and the reference
    group member are used to align the grid phase safely.
    """
    g = getattr(cfg, "grid", None)

    if g is None:
        return build_release_grid_from_summary(
            df,
            lon_col=lon_col,
            lat_col=lat_col,
            time_col=time_col,
            primary_group_member=1,
            round_decimals=round_decimals,
        )

    if g.mode == "explicit_edges":
        return RegularGrid(
            lon_min=g.lon_min,
            lon_max=g.lon_max,
            lat_min=g.lat_min,
            lat_max=g.lat_max,
            dlon=g.dlon,
            dlat=g.dlat,
        )

    if g.mode == "from_initial_centers":
        if time_col is not None and time_col in df.columns:
            t0 = df[time_col].min()
            df0 = df.loc[df[time_col] == t0].copy()
        else:
            df0 = df.copy()

        if "group_member" in df0.columns:
            df0_primary = df0.loc[df0["group_member"] == 1].copy()
            if not df0_primary.empty:
                df0 = df0_primary

        if df0.empty:
            raise ValueError("Cannot build grid from initial centers: no valid release points found.")

        df0 = df0.drop_duplicates(subset=[lon_col, lat_col]).reset_index(drop=True)

        return RegularGrid.from_aligned_initial_centers(
            df0[lon_col],
            df0[lat_col],
            lon_min=g.lon_min,
            lon_max=g.lon_max,
            lat_min=g.lat_min,
            lat_max=g.lat_max,
            dlon=g.dlon,
            dlat=g.dlat,
        )

    raise ValueError(f"Unsupported grid mode: {g.mode}")