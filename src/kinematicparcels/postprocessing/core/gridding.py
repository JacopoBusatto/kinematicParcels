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

        valid = (
            (lon >= self.lon_min)
            & (lon < self.lon_max)
            & (lat >= self.lat_min)
            & (lat < self.lat_max)
        )

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