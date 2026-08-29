from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.patches import Ellipse
from matplotlib.path import Path as MatplotlibPath
import numpy as np
import pandas as pd

from .config import GridConfig, PlottingConfig


DATA_CRS = ccrs.PlateCarree()


def map_projection(config: PlottingConfig):
    constructors = {
        "PlateCarree": lambda: ccrs.PlateCarree(central_longitude=config.central_longitude),
        "SouthPolarStereo": lambda: ccrs.SouthPolarStereo(central_longitude=config.central_longitude),
        "NorthPolarStereo": lambda: ccrs.NorthPolarStereo(central_longitude=config.central_longitude),
        "Robinson": lambda: ccrs.Robinson(central_longitude=config.central_longitude),
        "Mercator": lambda: ccrs.Mercator(central_longitude=config.central_longitude),
    }
    return constructors[config.projection]()


def _save(fig, path: Path, config: PlottingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=config.dpi); plt.close(fig)


def _map_axes(config: PlottingConfig, grid: GridConfig, *, title: str):
    polar = config.projection in {"SouthPolarStereo", "NorthPolarStereo"}
    fig, ax = plt.subplots(figsize=(8, 8) if polar else (12, 5),
                           subplot_kw={"projection": map_projection(config)})
    ax.set_extent([grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max], crs=DATA_CRS)
    if config.circular_boundary:
        theta = np.linspace(0, 2 * np.pi, 200)
        circle = np.vstack([np.sin(theta), np.cos(theta)]).T * .5 + .5
        ax.set_boundary(MatplotlibPath(circle), transform=ax.transAxes)
    if config.draw_coastlines:
        ax.coastlines(linewidth=.5, color="0.25")
    ax.gridlines(linewidth=.35, color="0.5", alpha=.45, linestyle=":")
    ax.set_title(title)
    return fig, ax


def _grid_values(frame: pd.DataFrame, value: str, grid: GridConfig) -> np.ndarray:
    values = np.full((grid.nlat, grid.nlon), np.nan)
    if {"start_lon_bin", "start_lat_bin"} <= set(frame):
        lon_bin = frame.start_lon_bin.to_numpy(int); lat_bin = frame.start_lat_bin.to_numpy(int)
    else:
        lon_bin = np.floor((frame.lon.to_numpy(float) - grid.lon_min) / grid.dlon).astype(int)
        lat_bin = np.floor((frame.lat.to_numpy(float) - grid.lat_min) / grid.dlat).astype(int)
    valid = (lon_bin >= 0) & (lon_bin < grid.nlon) & (lat_bin >= 0) & (lat_bin < grid.nlat)
    values[lat_bin[valid], lon_bin[valid]] = frame[value].to_numpy(float)[valid]
    return values


def _grid_map(frame, value, title, path, grid, config, cmap="viridis", norm=None):
    fig, ax = _map_axes(config, grid, title=title)
    lon_edges = grid.lon_min + np.arange(grid.nlon + 1) * grid.dlon
    lat_edges = grid.lat_min + np.arange(grid.nlat + 1) * grid.dlat
    values = np.ma.masked_invalid(_grid_values(frame, value, grid))
    plot = ax.pcolormesh(lon_edges, lat_edges, values, transform=DATA_CRS,
                         shading="flat", cmap=cmap, norm=norm)
    fig.colorbar(plot, ax=ax, label=value, shrink=.75 if config.circular_boundary else 1)
    _save(fig, path, config)


def _scatter_map(frame, value, title, path, grid, config, cmap="viridis"):
    fig, ax = _map_axes(config, grid, title=title)
    values = frame[value].to_numpy(float)
    finite = np.isfinite(values)
    plot = ax.scatter(frame.lon.to_numpy(float)[finite], frame.lat.to_numpy(float)[finite],
                      c=values[finite], s=4, cmap=cmap, transform=DATA_CRS)
    fig.colorbar(plot, ax=ax, label=value, shrink=.75 if config.circular_boundary else 1)
    _save(fig, path, config)


def _line_map(points, title, path, grid, config):
    fig, ax = _map_axes(config, grid, title=title)
    for _, branch in points.groupby("branch_id", sort=True):
        branch = branch.sort_values("point_order")
        lon = branch.lon.to_numpy(float); lat = branch.lat.to_numpy(float)
        jumps = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1
        for part in np.split(np.arange(len(branch)), jumps):
            if len(part) > 1:
                ax.plot(lon[part], lat[part], linewidth=.8, alpha=.8, transform=DATA_CRS)
    _save(fig, path, config)


def _empty_figure(title, path, config, message="No qualifying data in this run"):
    fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
    ax.set_title(title); ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
    _save(fig, path, config)


def produce_figures(
    figures: Path, cells: pd.DataFrame, transitions: pd.DataFrame, modes: pd.DataFrame,
    edges_all: pd.DataFrame, edges_selected: pd.DataFrame, branches: pd.DataFrame,
    branch_summary: pd.DataFrame, cross: pd.DataFrame, candidates: pd.DataFrame,
    barrier_points: pd.DataFrame, grid: GridConfig, config: PlottingConfig,
) -> None:
    if not config.enabled:
        return
    positive_support = cells.loc[cells.N_i > 0, "N_i"]
    support_norm = LogNorm(vmin=max(1, float(positive_support.min())), vmax=float(positive_support.max()))
    _grid_map(cells, "N_i", "Transition support", figures / "01_transition_support.png",
              grid, config, norm=support_norm)
    _grid_map(cells, "P_stay", "Stay probability", figures / "02_stay_probability.png", grid, config)
    _grid_map(cells, "P_move", "Move probability", figures / "02_move_probability.png", grid, config)

    fig, ax = _map_axes(config, grid, title="Mean transition vectors")
    sample = cells.loc[cells.N_i >= cells.N_i.quantile(.5)].dropna(subset=["mean_dx_km", "mean_dy_km"])
    vector = ax.quiver(sample.lon, sample.lat, sample.mean_dx_km, sample.mean_dy_km,
                       sample.N_i, scale=5000, width=.0015, cmap="viridis",
                       transform=DATA_CRS, regrid_shape=35)
    fig.colorbar(vector, ax=ax, label="N_i", shrink=.75 if config.circular_boundary else 1)
    _save(fig, figures / "03_mean_transition_vectors.png", config)

    fig, ax = _map_axes(config, grid, title="Conditional transition variance ellipses")
    ellipse_cells = cells.loc[cells.N_i >= cells.N_i.quantile(.9)].dropna(
        subset=["ellipse_major_scale_km", "ellipse_minor_scale_km"])
    for row in ellipse_cells.itertuples():
        km_per_lon = max(5.0, 111.0 * np.cos(np.deg2rad(row.lat)))
        ellipse = Ellipse(
            (row.lon, row.lat), width=2 * row.ellipse_major_scale_km / km_per_lon,
            height=2 * row.ellipse_minor_scale_km / 111.0,
            angle=90 - row.ellipse_angle_deg, fill=False, edgecolor="tab:blue",
            alpha=.35, linewidth=.5, transform=DATA_CRS,
        )
        ax.add_patch(ellipse)
    ax.scatter(ellipse_cells.lon, ellipse_cells.lat, s=2, c=ellipse_cells.N_i,
               cmap="viridis", transform=DATA_CRS)
    _save(fig, figures / "04_variance_ellipses.png", config)

    mode_summary = modes.groupby("start_cell_id").agg(
        number_of_modes=("mode_id", "size"), dominant=("mode_probability_moving", "max")
    ).reset_index() if len(modes) else pd.DataFrame()
    if "number_of_modes" in cells:
        mode_cells = cells.copy()
        if "dominant" not in mode_cells:
            mode_cells["dominant"] = mode_cells.get("dominant_mode_probability", 0)
    else:
        mode_cells = cells.merge(mode_summary, on="start_cell_id", how="left")
    mode_cells = mode_cells.fillna({"number_of_modes": 0, "dominant": 0})
    max_modes = max(1, int(mode_cells.number_of_modes.max()))
    mode_norm = BoundaryNorm(np.arange(-.5, max_modes + 1.5, 1), plt.get_cmap("viridis").N)
    _grid_map(mode_cells, "number_of_modes", "Number of local modes",
              figures / "05_number_local_modes.png", grid, config, norm=mode_norm)
    _grid_map(mode_cells, "dominant", "Dominant modal probability",
              figures / "06_dominant_modal_probability.png", grid, config)

    if len(modes):
        fig, ax = _map_axes(config, grid, title="Local mode vectors")
        vector = ax.quiver(modes.start_lon, modes.start_lat, modes.modal_dx_km, modes.modal_dy_km,
                           modes.mode_probability_moving, scale=4000, width=.0015, cmap="plasma",
                           transform=DATA_CRS, regrid_shape=40)
        fig.colorbar(vector, ax=ax, label="moving modal probability",
                     shrink=.75 if config.circular_boundary else 1)
        _save(fig, figures / "07_local_mode_vectors.png", config)
        counts = modes.groupby("start_cell_id").size(); example_cells = []
        for wanted in (1, 2):
            candidate_cells = counts[counts.ge(wanted) if wanted == 2 else counts.eq(1)].index
            if len(candidate_cells):
                example_cells.append(int(cells.set_index("start_cell_id").loc[candidate_cells, "N_i"].idxmax()))
        for cell_id in example_cells[:config.max_example_cells]:
            data = transitions.loc[(transitions.start_cell_id == cell_id) & ~transitions.is_stay]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(data.bearing_deg, bins=36, range=(0, 360), weights=data.conditional_moving_probability)
            ax.set(title=f"Outgoing angular PDF cell {cell_id}", xlabel="Bearing [deg]", ylabel="Probability")
            _save(fig, figures / f"12_outgoing_pdf_cell_{cell_id}.png", config)

    if len(edges_all) and len(modes):
        lookup = modes.set_index("mode_id")
        edge_sets = ((edges_all.loc[edges_all.target_node.notna()], "08_graph_edges_all.png", "All graph edges"),
                     (edges_selected, "08_graph_edges_selected.png", "Selected graph edges"))
        for data, filename, title in edge_sets:
            fig, ax = _map_axes(config, grid, title=title)
            for edge in data.itertuples():
                a, b = lookup.loc[edge.source_node], lookup.loc[edge.target_node]
                if abs(a.start_lon - b.start_lon) < 180:
                    ax.plot([a.start_lon, b.start_lon], [a.start_lat, b.start_lat],
                            color="k", alpha=.08, linewidth=.4, transform=DATA_CRS)
            _save(fig, figures / filename, config)

    if len(branches):
        _line_map(branches, "Branch network", figures / "09_branch_network.png", grid, config)
        _scatter_map(branches, "radius_curvature_km", "Branch curvature radius",
                     figures / "11_branch_curvature.png", grid, config)
        _scatter_map(branches, "local_support_count", "Branch transition-count support",
                     figures / "10_branch_support.png", grid, config)

    if len(cross):
        major = set(branch_summary.loc[branch_summary.major_branch, "branch_id"]) if len(branch_summary) else set()
        for branch_id, data in cross.loc[cross.branch_id.isin(major)].groupby("branch_id"):
            pivot = data.pivot(index="offset_km", columns="s_km", values="P_cross")
            fig, ax = plt.subplots(figsize=(12, 4))
            finite = pivot.values[np.isfinite(pivot.values)]
            vmax = np.percentile(finite, 95) if len(finite) else 1
            image = ax.imshow(pivot.values, origin="lower", aspect="auto",
                              extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                              vmin=0, vmax=vmax)
            fig.colorbar(image, ax=ax, label="P_cross")
            ax.set(title=f"Permeability {branch_id}", xlabel="s [km]", ylabel="offset [km]")
            _save(fig, figures / f"13_permeability_heatmap_{branch_id}.png", config)
        examples = cross.groupby("branch_point_id").filter(
            lambda values: values.support_valid.sum() >= 5).branch_point_id.drop_duplicates().head(4)
        for point_id in examples:
            data = cross.loc[cross.branch_point_id.eq(point_id)].sort_values("offset_km")
            fig, ax = plt.subplots(); ax.plot(data.offset_km, data.P_cross, label="all")
            ax.plot(data.offset_km, data.P_cross_moving, label="moving"); ax.legend()
            ax.set(title=point_id, xlabel="offset [km]", ylabel="probability")
            _save(fig, figures / f"14_cross_section_{point_id.replace(':', '_')}.png", config)

    if len(candidates):
        fig, ax = _map_axes(config, grid, title="Accepted and rejected barrier minima")
        ax.scatter(candidates.lon, candidates.lat,
                   c=np.where(candidates.accepted, "tab:red", "0.6"), s=8, transform=DATA_CRS)
        _save(fig, figures / "15_barrier_candidates.png", config)

    if len(barrier_points):
        robust = barrier_points.loc[barrier_points.robust_segment]
        if len(robust):
            fig, ax = _map_axes(config, grid, title="Final barriers over branches")
            for _, branch in branches.groupby("branch_id"):
                ax.plot(branch.lon, branch.lat, color="0.7", linewidth=.5, transform=DATA_CRS)
            for _, barrier in robust.groupby(["barrier_id", "geometry_part"]):
                ax.plot(barrier.lon, barrier.lat, linewidth=1.5, transform=DATA_CRS)
            _save(fig, figures / "16_final_barriers.png", config)
            for variable, filename, title in (
                ("P_cross", "17_barrier_permeability.png", "Barrier permeability"),
                ("directional_asymmetry", "20_barrier_asymmetry.png", "Directional asymmetry"),
                ("total_support_count", "21_barrier_support.png", "Observational support"),
            ):
                fig, ax = plt.subplots(figsize=(10, 4))
                for barrier_id, data in robust.groupby("barrier_id"):
                    ax.plot(data.along_barrier_km, data[variable], label=barrier_id)
                ax.set(title=title, xlabel="along-barrier distance [km]", ylabel=variable)
                _save(fig, figures / filename, config)
            fig, ax = plt.subplots(figsize=(10, 4))
            for barrier_id, data in robust.groupby("barrier_id"):
                ax.plot(data.along_barrier_km, data.P_minus_to_plus, label=f"{barrier_id} - to +")
                ax.plot(data.along_barrier_km, data.P_plus_to_minus, linestyle="--", label=f"{barrier_id} + to -")
            ax.set(title="Directional barrier permeability", xlabel="along-barrier distance [km]", ylabel="probability")
            _save(fig, figures / "18_directional_permeability.png", config)
            fig, ax = plt.subplots(figsize=(10, 4))
            for barrier_id, data in robust.groupby("barrier_id"):
                ax.plot(data.along_barrier_km, data.P_cross_moving, label=barrier_id)
            ax.set(title="Moving-conditional permeability", xlabel="along-barrier distance [km]", ylabel="probability")
            _save(fig, figures / "19_moving_conditional_permeability.png", config)
        else:
            for filename, title in (
                ("16_final_barriers.png", "Final barriers over branches"),
                ("17_barrier_permeability.png", "Barrier permeability"),
                ("18_directional_permeability.png", "Directional barrier permeability"),
                ("19_moving_conditional_permeability.png", "Moving-conditional permeability"),
                ("20_barrier_asymmetry.png", "Directional asymmetry"),
                ("21_barrier_support.png", "Observational support"),
            ):
                _empty_figure(title, figures / filename, config,
                              "No robust barrier segment passed the baseline gates")
        fig, ax = _map_axes(config, grid, title="Barrier geometry and quality flags")
        flagged = barrier_points.quality_flags.fillna("").ne("") if "quality_flags" in barrier_points else np.zeros(len(barrier_points), bool)
        ax.scatter(barrier_points.lon, barrier_points.lat,
                   c=np.where(flagged, "tab:orange", "tab:blue"), s=8, transform=DATA_CRS)
        _save(fig, figures / "22_barrier_quality_flags.png", config)
