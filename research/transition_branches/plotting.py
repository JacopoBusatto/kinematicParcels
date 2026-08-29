"""Five publication-ready maps for the production current/front workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MatplotlibPath

from .config import CompactConfig
from .cores import CoreSolution
from .fronts import FrontSolution
from .validation import ValidationSolution

CORE_MARKER_STYLES = {
    "two_sided": ("o", "tab:blue", "Current core (two-sided observed)"),
    "one_sided": ("^", "tab:orange", "Current core (one-sided observed)"),
}
FRONT_MARKER_STYLES = {
    "left": (">", "cyan", "Left transport front"),
    "right": ("<", "magenta", "Right transport front"),
}


def _finite_percentile_max(values, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    maximum = float(np.percentile(finite, percentile))
    return maximum if maximum > 0 else 1.0


def _projection(config):
    import cartopy.crs as ccrs

    data = ccrs.PlateCarree()
    projection = (
        ccrs.SouthPolarStereo(central_longitude=config.central_longitude)
        if config.projection == "SouthPolarStereo"
        else data
    )
    return projection, data


def create_standard_figures(
    cells,
    cores: CoreSolution,
    fronts: FrontSolution,
    config: CompactConfig,
    output_dir: Path,
    *,
    validation: ValidationSolution | None = None,
) -> list[Path]:
    """Create the five standard maps and an optional labeled validation map."""
    if not config.plotting.enabled:
        return []
    projection, data_crs = _projection(config.plotting)
    output_dir.mkdir(parents=True, exist_ok=True)
    lon_edges = np.linspace(
        config.grid.lon_min, config.grid.lon_max, config.grid.nlon + 1
    )
    lat_edges = np.linspace(
        config.grid.lat_min, config.grid.lat_max, config.grid.nlat + 1
    )
    support = cells.N_out_move.ge(config.statistics.min_moving_support)
    created: list[Path] = []

    def grid_array(field):
        values = np.full((config.grid.nlat, config.grid.nlon), np.nan)
        rows = cells.loc[support, ["lat_bin", "lon_bin", field]]
        values[rows.lat_bin.to_numpy(int), rows.lon_bin.to_numpy(int)] = rows[field]
        return values

    def base_axis():
        figure, axis = plt.subplots(
            figsize=(9, 8), subplot_kw={"projection": projection}
        )
        axis.set_extent(
            [
                config.grid.lon_min,
                config.grid.lon_max,
                config.grid.lat_min,
                config.grid.lat_max,
            ],
            crs=data_crs,
        )
        if config.plotting.draw_coastlines:
            axis.coastlines(linewidth=0.5)
        if (
            config.plotting.circular_boundary
            and config.plotting.projection == "SouthPolarStereo"
        ):
            angles = np.linspace(0, 2 * np.pi, 128)
            boundary = np.column_stack(
                [0.5 + 0.5 * np.sin(angles), 0.5 + 0.5 * np.cos(angles)]
            )
            axis.set_boundary(MatplotlibPath(boundary), transform=axis.transAxes)
        axis.gridlines(linewidth=0.3, alpha=0.35)
        return figure, axis

    def add_transport_background(axis, *, cmap="viridis", max_percentile=99.0):
        values = grid_array("U_out_all_magnitude_km_day")
        return axis.pcolormesh(
            lon_edges,
            lat_edges,
            values,
            transform=data_crs,
            shading="flat",
            cmap=cmap,
            vmin=0,
            vmax=_finite_percentile_max(values, max_percentile),
        )

    def add_transport_vectors(axis):
        stride = config.plotting.vector_stride_cells
        arrows = cells.loc[
            support
            & cells.lon_bin.mod(stride).eq(0)
            & cells.lat_bin.mod(stride).eq(0)
            & cells.U_out_all_east_km_day.notna()
            & cells.U_out_all_north_km_day.notna()
        ]
        quiver = axis.quiver(
            arrows.lon,
            arrows.lat,
            arrows.U_out_all_east_km_day,
            arrows.U_out_all_north_km_day,
            transform=data_crs,
            color="black",
            width=0.0022,
            headwidth=3.5,
            zorder=3,
        )
        axis.quiverkey(
            quiver,
            0.80,
            0.035,
            config.plotting.vector_reference_km_day,
            f"{config.plotting.vector_reference_km_day:g} km day$^{{-1}}$ transport vector",
            labelpos="N",
            coordinates="axes",
        )

    def save(figure, filename):
        path = output_dir / filename
        figure.savefig(path, dpi=config.plotting.dpi, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    figure, axis = base_axis()
    mesh = add_transport_background(axis)
    add_transport_vectors(axis)
    figure.colorbar(mesh, ax=axis, shrink=0.7, label="|U_out,all| [km day$^{-1}$]")
    axis.set_title("Lagrangian transport field")
    save(figure, "01_transport_vectors.png")

    scalar_maps = (
        (
            "R1_out",
            "02_R1.png",
            "Outgoing first angular harmonic $R_1$",
            "R1",
            "viridis",
        ),
        (
            "R2_out",
            "03_R2.png",
            "Outgoing second angular harmonic $R_2$",
            "R2",
            "viridis",
        ),
        (
            "angular_entropy_out",
            "04_angular_entropy.png",
            "Normalized outgoing angular entropy",
            "Normalized angular entropy",
            "magma",
        ),
    )
    for field, filename, title, label, cmap in scalar_maps:
        figure, axis = base_axis()
        mesh = axis.pcolormesh(
            lon_edges,
            lat_edges,
            grid_array(field),
            transform=data_crs,
            shading="flat",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        figure.colorbar(mesh, ax=axis, shrink=0.7, label=label)
        axis.set_title(title)
        save(figure, filename)

    figure, axis = base_axis()
    mesh = add_transport_background(
        axis,
        cmap="Greys",
        max_percentile=config.plotting.structure_map_max_percentile,
    )
    add_transport_vectors(axis)
    for ridge_type, (marker, color, label) in CORE_MARKER_STYLES.items():
        rows = cores.cores.loc[cores.cores.ridge_type.eq(ridge_type)]
        if not rows.empty:
            axis.scatter(
                rows.lon,
                rows.lat,
                transform=data_crs,
                s=15,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.25,
                label=label,
                zorder=5,
            )
    detected = fronts.fronts.loc[fronts.fronts.front_detected]
    for side, (marker, color, label) in FRONT_MARKER_STYLES.items():
        rows = detected.loc[detected.side.eq(side)]
        if not rows.empty:
            axis.scatter(
                rows.front_lon,
                rows.front_lat,
                transform=data_crs,
                s=18,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                label=label,
                zorder=6,
            )
    axis.legend(loc="lower left", fontsize=7, framealpha=0.9)
    figure.colorbar(mesh, ax=axis, shrink=0.7, label="|U_out,all| [km day$^{-1}$]")
    axis.set_title("Lagrangian current cores and probable transport fronts")
    save(figure, "05_cores_and_fronts.png")

    if validation is not None and config.plotting.debug_plots:
        values = np.full((config.grid.nlat, config.grid.nlon), np.nan)
        frame = validation.global_fields
        values[frame.lat_bin.to_numpy(int), frame.lon_bin.to_numpy(int)] = frame[
            "abs_G_perp"
        ]
        figure, axis = base_axis()
        mesh = axis.pcolormesh(
            lon_edges,
            lat_edges,
            values,
            transform=data_crs,
            shading="flat",
            cmap="magma",
            vmin=0,
            vmax=_finite_percentile_max(
                values, config.plotting.structure_map_max_percentile
            ),
        )
        rows = validation.validation
        axis.scatter(
            rows.flank_lon,
            rows.flank_lat,
            transform=data_crs,
            s=12,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.6,
            label="Probable transport front",
        )
        axis.legend(loc="lower left", fontsize=7, framealpha=0.9)
        figure.colorbar(mesh, ax=axis, shrink=0.7, label="|G_perp| [day$^{-1}$]")
        axis.set_title("Optional cross-stream-gradient validation")
        save(figure, "debug_gradient_validation.png")

    return created
