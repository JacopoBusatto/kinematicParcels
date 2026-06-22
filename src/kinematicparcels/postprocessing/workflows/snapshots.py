from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..plotting import plot_grid_map
from ..plotting.masking import mask_values_below


def resolve_snapshot_indices(
    timestep_snaps: int | tuple[int, ...] | None,
    *,
    n_times: int,
    config_name: str,
) -> tuple[int, ...]:
    if timestep_snaps is None:
        raise ValueError(f"{config_name}.timestep_snaps is required when plot_snaps is true.")
    if n_times < 1:
        raise ValueError("Cannot plot snapshots from a dataset with no time steps.")

    raw_indices = (timestep_snaps,) if isinstance(timestep_snaps, int) else tuple(timestep_snaps)
    resolved: list[int] = []

    for raw_idx in raw_indices:
        if not isinstance(raw_idx, int):
            raise ValueError(f"{config_name}.timestep_snaps must contain only integers.")
        idx = raw_idx if raw_idx >= 0 else n_times + raw_idx
        if idx < 0 or idx >= n_times:
            raise IndexError(
                f"{config_name} snapshot index {raw_idx} is out of range for {n_times} time steps."
            )
        if idx not in resolved:
            resolved.append(idx)

    return tuple(resolved)


def format_time_for_filename(value) -> str:
    return pd.Timestamp(value).strftime("%Y%m%dT%H%M%S")


def save_gridded_snapshots(
    ds,
    *,
    var_name: str,
    timestep_snaps: int | tuple[int, ...] | None,
    config_name: str,
    outdir: Path,
    filename_prefix: str,
    title_prefix: str,
    projection: str,
    vmin: float | None,
    vmax: float | None,
    min_mask_value: float | None = None,
    cmap: str | None = None,
    colorbar_label: str | None = None,
    title_fontsize: int | None = None,
    colorbar_fontsize: int | None = None,
    colorbar_tick_fontsize: int | None = None,
    axis_tick_fontsize: int | None = None,
) -> None:
    snapshot_indices = resolve_snapshot_indices(
        timestep_snaps,
        n_times=int(ds.sizes.get("time", 0)),
        config_name=config_name,
    )

    for idx in snapshot_indices:
        time_value = ds["time"].values[idx]
        timestamp = format_time_for_filename(time_value)
        plot_path = outdir / f"{filename_prefix}_timestep_{timestamp}.png"
        frame_ds = ds.isel(time=idx)
        if min_mask_value is not None:
            frame_ds = frame_ds.copy()
            frame_ds[var_name] = mask_values_below(frame_ds[var_name], min_mask_value)
        print(f"Saving {title_prefix.lower()} snapshot:", plot_path)
        plot_grid_map(
            frame_ds,
            var_name=var_name,
            outpath=plot_path,
            projection=projection,
            title=f"{title_prefix} {timestamp}",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar_label=colorbar_label or var_name,
            title_fontsize=title_fontsize,
            colorbar_fontsize=colorbar_fontsize,
            colorbar_tick_fontsize=colorbar_tick_fontsize,
            axis_tick_fontsize=axis_tick_fontsize,
        )
