from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.models import DatasetCoordinatesConfig
from ..core import build_particle_summary
from ..io import load_trajectory_table
from ..plotting import plot_trajectories_map


def quicklook_trajectories(
    input_path: str | Path,
    output_plot: str | Path,
    *,
    coordinates: DatasetCoordinatesConfig = DatasetCoordinatesConfig(),
    title: str = "Trajectories",
    extra_vars: list[str] | None = None,
    sort: bool = True,
    drop_empty: bool = True,
    truncate_at_first_invalid: bool = True,
    show_start: bool = True,
    show_end: bool = True,
    title_fontsize: int | None = None,
    colorbar_fontsize: int | None = None,
    colorbar_tick_fontsize: int | None = None,
    axis_tick_fontsize: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level quicklook workflow for Parcels trajectories.
    """
    df = load_trajectory_table(
        input_path,
        coordinates=coordinates,
        extra_vars=extra_vars,
        sort=sort,
        drop_empty=drop_empty,
        truncate_at_first_invalid=truncate_at_first_invalid,
    )

    summary = build_particle_summary(df)

    plot_trajectories_map(
        df,
        outpath=output_plot,
        title=title,
        show_start=show_start,
        show_end=show_end,
        title_fontsize=title_fontsize,
        colorbar_fontsize=colorbar_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        axis_tick_fontsize=axis_tick_fontsize,
    )

    return df, summary
