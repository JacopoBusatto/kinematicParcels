from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


_TARGET_LINESTYLES = ["--", ":", "-.", "-"]


def _resolve_colormap_name(region_labels: list[str], colormap: str | None) -> str:
    if colormap is not None:
        return colormap

    if len(region_labels) <= 10:
        return "Set2"
    if len(region_labels) <= 20:
        return "tab20"
    return "hsv"


def _build_palette(
    region_labels: list[str],
    *,
    colormap: str | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap(_resolve_colormap_name(region_labels, colormap))

    denom = max(len(region_labels) - 1, 1)
    return {
        label: cmap((idx / denom) if len(region_labels) > 1 else 0.0)
        for idx, label in enumerate(region_labels)
    }


def _target_style_map(region_labels: list[str]) -> dict[str, str]:
    return {
        label: _TARGET_LINESTYLES[idx % len(_TARGET_LINESTYLES)]
        for idx, label in enumerate(region_labels)
    }


def _apply_axis_scales(
    ax,
    *,
    x_log_scale: bool,
    y_log_scale: bool,
) -> None:
    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")


def _mask_for_log_axes(
    ages: pd.Series,
    values: pd.Series,
    *,
    x_log_scale: bool,
    y_log_scale: bool,
) -> tuple[pd.Series, pd.Series]:
    mask = pd.Series(True, index=ages.index)
    if x_log_scale:
        mask &= ages > 0
    if y_log_scale:
        mask &= values > 0
    return ages.loc[mask], values.loc[mask]


def _compute_shared_y_limits(
    transition_table: pd.DataFrame,
    *,
    region_labels: list[str],
    x_log_scale: bool,
    y_log_scale: bool,
) -> tuple[float, float] | None:
    ages = transition_table["age_days"]
    plotted_values: list[np.ndarray] = []

    for origin in region_labels:
        for target in region_labels:
            column_name = f"p_{origin}__{target}"
            if column_name not in transition_table.columns:
                raise KeyError(f"Missing transition probability column '{column_name}'.")

            _, y_vals = _mask_for_log_axes(
                ages,
                transition_table[column_name],
                x_log_scale=x_log_scale,
                y_log_scale=y_log_scale,
            )
            if len(y_vals) > 0:
                plotted_values.append(y_vals.to_numpy(dtype=float))

        represented_origin = _represented_fraction_for_origin(
            transition_table,
            origin=origin,
            region_labels=region_labels,
        )
        _, y_vals = _mask_for_log_axes(
            ages,
            represented_origin,
            x_log_scale=x_log_scale,
            y_log_scale=y_log_scale,
        )
        if len(y_vals) > 0:
            plotted_values.append(y_vals.to_numpy(dtype=float))

    represented_total = _represented_fraction_total(transition_table, region_labels=region_labels)
    if represented_total is not None:
        _, y_vals = _mask_for_log_axes(
            ages,
            represented_total,
            x_log_scale=x_log_scale,
            y_log_scale=y_log_scale,
        )
        if len(y_vals) > 0:
            plotted_values.append(y_vals.to_numpy(dtype=float))

    if len(plotted_values) == 0:
        return None

    values = np.concatenate(plotted_values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    if y_log_scale:
        positive_values = values[values > 0]
        if len(positive_values) == 0:
            return 1.0e-6, 1.0

        ymin = 10.0 ** float(np.floor(np.log10(float(positive_values.min()))))
        ymax = 10.0 ** float(np.ceil(np.log10(float(positive_values.max()))))
        if ymax <= ymin:
            ymax = ymin * 10.0
        return ymin, ymax

    ymax = float(values.max())
    if ymax <= 0:
        ymax = 1.0
    return 0.0, ymax


def _represented_fraction_for_origin(
    transition_table: pd.DataFrame,
    *,
    origin: str,
    region_labels: list[str],
) -> pd.Series:
    columns = [f"p_{origin}__{target}" for target in region_labels]
    missing = [column_name for column_name in columns if column_name not in transition_table.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing transition probability columns for origin '{origin}': {missing_str}")
    return transition_table[columns].sum(axis=1, min_count=1)


def _represented_fraction_total(
    transition_table: pd.DataFrame,
    *,
    region_labels: list[str],
) -> pd.Series | None:
    if "represented_fraction_total" in transition_table.columns:
        return transition_table["represented_fraction_total"]

    count_columns = [f"n_{origin}" for origin in region_labels]
    if not all(column_name in transition_table.columns for column_name in count_columns):
        return None

    if transition_table.empty:
        return pd.Series(dtype=float)

    total_count = sum(float(transition_table.iloc[0][column_name]) for column_name in count_columns)
    if total_count <= 0:
        return pd.Series(np.nan, index=transition_table.index, dtype=float)

    weighted_total = pd.Series(0.0, index=transition_table.index, dtype=float)
    for origin in region_labels:
        origin_count = float(transition_table.iloc[0][f"n_{origin}"])
        if origin_count <= 0:
            continue
        weighted_total = weighted_total + origin_count * _represented_fraction_for_origin(
            transition_table,
            origin=origin,
            region_labels=region_labels,
        ).fillna(0.0)
    return weighted_total / total_count


def _add_log_note(ax, *, x_log_scale: bool, y_log_scale: bool) -> None:
    if x_log_scale or y_log_scale:
        ax.text(
            0.0,
            -0.095,
            "Zero and negative values are masked on log axes.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="dimgray",
        )


def _validate_x_limits(
    *,
    x_log_scale: bool,
    x_limit_min: float | None,
    x_limit_max: float | None,
) -> None:
    if x_limit_min is not None and x_limit_max is not None and x_limit_min >= x_limit_max:
        raise ValueError("x_limit_min must be smaller than x_limit_max.")

    if x_log_scale:
        if x_limit_min is not None and x_limit_min <= 0:
            raise ValueError("x_limit_min must be > 0 when x_log_scale is enabled.")
        if x_limit_max is not None and x_limit_max <= 0:
            raise ValueError("x_limit_max must be > 0 when x_log_scale is enabled.")


def _apply_x_limits(
    ax,
    *,
    x_limit_min: float | None,
    x_limit_max: float | None,
) -> None:
    if x_limit_min is None and x_limit_max is None:
        return

    current_min, current_max = ax.get_xlim()
    ax.set_xlim(
        x_limit_min if x_limit_min is not None else current_min,
        x_limit_max if x_limit_max is not None else current_max,
    )


def _save_figure(
    fig,
    outpath: Path,
    *,
    bbox_extra_artists: tuple | None = None,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 150, "bbox_inches": "tight"}
    if bbox_extra_artists is not None:
        save_kwargs["bbox_extra_artists"] = bbox_extra_artists
    fig.savefig(outpath, **save_kwargs)
    plt.close(fig)


def plot_transition_probability_overview(
    transition_table: pd.DataFrame,
    *,
    region_labels: list[str],
    outpath: str | Path,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    colormap: str | None = None,
    x_limit_min: float | None = None,
    x_limit_max: float | None = None,
) -> Path:
    if "age_days" not in transition_table.columns:
        raise KeyError("transition probability table must contain an 'age_days' column.")

    _validate_x_limits(
        x_log_scale=x_log_scale,
        x_limit_min=x_limit_min,
        x_limit_max=x_limit_max,
    )

    ages = transition_table["age_days"]
    start_colors = _build_palette(region_labels, colormap=colormap)
    target_styles = _target_style_map(region_labels)
    shared_y_limits = _compute_shared_y_limits(
        transition_table,
        region_labels=region_labels,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )

    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)

    for origin in region_labels:
        for target in region_labels:
            column_name = f"p_{origin}__{target}"
            if column_name not in transition_table.columns:
                raise KeyError(f"Missing transition probability column '{column_name}'.")

            x_vals, y_vals = _mask_for_log_axes(
                ages,
                transition_table[column_name],
                x_log_scale=x_log_scale,
                y_log_scale=y_log_scale,
            )
            if len(x_vals) == 0:
                continue

            ax.plot(
                x_vals,
                y_vals,
                color=start_colors[origin],
                linestyle=target_styles[target],
                linewidth=2.5,
                alpha=0.95,
            )

    represented_total = _represented_fraction_total(
        transition_table,
        region_labels=region_labels,
    )
    if represented_total is not None:
        x_vals, y_vals = _mask_for_log_axes(
            ages,
            represented_total,
            x_log_scale=x_log_scale,
            y_log_scale=y_log_scale,
        )
        if len(x_vals) > 0:
            ax.plot(
                x_vals,
                y_vals,
                color="black",
                linestyle="-",
                linewidth=1.8,
                alpha=0.95,
            )

    ax.set_title("Transition fractions by source and target region", fontsize=28, pad=22)
    ax.set_xlabel("age (days)", fontsize=22)
    ax.set_ylabel("transition fraction", fontsize=22)
    ax.tick_params(labelsize=16)
    ax.grid(True, which="major", color="#bfc7d5", alpha=0.8, linewidth=0.8)
    ax.grid(True, which="minor", color="#d7dde8", alpha=0.55, linewidth=0.5)
    _apply_axis_scales(ax, x_log_scale=x_log_scale, y_log_scale=y_log_scale)
    _apply_x_limits(
        ax,
        x_limit_min=x_limit_min,
        x_limit_max=x_limit_max,
    )
    if shared_y_limits is not None:
        ax.set_ylim(*shared_y_limits)
    # _add_log_note(ax, x_log_scale=x_log_scale, y_log_scale=y_log_scale)

    start_handles = [
        Line2D([0], [0], color=start_colors[label], linewidth=4, label=label)
        for label in region_labels
    ]
    target_handles = [
        Line2D([0], [0], color="black", linestyle=target_styles[label], linewidth=4, label=label)
        for label in region_labels
    ]
    first_legend = ax.legend(
        handles=start_handles,
        title="Starting region",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=18,
        title_fontsize=20,
        framealpha=0.95,
    )
    ax.add_artist(first_legend)
    second_legend = ax.legend(
        handles=target_handles,
        title="Target region",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
        fontsize=18,
        title_fontsize=20,
        framealpha=0.95,
    )

    # Reserve explicit right margin so outside legends are fully included in the export.
    fig.subplots_adjust(right=0.74)

    outpath = Path(outpath)
    _save_figure(
        fig,
        outpath,
        bbox_extra_artists=(first_legend, second_legend),
    )
    return outpath


def plot_transition_probability_by_source(
    transition_table: pd.DataFrame,
    *,
    region_labels: list[str],
    outdir: str | Path,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    colormap: str | None = None,
    x_limit_min: float | None = None,
    x_limit_max: float | None = None,
) -> list[Path]:
    if transition_table.empty:
        return []

    _validate_x_limits(
        x_log_scale=x_log_scale,
        x_limit_min=x_limit_min,
        x_limit_max=x_limit_max,
    )

    ages = transition_table["age_days"]
    target_colors = _build_palette(region_labels, colormap=colormap)
    target_styles = _target_style_map(region_labels)
    shared_y_limits = _compute_shared_y_limits(
        transition_table,
        region_labels=region_labels,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for origin in region_labels:
        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

        for target in region_labels:
            column_name = f"p_{origin}__{target}"
            if column_name not in transition_table.columns:
                raise KeyError(f"Missing transition probability column '{column_name}'.")

            x_vals, y_vals = _mask_for_log_axes(
                ages,
                transition_table[column_name],
                x_log_scale=x_log_scale,
                y_log_scale=y_log_scale,
            )
            if len(x_vals) == 0:
                continue

            ax.plot(
                x_vals,
                y_vals,
                color=target_colors[target],
                linestyle=target_styles[target],
                linewidth=2.8,
                alpha=0.95,
                label=target,
            )

        represented_origin = _represented_fraction_for_origin(
            transition_table,
            origin=origin,
            region_labels=region_labels,
        )
        x_vals, y_vals = _mask_for_log_axes(
            ages,
            represented_origin,
            x_log_scale=x_log_scale,
            y_log_scale=y_log_scale,
        )
        if len(x_vals) > 0:
            ax.plot(
                x_vals,
                y_vals,
                color="black",
                linestyle="-",
                linewidth=1.8,
                alpha=0.95,
            )

        ax.set_title(f"Transition fractions from source region {origin}", fontsize=28, pad=22)
        ax.set_xlabel("age (days)", fontsize=22)
        ax.set_ylabel("transition fraction", fontsize=22)
        ax.tick_params(labelsize=16)
        ax.grid(True, which="major", color="#bfc7d5", alpha=0.8, linewidth=0.8)
        ax.grid(True, which="minor", color="#d7dde8", alpha=0.55, linewidth=0.5)
        _apply_axis_scales(ax, x_log_scale=x_log_scale, y_log_scale=y_log_scale)
        _apply_x_limits(
            ax,
            x_limit_min=x_limit_min,
            x_limit_max=x_limit_max,
        )
        if shared_y_limits is not None:
            ax.set_ylim(*shared_y_limits)
        # _add_log_note(ax, x_log_scale=x_log_scale, y_log_scale=y_log_scale)
        ax.legend(
            title="Target region",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=18,
            title_fontsize=20,
            framealpha=0.95,
        )

        outpath = outdir / f"transition_probability_{origin}_plot.png"
        _save_figure(fig, outpath)
        outputs.append(outpath)

    return outputs
