from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_SLOPES = {
    "delta^-2/3": (-2.0 / 3.0, r"$\delta^{-2/3}$"),
    "delta^-1": (-1.0, r"$\delta^{-1}$"),
    "delta^-2": (-2.0, r"$\delta^{-2}$"),
}


def _reference_line(
    x: np.ndarray,
    *,
    x_anchor: float,
    y_anchor: float,
    power: float,
) -> np.ndarray:
    return y_anchor * (x / x_anchor) ** power


def _build_reference_lines(
    spectrum: pd.DataFrame,
    *,
    reference_slopes: tuple[str, ...] | list[str],
    reference_slope_anchor_scales: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    if spectrum.empty:
        return []

    anchor_scales = reference_slope_anchor_scales or {}
    df = spectrum.sort_values("scale").copy()
    x = df["scale"].to_numpy(dtype=float)
    y = df["fsle"].to_numpy(dtype=float)
    default_anchor_idx = len(df) // 2

    reference_lines: list[dict[str, object]] = []
    for slope_id in reference_slopes:
        if slope_id not in REFERENCE_SLOPES:
            raise ValueError(
                f"Unsupported reference slope '{slope_id}'. Supported: {sorted(REFERENCE_SLOPES)}"
            )

        power, label = REFERENCE_SLOPES[slope_id]
        requested_anchor = anchor_scales.get(slope_id)
        if requested_anchor is None:
            anchor_idx = default_anchor_idx
        else:
            anchor_idx = int(np.abs(x - requested_anchor).argmin())

        x_anchor = float(x[anchor_idx])
        y_anchor = float(y[anchor_idx])
        reference_lines.append(
            {
                "slope_id": slope_id,
                "power": power,
                "label": label,
                "x_anchor": x_anchor,
                "y_anchor": y_anchor,
                "y": _reference_line(x, x_anchor=x_anchor, y_anchor=y_anchor, power=power),
            }
        )

    return reference_lines


def plot_fsle_spectrum(
    spectrum: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "FSLE spectrum",
    reference_slopes: tuple[str, ...] | list[str] = ("delta^-2/3", "delta^-1", "delta^-2"),
    reference_slope_anchor_scales: dict[str, float] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> Path:
    if spectrum.empty:
        raise ValueError("Cannot plot an empty FSLE spectrum.")

    df = spectrum.sort_values("scale").copy()
    x = df["scale"].to_numpy(dtype=float)
    y = df["fsle"].to_numpy(dtype=float)
    yerr = df["std"].to_numpy(dtype=float) if "std" in df.columns else None
    reference_lines = _build_reference_lines(
        df,
        reference_slopes=reference_slopes,
        reference_slope_anchor_scales=reference_slope_anchor_scales,
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(x, y, color="C0", linewidth=1.25, alpha=0.8)
    ax.scatter(x, y, s=45, facecolors="none", edgecolors="C0", linewidths=1.2, label="FSLE")

    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", color="C0", capsize=3, alpha=0.8)

    for reference_line in reference_lines:
        ax.plot(
            x,
            reference_line["y"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=reference_line["label"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta$ [km]")
    ax.set_ylabel(r"$\lambda(\delta)$ [day$^{-1}$]")
    ax.set_title(title)
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", linestyle="--", alpha=0.2)
    ax.legend()

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath