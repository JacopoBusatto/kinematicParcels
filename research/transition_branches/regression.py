"""Exact regression of production results against the validated compact run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .cores import compute_current_cores
from .fronts import compute_probable_fronts
from .statistics import compute_transition_statistics

CELL_FIELDS = (
    "P_stay",
    "P_move",
    "U_out_all_east_km_day",
    "U_out_all_north_km_day",
    "U_out_all_magnitude_km_day",
    "theta_mu_out",
    "R1_out",
    "R2_out",
    "angular_entropy_out",
    "R1_in",
    "R2_in",
    "theta1_in_motion_destination",
    "theta_mu_in_motion_destination",
    "delta_theta_mu1_in",
    "delta_theta_io",
)
CORE_FIELDS = (
    "U_out_all_magnitude_km_day",
    "theta_mu_out",
    "R1_out",
    "R2_out",
)
DROP_FIELDS = (
    "candidate_distance_km",
    "candidate_lon",
    "candidate_lat",
    "absolute_drop",
    "relative_drop",
    "drop_slope",
    "drop_width_km",
)
FRONT_FIELDS = (
    "front_lon",
    "front_lat",
    "distance_from_core_km",
    "transport_loss_km_day",
    "relative_transport_loss",
)


def _numeric_comparison(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    keys: list[str],
    fields: tuple[str, ...],
) -> dict:
    merged = old[[*keys, *fields]].merge(
        new[[*keys, *fields]],
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    result = {
        "old_rows": len(old),
        "new_rows": len(new),
        "matched_rows": int(merged._merge.eq("both").sum()),
        "old_only_rows": int(merged._merge.eq("left_only").sum()),
        "new_only_rows": int(merged._merge.eq("right_only").sum()),
        "fields": {},
    }
    matched = merged.loc[merged._merge.eq("both")]
    for field in fields:
        left = matched[f"{field}_old"].to_numpy(float)
        right = matched[f"{field}_new"].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        result["fields"][field] = {
            "max_absolute_difference": (
                float(np.max(np.abs(left[finite] - right[finite])))
                if finite.any()
                else None
            ),
            "nan_pattern_mismatches": int(
                np.count_nonzero(np.isnan(left) != np.isnan(right))
            ),
        }
    return result


def _rank_repeated_drops(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["cell_id", "side", "candidate_distance_km", "absolute_drop", "relative_drop"],
        kind="stable",
    ).copy()
    ordered["candidate_ordinal"] = ordered.groupby(
        ["cell_id", "side"], sort=False
    ).cumcount()
    return ordered


def _exact_numeric(section: dict) -> bool:
    return (
        section["old_only_rows"] == 0
        and section["new_only_rows"] == 0
        and all(
            field["max_absolute_difference"] in (None, 0.0)
            and field["nan_pattern_mismatches"] == 0
            for field in section["fields"].values()
        )
    )


def compare(reference_run: Path, production_run: Path) -> dict:
    old_cells = pd.read_parquet(reference_run / "cell_statistics.parquet")
    new_cells = pd.read_parquet(production_run / "cell_statistics.parquet")
    cell_result = _numeric_comparison(
        old_cells, new_cells, keys=["cell_id"], fields=CELL_FIELDS
    )

    old_cores = pd.read_parquet(reference_run / "branch_cores.parquet").rename(
        columns={"transport_km_day": "U_out_all_magnitude_km_day"}
    )
    new_cores = pd.read_parquet(production_run / "branch_cores.parquet")
    core_result = _numeric_comparison(
        old_cores, new_cores, keys=["cell_id"], fields=CORE_FIELDS
    )
    core_classes = old_cores[
        ["cell_id", "ridge_type", "missing_side", "component_id"]
    ].merge(
        new_cores[["cell_id", "ridge_type", "missing_side", "component_id"]],
        on="cell_id",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    core_result["classification_mismatches"] = int(
        (
            core_classes.ridge_type_old.ne(core_classes.ridge_type_new)
            | core_classes.missing_side_old.ne(core_classes.missing_side_new)
            | core_classes.component_id_old.ne(core_classes.component_id_new)
        ).sum()
    )

    config = load_config(production_run / "resolved_config.yaml")
    matrix = pd.read_parquet(config.input.transition_table)
    statistics = compute_transition_statistics(matrix, config)
    computed_cores = compute_current_cores(statistics.cells, config)
    computed_fronts = compute_probable_fronts(statistics.cells, computed_cores, config)

    old_drops = pd.read_parquet(reference_run / "candidate_drops.parquet")
    new_drops = computed_fronts.candidate_drops.loc[
        computed_fronts.candidate_drops.candidate_eligible
    ]
    drop_result = _numeric_comparison(
        _rank_repeated_drops(old_drops),
        _rank_repeated_drops(new_drops),
        keys=["cell_id", "side", "candidate_ordinal"],
        fields=DROP_FIELDS,
    )

    old_fronts = pd.read_parquet(reference_run / "retained_flanks.parquet").rename(
        columns={
            "ridge_cell_id": "core_cell_id",
            "flank_lon": "front_lon",
            "flank_lat": "front_lat",
            "flank_distance_km": "distance_from_core_km",
            "absolute_transport_loss": "transport_loss_km_day",
            "relative_transport_loss": "relative_transport_loss",
        }
    )
    all_new_fronts = pd.read_parquet(production_run / "fronts.parquet")
    new_fronts = all_new_fronts.loc[all_new_fronts.front_detected]
    front_result = _numeric_comparison(
        old_fronts,
        new_fronts,
        keys=["core_cell_id", "side"],
        fields=FRONT_FIELDS,
    )
    expected_unobservable = int(
        old_cores.missing_side.isin(["left", "right"]).sum()
        + 2 * old_cores.missing_side.eq("left_and_right").sum()
    )
    statuses = all_new_fronts.front_status.value_counts()
    front_result["status_counts"] = {
        "probable_transport_front": int(statuses.get("probable_transport_front", 0)),
        "observable_no_retained_front": int(
            statuses.get("observable_no_retained_front", 0)
        ),
        "side_not_observable": int(statuses.get("side_not_observable", 0)),
    }
    front_result["expected_status_counts"] = {
        "probable_transport_front": len(old_fronts),
        "observable_no_retained_front": (
            2 * len(old_cores) - expected_unobservable - len(old_fronts)
        ),
        "side_not_observable": expected_unobservable,
    }

    exact = all(
        _exact_numeric(section)
        for section in (cell_result, core_result, drop_result, front_result)
    )
    exact &= core_result["classification_mismatches"] == 0
    exact &= front_result["status_counts"] == front_result["expected_status_counts"]
    return {
        "status": "exact" if exact else "different",
        "reference_run": str(reference_run.resolve()),
        "production_run": str(production_run.resolve()),
        "comparison": {
            "cell_statistics": cell_result,
            "current_cores": core_result,
            "candidate_drop_selection": drop_result,
            "probable_fronts": front_result,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--production-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.reference_run, args.production_run)
    output = args.output or args.production_run.with_name(
        f"{args.production_run.name}_regression.json"
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    if report["status"] != "exact":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
