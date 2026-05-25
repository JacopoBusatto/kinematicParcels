from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_postprocess_config
from ..config.models import PostprocessConfig


def run_postprocessing(config_path: str | Path) -> None:
    """
    Main entry point for the post-processing pipeline.
    """
    cfg = load_postprocess_config(config_path)

    print("Post-processing started")
    print("Requested analyses:", cfg.analysis.types)

    context: dict = {}

    for analysis_name in cfg.analysis.types:
        print(f"\nRunning analysis: {analysis_name}")
        _run_single_analysis(cfg, analysis_name, context)


def _run_single_analysis(
    cfg: PostprocessConfig,
    analysis_name: str,
    context: dict,
) -> None:
    if analysis_name == "density":
        from ..workflows.run_density import run_density
        run_density(cfg, context)

    elif analysis_name == "beaching_times":
        from ..workflows.run_beaching_times import run_beaching_times
        run_beaching_times(cfg, context)

    elif analysis_name == "fsle":
        from ..workflows.run_fsle import run_fsle
        run_fsle(cfg, context)

    elif analysis_name == "trajectories":
        from ..workflows.run_trajectories import run_trajectories
        run_trajectories(cfg, context)

    elif analysis_name == "start_end_regions":
        from ..workflows.run_start_end_regions import run_start_end_regions
        run_start_end_regions(cfg, context)

    elif analysis_name == "meridional_crossing":
        from ..workflows.run_meridional_crossing import run_meridional_crossing
        run_meridional_crossing(cfg, context)

    elif analysis_name == "transition_probability":
        from ..workflows.run_transition_probability import run_transition_probability
        run_transition_probability(cfg, context)

    elif analysis_name == "summary":
        from ..workflows.run_summary import run_summary
        run_summary(cfg, context)

    else:
        raise ValueError(f"Unknown analysis type: {analysis_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Parcels post-processing from a YAML configuration file."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the post-processing YAML configuration file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_postprocessing(args.config)


if __name__ == "__main__":
    main()