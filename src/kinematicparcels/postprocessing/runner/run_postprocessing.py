from __future__ import annotations

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

    else:
        raise ValueError(f"Unknown analysis type: {analysis_name}")