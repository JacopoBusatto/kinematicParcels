from __future__ import annotations

import argparse
from pathlib import Path

from .run_postprocessing import run_postprocessing


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

    config_path = Path(args.config)
    run_postprocessing(config_path)


if __name__ == "__main__":
    main()