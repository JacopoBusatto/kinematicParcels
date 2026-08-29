from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import STAGES, run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Lagrangian transport branches and finite-time barriers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES)
    args = parser.parse_args()
    result = run_analysis(load_config(args.config), overwrite=args.overwrite,
                          resume=args.resume, stop_after=args.stop_after)
    print(result.run_dir)


if __name__ == "__main__":
    main()
