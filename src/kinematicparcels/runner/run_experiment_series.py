from __future__ import annotations

import argparse
import copy
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# -----------------------------------------------------------------------------
# Time parsing helpers
# -----------------------------------------------------------------------------
def parse_datetime_like(value: str) -> datetime:
    value = str(value).strip()
    for fmt in (
        "%Y%m%d-%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(
        f"Unsupported date format: {value}. "
        "Use one of: YYYYMMDD-HH:MM, YYYY-MM-DD HH:MM, YYYY-MM-DDTHH:MM, YYYYMMDD"
    )


def parse_timedelta_like(value: str) -> timedelta:
    """
    Supported examples:
      - 30min
      - 12H
      - 1D
      - 2.5D
    """
    s = str(value).strip().lower()

    if s.endswith("min"):
        return timedelta(minutes=float(s[:-3]))
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))

    raise ValueError(
        f"Unsupported frequency/duration format: {value}. "
        "Use e.g. 30min, 12H, 1D, 2.5D"
    )


def build_start_times(start_time: str, frequency: str, duration: str) -> list[datetime]:
    t0 = parse_datetime_like(start_time)
    dt = parse_timedelta_like(frequency)
    total = parse_timedelta_like(duration)

    if dt.total_seconds() <= 0:
        raise ValueError("frequency must be > 0")
    if total.total_seconds() < 0:
        raise ValueError("duration must be >= 0")

    times = []
    t = t0
    t_end = t0 + total

    while t <= t_end:
        times.append(t)
        t += dt

    return times


# -----------------------------------------------------------------------------
# Config generation
# -----------------------------------------------------------------------------
def ensure_nested_dict(d: dict, key: str) -> dict:
    if key not in d or d[key] is None:
        d[key] = {}
    return d[key]


def build_single_config(
    template_cfg: dict,
    *,
    run_start: datetime,
    output_root: Path,
    output_subdir_format: str,
) -> tuple[dict, Path]:
    cfg = copy.deepcopy(template_cfg)

    exp_cfg = ensure_nested_dict(cfg, "experiment")
    sim_cfg = ensure_nested_dict(cfg, "simulation")

    run_dir = output_root / run_start.strftime(output_subdir_format)
    exp_cfg["output_dir"] = str(run_dir)
    sim_cfg["start_time"] = run_start.strftime("%Y-%m-%d %H:%M")

    if "name" in exp_cfg and exp_cfg["name"]:
        exp_cfg["name"] = f"{exp_cfg['name']}_{run_start.strftime('%Y%m%d-%H%M')}"

    return cfg, run_dir


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
def run_single_experiment(runner_exe: str, config_path: Path) -> int:
    cmd = [runner_exe, str(config_path)]
    print("Launching:", " ".join(cmd))
    completed = subprocess.run(cmd)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Generate and run a series of Parcels experiments from a master YAML"
    )
    parser.add_argument("master_config", help="Path to master series YAML")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate single-run YAMLs, do not execute them",
    )
    args = parser.parse_args()

    master_path = Path(args.master_config)
    master_cfg = load_yaml(master_path)

    template_path = Path(master_cfg["template_config"])
    if not template_path.is_absolute():
        template_path = (master_path.parent / template_path).resolve()

    template_cfg = load_yaml(template_path)

    series_cfg = master_cfg["series"]
    output_root = Path(series_cfg["output_root"])
    output_subdir_format = series_cfg.get("output_subdir_format", "%Y%m%d-%H%M")
    config_filename = series_cfg.get("config_filename", "experiment.yml")
    runner_exe = series_cfg.get("runner_exe", "run-parcels-experiment.exe")

    start_times = build_start_times(
        start_time=series_cfg["start_time"],
        frequency=series_cfg["frequency"],
        duration=series_cfg["duration"],
    )

    print(f"Template config : {template_path}")
    print(f"Output root     : {output_root}")
    print(f"Generated runs  : {len(start_times)}")

    generated = []

    for i, run_start in enumerate(start_times, start=1):
        single_cfg, run_dir = build_single_config(
            template_cfg,
            run_start=run_start,
            output_root=output_root,
            output_subdir_format=output_subdir_format,
        )

        config_path = run_dir / config_filename
        dump_yaml(single_cfg, config_path)

        generated.append((run_start, run_dir, config_path))
        print(f"[{i}/{len(start_times)}] generated {config_path}")

    if args.generate_only:
        print("Generation completed. No simulations executed.")
        return

    failures = []

    for i, (run_start, run_dir, config_path) in enumerate(generated, start=1):
        print("-" * 80)
        print(f"[{i}/{len(generated)}] running start_time={run_start}")
        code = run_single_experiment(runner_exe, config_path)

        if code != 0:
            failures.append((run_start, config_path, code))
            print(f"FAILED with return code {code}: {config_path}")
        else:
            print(f"OK: {config_path}")

    print("=" * 80)
    print(f"Total runs   : {len(generated)}")
    print(f"Failures     : {len(failures)}")

    if failures:
        print("Failed runs:")
        for run_start, config_path, code in failures:
            print(f"  - {run_start} | code={code} | {config_path}")


if __name__ == "__main__":
    main()