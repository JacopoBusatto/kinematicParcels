from __future__ import annotations

import argparse
import copy
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml


# -----------------------------------------------------------------------------
# YAML helpers
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


def ensure_nested_dict(d: dict, key: str) -> dict:
    if key not in d or d[key] is None:
        d[key] = {}
    return d[key]


# -----------------------------------------------------------------------------
# Time helpers
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


def resolve_run_names(series_cfg: dict) -> list[str]:
    if "schedule" in series_cfg and series_cfg["schedule"] is not None:
        sch = series_cfg["schedule"]
        subdir_fmt = sch.get("input_subdir_format", "%Y%m%d-%H%M")

        times = build_start_times(
            start_time=sch["start_time"],
            frequency=sch["frequency"],
            duration=sch["duration"],
        )
        return [t.strftime(subdir_fmt) for t in times]

    if "run_dirs" in series_cfg and series_cfg["run_dirs"] is not None:
        return list(series_cfg["run_dirs"])

    raise ValueError("You must define either series.schedule or series.run_dirs")


# -----------------------------------------------------------------------------
# Config generation
# -----------------------------------------------------------------------------
def build_single_postprocess_config(
    template_cfg: dict,
    *,
    sim_run_dir: Path,
    post_run_dir: Path,
    dataset_filename: str,
) -> dict:
    cfg = copy.deepcopy(template_cfg)

    dataset_cfg = ensure_nested_dict(cfg, "dataset")
    output_cfg = ensure_nested_dict(cfg, "output")

    dataset_cfg["input_path"] = str(sim_run_dir / dataset_filename)
    output_cfg["output_dir"] = str(post_run_dir)

    return cfg


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
def run_single_postprocess(runner_exe: str, config_path: Path) -> int:
    cmd = [runner_exe, str(config_path)]
    print("Launching:", " ".join(cmd))
    completed = subprocess.run(cmd)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Generate and run a series of post-processing jobs from a master YAML"
    )
    parser.add_argument("master_config", help="Path to the master postprocess series YAML")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate YAMLs, do not execute postprocessing",
    )
    args = parser.parse_args()

    master_path = Path(args.master_config)
    master_cfg = load_yaml(master_path)

    template_path = Path(master_cfg["template_config"])
    if not template_path.is_absolute():
        template_path = (master_path.parent / template_path).resolve()

    template_cfg = load_yaml(template_path)

    series_cfg = master_cfg["series"]

    simulation_output_root = Path(series_cfg["simulation_output_root"])
    postprocess_output_root = Path(series_cfg["postprocess_output_root"])

    run_names = resolve_run_names(series_cfg)
    dataset_filename = series_cfg.get("dataset_filename", "output.zarr")
    config_filename = series_cfg.get("config_filename", "postprocess.yml")
    runner_exe = series_cfg.get("runner_exe", "run-parcels-postprocess.exe")

    generated = []

    print(f"Template config         : {template_path}")
    print(f"Simulation output root  : {simulation_output_root}")
    print(f"Postprocess output root : {postprocess_output_root}")
    print(f"Runs to process         : {len(run_names)}")

    for i, run_name in enumerate(run_names, start=1):
        sim_run_dir = simulation_output_root / run_name
        post_run_dir = postprocess_output_root / run_name

        single_cfg = build_single_postprocess_config(
            template_cfg,
            sim_run_dir=sim_run_dir,
            post_run_dir=post_run_dir,
            dataset_filename=dataset_filename,
        )

        config_path = post_run_dir / config_filename
        dump_yaml(single_cfg, config_path)

        generated.append((run_name, sim_run_dir, post_run_dir, config_path))
        print(f"[{i}/{len(run_names)}] generated {config_path}")

    if args.generate_only:
        print("Generation completed. No postprocessing executed.")
        return

    failures = []

    for i, (run_name, sim_run_dir, post_run_dir, config_path) in enumerate(generated, start=1):
        print("-" * 80)
        print(f"[{i}/{len(generated)}] postprocessing run={run_name}")
        print(f"Input : {sim_run_dir / dataset_filename}")
        print(f"Output: {post_run_dir}")

        code = run_single_postprocess(runner_exe, config_path)

        if code != 0:
            failures.append((run_name, config_path, code))
            print(f"FAILED with return code {code}: {config_path}")
        else:
            print(f"OK: {config_path}")

    print("=" * 80)
    print(f"Total runs : {len(generated)}")
    print(f"Failures   : {len(failures)}")

    if failures:
        print("Failed runs:")
        for run_name, config_path, code in failures:
            print(f"  - {run_name} | code={code} | {config_path}")


if __name__ == "__main__":
    main()