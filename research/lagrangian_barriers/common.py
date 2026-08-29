from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "start_lon_bin", "start_lat_bin", "end_lon_bin", "end_lat_bin",
    "start_lon_center", "start_lat_center", "end_lon_center", "end_lat_center",
    "transition_count", "transition_probability",
)
KEY_COLUMNS = ("start_lon_bin", "start_lat_bin", "end_lon_bin", "end_lat_bin")


@dataclass(frozen=True)
class StageResult:
    name: str
    counts: dict[str, int | float]
    warnings: tuple[str, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, calculate_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path) if calculate_hash else None,
    }


def git_record(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def environment_record(repo: Path) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_record(repo),
        "packages": package_versions(
            ["numpy", "pandas", "pyarrow", "scipy", "xarray", "networkx", "pyproj", "shapely", "matplotlib", "cartopy"]
        ),
    }


def json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def circular_difference_degrees(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def unwrap_longitudes(lon: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(lon, dtype=float))))


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"lagrangian_barriers.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for filename, level in (("analysis.log", logging.INFO), ("warnings.log", logging.WARNING)):
        handler = logging.FileHandler(run_dir / "logs" / filename, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "timestamp_utc": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname, "logger": record.name,
                "message": record.getMessage(),
            })
    structured = logging.FileHandler(run_dir / "logs" / "events.jsonl", encoding="utf-8")
    structured.setLevel(logging.INFO); structured.setFormatter(JsonFormatter())
    logger.addHandler(structured)
    return logger


def local_grid_diagonal_km(geod: Any, lon: float, lat: float, dlon: float, dlat: float) -> float:
    _, _, dx = geod.inv(lon, lat, lon + dlon, lat)
    _, _, dy = geod.inv(lon, lat, lon, lat + dlat)
    return math.hypot(dx, dy) / 1000.0
