from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys

import copernicusmarine


def daterange(start_date: datetime, end_date: datetime):
    """Yield one datetime per day from start_date to end_date inclusive."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_daily_files(
    output_dir: str | Path,
    username: str,
    password: str,
    dataset_id: str = "cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m",
    dataset_version: str = "202511",
    variables: list[str] | None = None,
    minimum_longitude: float = 11.0,
    maximum_longitude: float = 17.0,
    minimum_latitude: float = 35.0,
    maximum_latitude: float = 38.5,
    minimum_depth: float = 1.0182366371154785,
    maximum_depth: float = 104.94397735595703,
    start_date: str = "2026-05-02",
    end_date: str = "2026-05-10",
    overwrite: bool = False,
) -> None:
    """
    Download one NetCDF file per day from Copernicus Marine (3D forecast product).

    Files are saved as:
        med_currents_3d_YYYYMMDD.nc
    """
    if variables is None:
        variables = ["uo", "vo"]

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if end_dt < start_dt:
        raise ValueError("end_date must be >= start_date")

    for day in daterange(start_dt, end_dt):
        day_start = day.strftime("%Y-%m-%dT00:00:00")
        day_end = day.strftime("%Y-%m-%dT23:00:00")
        day_tag = day.strftime("%Y%m%d")
        filename = f"med_currents_3d_{day_tag}.nc"
        target_file = output_dir / filename

        if target_file.exists() and not overwrite:
            print(f"[SKIP] File already exists: {target_file}")
            continue

        print(f"[DOWNLOADING] {day_tag} -> {target_file}")

        try:
            copernicusmarine.subset(
                username=username,
                password=password,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                variables=variables,
                minimum_longitude=minimum_longitude,
                maximum_longitude=maximum_longitude,
                minimum_latitude=minimum_latitude,
                maximum_latitude=maximum_latitude,
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
                start_datetime=day_start,
                end_datetime=day_end,
                coordinates_selection_method="strict-inside",
                netcdf_compression_level=1,
                disable_progress_bar=True,
                output_directory=str(output_dir),
                output_filename=filename,
                overwrite=overwrite,
            )
            print(f"[OK] Saved: {target_file}")

        except Exception as exc:
            print(f"[ERROR] Failed on {day_tag}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download daily Copernicus Marine 3D forecast current files."
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Copernicus Marine username.",
    )
    parser.add_argument(
        "--password",
        required=True,
        help="Copernicus Marine password.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where NetCDF daily files will be saved.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-05-02",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-05-10",
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--dataset-id",
        default="cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m",
        help="Copernicus Marine dataset ID.",
    )
    parser.add_argument(
        "--dataset-version",
        default="202511",
        help="Dataset version string.",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=10.0,
        help="Minimum longitude (default: 10.0).",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=17.0,
        help="Maximum longitude (default: 17.0).",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=35.0,
        help="Minimum latitude (default: 35.0).",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=38.5,
        help="Maximum latitude (default: 38.5).",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=1.0182366371154785,
        help="Minimum depth in metres (default: ~1.02 m).",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=104.94397735595703,
        help="Maximum depth in metres (default: ~104.94 m).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    download_daily_files(
        output_dir=args.output_dir,
        username=args.username,
        password=args.password,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        minimum_longitude=args.lon_min,
        maximum_longitude=args.lon_max,
        minimum_latitude=args.lat_min,
        maximum_latitude=args.lat_max,
        minimum_depth=args.depth_min,
        maximum_depth=args.depth_max,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# python download_mfs_forecast_3d.py --username jbusatto --password TELEroma3FONO1! --output-dir "C:/Users/Jacopo/Documents/DATI/MFS_FORECAST/3D/SICILY/HOURLY/" --start-date 2026-04-01 --end-date 2026-04-27 --overwrite
