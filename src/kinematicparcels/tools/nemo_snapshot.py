from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy as np
from netCDF4 import Dataset, num2date


_FILE_ATTRIBUTE_OFFLINE = 0x00001000
_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000

_COMMON_TIME_NAMES = {
    "t",
    "time",
    "time_centered",
    "time_counter",
    "nav_time",
    "ocean_time",
}


@dataclass(frozen=True)
class TimeRecord:
    dimension: str
    variable: str
    index: int
    raw_value: Any
    units: str | None
    calendar: str | None
    decoded_value: Any | None

    @property
    def label(self) -> str:
        if self.decoded_value is not None:
            return str(self.decoded_value)
        suffix = f" {self.units}" if self.units else ""
        return f"{self.raw_value}{suffix}"


@dataclass(frozen=True)
class NemoSnapshotResult:
    u_output: Path
    v_output: Path
    selected_time: str
    time_index: int
    u_time_dimension: str
    v_time_dimension: str


@dataclass(frozen=True)
class NemoFCoordinatesResult:
    output: Path
    lon_variable: str
    lat_variable: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]


def _cloud_recall_attributes(path: Path) -> tuple[str, ...]:
    """Return Windows cloud-placeholder attributes without opening file content."""
    attributes = int(getattr(path.stat(), "st_file_attributes", 0))
    flags: list[str] = []
    if attributes & _FILE_ATTRIBUTE_OFFLINE:
        flags.append("OFFLINE")
    if attributes & _FILE_ATTRIBUTE_RECALL_ON_OPEN:
        flags.append("RECALL_ON_OPEN")
    if attributes & _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS:
        flags.append("RECALL_ON_DATA_ACCESS")
    return tuple(flags)


def _resolve_input(path: str | Path, *, allow_cloud_download: bool) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"NEMO input file not found: {resolved}")

    recall_flags = _cloud_recall_attributes(resolved)
    if recall_flags and not allow_cloud_download:
        flags = ", ".join(recall_flags)
        raise RuntimeError(
            f"Refusing to open online-only cloud file {resolved} ({flags}). "
            "Opening it may download the complete source file. Run this tool where the "
            "file is already local, or pass --allow-cloud-download to accept that transfer."
        )
    return resolved


def _canonical_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _time_score(name: str, variable: Any, dimension: str) -> int:
    score = 0
    lower_name = name.lower()
    standard_name = str(getattr(variable, "standard_name", "")).lower()
    axis = str(getattr(variable, "axis", "")).upper()
    units = str(getattr(variable, "units", "")).lower()

    if standard_name == "time":
        score += 100
    if axis == "T":
        score += 100
    if " since " in units:
        score += 80
    if lower_name == dimension.lower():
        score += 40
    if lower_name in _COMMON_TIME_NAMES:
        score += 30
    if "time" in lower_name:
        score += 20
    return score


def _detect_time_dimension(dataset: Dataset, explicit: str | None, *, component: str) -> str:
    if explicit is not None:
        if explicit not in dataset.dimensions:
            available = ", ".join(dataset.dimensions)
            raise ValueError(
                f"{component} time dimension '{explicit}' was not found. "
                f"Available dimensions: {available}"
            )
        return explicit

    scores: dict[str, int] = {}
    for name, variable in dataset.variables.items():
        if len(variable.dimensions) != 1:
            continue
        dimension = variable.dimensions[0]
        score = _time_score(name, variable, dimension)
        if score:
            scores[dimension] = max(scores.get(dimension, 0), score)

    for dimension in dataset.dimensions:
        lower_dimension = dimension.lower()
        if lower_dimension in _COMMON_TIME_NAMES:
            scores[dimension] = max(scores.get(dimension, 0), 30)
        elif "time" in lower_dimension:
            scores[dimension] = max(scores.get(dimension, 0), 20)

    if not scores:
        available = ", ".join(dataset.dimensions)
        raise ValueError(
            f"Could not detect the {component} time dimension from CF metadata. "
            f"Available dimensions: {available}. Supply --time-dim explicitly."
        )

    best_score = max(scores.values())
    best = sorted(dimension for dimension, score in scores.items() if score == best_score)
    if len(best) != 1:
        raise ValueError(
            f"Ambiguous {component} time dimensions: {', '.join(best)}. "
            "Supply --time-dim explicitly."
        )
    return best[0]


def _find_time_variable(dataset: Dataset, time_dimension: str, *, component: str) -> str:
    candidates: list[tuple[int, str]] = []
    for name, variable in dataset.variables.items():
        if variable.dimensions != (time_dimension,):
            continue
        candidates.append((_time_score(name, variable, time_dimension), name))

    if not candidates:
        raise ValueError(
            f"Could not find a one-dimensional {component} time coordinate variable "
            f"for dimension '{time_dimension}'."
        )

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _read_time_record(
    dataset: Dataset,
    time_dimension: str,
    time_index: int,
    *,
    component: str,
) -> TimeRecord:
    size = len(dataset.dimensions[time_dimension])
    if time_index < 0 or time_index >= size:
        raise IndexError(
            f"{component} time index {time_index} is outside dimension "
            f"'{time_dimension}' with length {size}."
        )

    variable_name = _find_time_variable(dataset, time_dimension, component=component)
    variable = dataset.variables[variable_name]
    value = np.ma.asarray(variable[time_index])
    if value.size != 1 or np.ma.is_masked(value):
        raise ValueError(
            f"{component} time coordinate '{variable_name}' has no usable scalar "
            f"at index {time_index}."
        )
    raw_value = value.reshape(-1)[0].item()

    units_value = getattr(variable, "units", None)
    units = str(units_value) if units_value is not None else None
    calendar_value = getattr(variable, "calendar", None)
    calendar = str(calendar_value) if calendar_value is not None else None
    decoded_value = None
    if units and " since " in units.lower():
        try:
            decoded_value = num2date(
                raw_value,
                units=units,
                calendar=calendar or "standard",
                only_use_cftime_datetimes=True,
            )
        except (TypeError, ValueError, OverflowError):
            decoded_value = None

    return TimeRecord(
        dimension=time_dimension,
        variable=variable_name,
        index=time_index,
        raw_value=raw_value,
        units=units,
        calendar=calendar,
        decoded_value=decoded_value,
    )


def _datetime_key(value: Any) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(value.year),
        int(value.month),
        int(value.day),
        int(value.hour),
        int(value.minute),
        int(value.second),
        int(getattr(value, "microsecond", 0)),
    )


def _validate_matching_times(u_time: TimeRecord, v_time: TimeRecord) -> None:
    if u_time.decoded_value is not None and v_time.decoded_value is not None:
        matches = _datetime_key(u_time.decoded_value) == _datetime_key(v_time.decoded_value)
    else:
        matches = (
            u_time.raw_value == v_time.raw_value
            and u_time.units == v_time.units
            and (u_time.calendar or "standard") == (v_time.calendar or "standard")
        )

    if not matches:
        raise ValueError(
            "Selected U/V timestamps do not match: "
            f"U={u_time.label} ({u_time.variable}), "
            f"V={v_time.label} ({v_time.variable})."
        )


def _variable_creation_options(
    variable: Any,
    dataset: Dataset,
    target_dimension_sizes: dict[str, int],
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if "_FillValue" in variable.ncattrs():
        options["fill_value"] = variable.getncattr("_FillValue")
    else:
        options["fill_value"] = False

    filters = variable.filters()
    if filters and filters.get("zlib"):
        options.update(
            compression="zlib",
            complevel=int(filters.get("complevel", 4)),
            shuffle=bool(filters.get("shuffle", True)),
            fletcher32=bool(filters.get("fletcher32", False)),
        )
    elif filters and any(filters.get(name) for name in ("szip", "zstd", "bzip2", "blosc")):
        warnings.warn(
            f"Variable '{variable.name}' uses a non-zlib compression filter; "
            "the snapshot will use the NetCDF default encoding.",
            RuntimeWarning,
            stacklevel=2,
        )

    chunking = variable.chunking()
    if isinstance(chunking, (list, tuple)):
        chunksizes: list[int] = []
        for dimension, chunk_size in zip(variable.dimensions, chunking):
            target_size = target_dimension_sizes.get(
                dimension,
                len(dataset.dimensions[dimension]),
            )
            chunksizes.append(max(1, min(int(chunk_size), max(1, target_size))))
        options["chunksizes"] = tuple(chunksizes)
    elif chunking == "contiguous" and not any(
        dataset.dimensions[dimension].isunlimited() for dimension in variable.dimensions
    ):
        options["contiguous"] = True

    try:
        dtype = np.dtype(variable.dtype)
    except TypeError:
        dtype = None
    endian = variable.endian()
    if dtype is not None and dtype.kind not in {"S", "U", "O"} and endian in {"little", "big"}:
        options["endian"] = endian

    return options


def _copy_dataset_snapshot(
    source: Dataset,
    destination_path: Path,
    *,
    time_dimension: str,
    time_index: int,
) -> None:
    if source.groups:
        raise NotImplementedError("NetCDF groups are not supported by the NEMO snapshot extractor.")

    with Dataset(destination_path, mode="w", format=source.data_model) as destination:
        destination.setncatts({name: source.getncattr(name) for name in source.ncattrs()})

        for name, dimension in source.dimensions.items():
            if name == time_dimension:
                size = None if dimension.isunlimited() else 1
            else:
                size = None if dimension.isunlimited() else len(dimension)
            destination.createDimension(name, size)

        for name, source_variable in source.variables.items():
            options = _variable_creation_options(
                source_variable,
                source,
                {time_dimension: 1},
            )
            try:
                destination_variable = destination.createVariable(
                    name,
                    source_variable.datatype,
                    source_variable.dimensions,
                    **options,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"Could not reproduce NetCDF variable '{name}': {exc}") from exc

            attributes = {
                attribute: source_variable.getncattr(attribute)
                for attribute in source_variable.ncattrs()
                if attribute != "_FillValue"
            }
            destination_variable.setncatts(attributes)
            source_variable.set_auto_maskandscale(False)
            destination_variable.set_auto_maskandscale(False)
            if hasattr(source_variable, "set_auto_chartostring"):
                source_variable.set_auto_chartostring(False)
                destination_variable.set_auto_chartostring(False)

            if time_dimension in source_variable.dimensions:
                source_slice: list[slice] = [slice(None)] * source_variable.ndim
                source_slice[source_variable.dimensions.index(time_dimension)] = slice(time_index, time_index + 1)
                destination_variable[...] = source_variable[tuple(source_slice)]
            else:
                destination_variable[...] = source_variable[...]


def _temporary_output_path(output_path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.nc",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    return temporary_path


def extract_nemo_snapshot(
    u_input: str | Path,
    v_input: str | Path,
    u_output: str | Path,
    v_output: str | Path,
    *,
    time_index: int = 0,
    time_dim: str | None = None,
    overwrite: bool = False,
    allow_cloud_download: bool = False,
) -> NemoSnapshotResult:
    """Extract one matching NEMO U/V record while retaining a length-one time axis."""
    if time_index < 0:
        raise ValueError("time_index must be a non-negative, zero-based index")

    u_source = _resolve_input(u_input, allow_cloud_download=allow_cloud_download)
    v_source = _resolve_input(v_input, allow_cloud_download=allow_cloud_download)
    u_destination = Path(u_output).expanduser().resolve()
    v_destination = Path(v_output).expanduser().resolve()

    source_paths = {_canonical_path(u_source), _canonical_path(v_source)}
    output_paths = {_canonical_path(u_destination), _canonical_path(v_destination)}
    if len(output_paths) != 2:
        raise ValueError("U and V output paths must be different")
    if source_paths & output_paths:
        raise ValueError("Output paths must not overwrite either source file")

    existing = [path for path in (u_destination, v_destination) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Snapshot output already exists: {paths}. Pass --overwrite to replace it.")

    temporary_paths: list[Path] = []
    try:
        with Dataset(u_source, mode="r") as u_dataset, Dataset(v_source, mode="r") as v_dataset:
            u_time_dimension = _detect_time_dimension(u_dataset, time_dim, component="U")
            v_time_dimension = _detect_time_dimension(v_dataset, time_dim, component="V")
            u_time = _read_time_record(
                u_dataset,
                u_time_dimension,
                time_index,
                component="U",
            )
            v_time = _read_time_record(
                v_dataset,
                v_time_dimension,
                time_index,
                component="V",
            )
            _validate_matching_times(u_time, v_time)

            u_destination.parent.mkdir(parents=True, exist_ok=True)
            v_destination.parent.mkdir(parents=True, exist_ok=True)
            u_temporary = _temporary_output_path(u_destination)
            v_temporary = _temporary_output_path(v_destination)
            temporary_paths.extend((u_temporary, v_temporary))

            _copy_dataset_snapshot(
                u_dataset,
                u_temporary,
                time_dimension=u_time_dimension,
                time_index=time_index,
            )
            _copy_dataset_snapshot(
                v_dataset,
                v_temporary,
                time_dimension=v_time_dimension,
                time_index=time_index,
            )

        os.replace(u_temporary, u_destination)
        temporary_paths.remove(u_temporary)
        os.replace(v_temporary, v_destination)
        temporary_paths.remove(v_temporary)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    return NemoSnapshotResult(
        u_output=u_destination,
        v_output=v_destination,
        selected_time=u_time.label,
        time_index=time_index,
        u_time_dimension=u_time_dimension,
        v_time_dimension=v_time_dimension,
    )


def _validate_f_coordinate_variables(
    source: Dataset,
    *,
    lon_var: str,
    lat_var: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if source.groups:
        raise NotImplementedError(
            "NetCDF groups are not supported by the NEMO coordinate extractor."
        )
    if lon_var == lat_var:
        raise ValueError("F-node longitude and latitude variables must be different")

    missing = [name for name in (lon_var, lat_var) if name not in source.variables]
    if missing:
        available = ", ".join(source.variables)
        raise ValueError(
            f"F-node coordinate variable(s) not found: {', '.join(missing)}. "
            f"Available variables: {available}"
        )

    lon = source.variables[lon_var]
    lat = source.variables[lat_var]
    if lon.dimensions != lat.dimensions or lon.shape != lat.shape:
        raise ValueError(
            f"F-node coordinates must have identical dimensions and shapes; "
            f"{lon_var}{lon.dimensions}{lon.shape} != {lat_var}{lat.dimensions}{lat.shape}"
        )
    if lon.ndim not in {2, 3}:
        raise ValueError(
            "F-node coordinates must be 2-D (y, x) or 3-D with one singleton "
            "leading dimension"
        )
    if lon.ndim == 3 and lon.shape[0] != 1:
        raise ValueError(
            "The leading F-node coordinate dimension must have length one; "
            f"found shape {lon.shape}"
        )
    if lon.shape[-2] < 2 or lon.shape[-1] < 2:
        raise ValueError(f"F-node coordinate grid is too small: {lon.shape}")

    for name, variable in ((lon_var, lon), (lat_var, lat)):
        try:
            dtype = np.dtype(variable.dtype)
        except TypeError as exc:
            raise ValueError(f"F-node coordinate variable '{name}' is not numeric") from exc
        if dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(f"F-node coordinate variable '{name}' is not numeric")

        values = np.ma.asarray(variable[:], dtype=float)
        usable = np.ma.compressed(np.ma.masked_invalid(values))
        if usable.size == 0:
            raise ValueError(f"F-node coordinate variable '{name}' has no finite values")

    return tuple(lon.dimensions), tuple(int(size) for size in lon.shape)


def _copy_f_coordinate_dataset(
    source: Dataset,
    destination_path: Path,
    *,
    lon_var: str,
    lat_var: str,
    coordinate_dimensions: tuple[str, ...],
) -> None:
    retained_variables = {lon_var, lat_var}
    for dimension in coordinate_dimensions:
        if dimension not in source.variables:
            continue
        dimension_variable = source.variables[dimension]
        if dimension_variable.dimensions == (dimension,):
            retained_variables.add(dimension)

    with Dataset(destination_path, mode="w", format=source.data_model) as destination:
        destination.setncatts({name: source.getncattr(name) for name in source.ncattrs()})

        for name in coordinate_dimensions:
            dimension = source.dimensions[name]
            size = None if dimension.isunlimited() else len(dimension)
            destination.createDimension(name, size)

        for name, source_variable in source.variables.items():
            if name not in retained_variables:
                continue

            options = _variable_creation_options(source_variable, source, {})
            try:
                destination_variable = destination.createVariable(
                    name,
                    source_variable.datatype,
                    source_variable.dimensions,
                    **options,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Could not reproduce NetCDF coordinate variable '{name}': {exc}"
                ) from exc

            attributes: dict[str, Any] = {}
            for attribute in source_variable.ncattrs():
                if attribute == "_FillValue":
                    continue
                value = source_variable.getncattr(attribute)
                if attribute in {"bounds", "grid_mapping", "ancillary_variables", "coordinates"}:
                    referenced = str(value).split()
                    if any(reference not in retained_variables for reference in referenced):
                        continue
                attributes[attribute] = value
            destination_variable.setncatts(attributes)

            source_variable.set_auto_maskandscale(False)
            destination_variable.set_auto_maskandscale(False)
            if hasattr(source_variable, "set_auto_chartostring"):
                source_variable.set_auto_chartostring(False)
                destination_variable.set_auto_chartostring(False)
            destination_variable[...] = source_variable[...]


def extract_nemo_f_coordinates(
    mesh_input: str | Path,
    output: str | Path,
    *,
    lon_var: str = "glamf",
    lat_var: str = "gphif",
    overwrite: bool = False,
    allow_cloud_download: bool = False,
) -> NemoFCoordinatesResult:
    """Extract only the shared F-node coordinates required by Parcels from_nemo."""
    source_path = _resolve_input(
        mesh_input,
        allow_cloud_download=allow_cloud_download,
    )
    destination_path = Path(output).expanduser().resolve()
    if _canonical_path(source_path) == _canonical_path(destination_path):
        raise ValueError("Coordinate output path must not overwrite the mesh source")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Coordinate output already exists: {destination_path}. "
            "Pass --overwrite to replace it."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_output_path(destination_path)
    try:
        with Dataset(source_path, mode="r") as source:
            dimensions, shape = _validate_f_coordinate_variables(
                source,
                lon_var=lon_var,
                lat_var=lat_var,
            )
            _copy_f_coordinate_dataset(
                source,
                temporary_path,
                lon_var=lon_var,
                lat_var=lat_var,
                coordinate_dimensions=dimensions,
            )
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return NemoFCoordinatesResult(
        output=destination_path,
        lon_variable=lon_var,
        lat_variable=lat_var,
        dimensions=dimensions,
        shape=shape,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one matching time record from separate NEMO U/V NetCDF files. "
            "The outputs retain a singleton time dimension for stationary Parcels runs."
        )
    )
    parser.add_argument("u_file", help="NEMO U-component NetCDF file")
    parser.add_argument("v_file", help="NEMO V-component NetCDF file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the reduced U/V snapshot files",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Zero-based time record to extract (default: 0)",
    )
    parser.add_argument(
        "--time-dim",
        default=None,
        help="Time dimension name; by default it is detected from CF metadata",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing snapshot outputs",
    )
    parser.add_argument(
        "--allow-cloud-download",
        action="store_true",
        help="Allow opening cloud-placeholder inputs, which may download the complete files",
    )
    return parser


def build_coordinates_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only the shared NEMO F-node longitude/latitude arrays needed "
            "by Parcels, omitting masks, metrics, bathymetry, and 3-D mesh fields."
        )
    )
    parser.add_argument("mesh_file", help="NEMO mesh NetCDF file")
    parser.add_argument("--output", required=True, help="Reduced coordinate NetCDF output")
    parser.add_argument("--lon-var", default="glamf", help="F-node longitude variable")
    parser.add_argument("--lat-var", default="gphif", help="F-node latitude variable")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    parser.add_argument(
        "--allow-cloud-download",
        action="store_true",
        help="Allow opening a cloud-placeholder mesh, which may download the complete file",
    )
    return parser


def _default_output_path(input_path: str | Path, output_dir: Path, time_index: int) -> Path:
    source = Path(input_path)
    suffix = source.suffix or ".nc"
    return output_dir / f"{source.stem}_snapshot_t{time_index:04d}{suffix}"


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    u_output = _default_output_path(args.u_file, output_dir, args.time_index)
    v_output = _default_output_path(args.v_file, output_dir, args.time_index)

    result = extract_nemo_snapshot(
        args.u_file,
        args.v_file,
        u_output,
        v_output,
        time_index=args.time_index,
        time_dim=args.time_dim,
        overwrite=args.overwrite,
        allow_cloud_download=args.allow_cloud_download,
    )
    print(f"Selected time: {result.selected_time}")
    print(f"U snapshot: {result.u_output}")
    print(f"V snapshot: {result.v_output}")
    print("Use allow_time_extrapolation=true and time_periodic=false for a stationary field.")


def coordinates_main() -> None:
    args = build_coordinates_parser().parse_args()
    result = extract_nemo_f_coordinates(
        args.mesh_file,
        args.output,
        lon_var=args.lon_var,
        lat_var=args.lat_var,
        overwrite=args.overwrite,
        allow_cloud_download=args.allow_cloud_download,
    )
    print(f"F-node coordinates: {result.output}")
    print(
        f"Variables: lon={result.lon_variable}, lat={result.lat_variable}; "
        f"dimensions={result.dimensions}, shape={result.shape}"
    )


if __name__ == "__main__":
    main()
