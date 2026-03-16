from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetCoordinatesConfig:
    trajectory: str = "trajectory"
    obs: str = "obs"
    time: str = "time"
    lon: str = "lon"
    lat: str = "lat"
    z: str | None = "z"


@dataclass(frozen=True)
class DatasetConfig:
    input_path: str
    coordinates: DatasetCoordinatesConfig = field(default_factory=DatasetCoordinatesConfig)


@dataclass(frozen=True)
class AnalysisConfig:
    type: str = "trajectories"


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "./outputs/postprocessing"


@dataclass(frozen=True)
class ExportsConfig:
    save_trajectory_table: bool = False
    save_particle_summary: bool = True
    table_format: str = "parquet"


@dataclass(frozen=True)
class GridConfig:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    dlon: float
    dlat: float


@dataclass(frozen=True)
class DensityConfig:
    lon_col: str = "lon"
    lat_col: str = "lat"
    time_col: str = "time"
    normalize_active: bool = True
    normalize_total: bool = True


@dataclass(frozen=True)
class CleaningConfig:
    truncate_stagnant: bool = False
    stagnant_tol: float = 1.0e-6
    stagnant_min_consecutive: int = 2


@dataclass(frozen=True)
class PostprocessConfig:
    dataset: DatasetConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    exports: ExportsConfig = field(default_factory=ExportsConfig)
    grid: GridConfig | None = None
    density: DensityConfig = field(default_factory=DensityConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)


@dataclass(frozen=True)
class ParcelsSchema:
    trajectory_dim: str
    obs_dim: str
    time_var: str
    lon_var: str
    lat_var: str
    z_var: str | None = None