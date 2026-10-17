"""Central configuration: one typed dataclass, overridden from config/config.yaml and the CLI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # paths
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    log_dir: str = "logs"
    # synthetic demo
    synthetic_days: int = 1100
    synthetic_seed: int = 7
    # data
    min_population: float = 1_000_000
    min_total_cases: int = 10_000
    start_cases: int = 100          # a country enters the modelling panel after this many cases
    smooth_window: int = 7
    use_mobility: bool = False
    # forecasting task
    horizon: int = 14
    surge_ratio: float = 1.25
    surge_min_level: float = 5.0    # cases / million / day
    # features
    n_neighbors: int = 5
    # walk-forward backtest
    first_cutoff: str = "2020-10-01"
    step_days: int = 90
    calib_days: int = 120
    min_train_rows: int = 2000
    interval_alpha: float = 0.2     # 80% prediction intervals
    eval_min_incidence: float = 1.0  # evaluate where incidence >= 1 case / million / day
    seed: int = 42
    # transmission model
    seir_window: int = 42
    seir_horizon: int = 28
    n_jobs: int = 1

    def for_synthetic(self) -> "Config":
        """Isolate every path used by the offline demo so real data/models are never touched."""
        self.raw_dir, self.interim_dir = "data/synthetic/raw", "data/synthetic/interim"
        self.processed_dir, self.models_dir = "data/synthetic/processed", "models/synthetic"
        self.artifacts_dir, self.log_dir = "artifacts/synthetic", "logs/synthetic"
        return self

    @classmethod
    def load(cls, path: str | Path | None = "config/config.yaml", **overrides) -> "Config":
        data: dict = {}
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
