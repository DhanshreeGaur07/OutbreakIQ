"""Data contracts. Errors stop the pipeline; warnings are logged. Run after every stage that
produces a persisted dataset, and again on the inference frame."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.ingest import JHU_FILE, OWID_FILE

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset violates its contract."""


def _fail(errors: list[str], stage: str) -> None:
    if errors:
        raise DataValidationError(f"[{stage}] " + "; ".join(errors))


def validate_raw(raw_dir: str | Path) -> None:
    raw, errors = Path(raw_dir), []
    for name in (JHU_FILE, OWID_FILE):
        f = raw / name
        if not f.exists() or f.stat().st_size < 1_000:
            errors.append(f"missing or empty raw file: {f}")
    if not errors:
        head = pd.read_csv(raw / JHU_FILE, nrows=1)
        for col in ("Province/State", "Country/Region", "Lat", "Long"):
            if col not in head.columns:
                errors.append(f"JHU file lacks column {col!r}")
        owid_cols = pd.read_csv(raw / OWID_FILE, nrows=0).columns
        for col in ("iso_code", "location", "date", "population"):
            if col not in owid_cols:
                errors.append(f"OWID file lacks column {col!r}")
    _fail(errors, "raw")


def validate_panel(panel: pd.DataFrame, cfg: Config, min_countries: int = 10) -> None:
    errors: list[str] = []
    need = {"iso3", "country", "continent", "date", "cum_cases", "new_cases", "new_smooth", "inc",
            "population", "lat", "lon", "active"}
    missing = need - set(panel.columns)
    if missing:
        raise DataValidationError(f"[panel] missing columns: {sorted(missing)}")
    if panel.duplicated(["iso3", "date"]).any():
        errors.append("duplicate (iso3, date) rows")
    if panel["iso3"].nunique() < min_countries:
        errors.append(f"only {panel['iso3'].nunique()} countries survived filtering (< {min_countries})")
    if (panel["new_cases"] < 0).any() or (panel["inc"] < 0).any():
        errors.append("negative case counts")
    if not np.isfinite(panel["inc"]).all():
        errors.append("non-finite incidence")
    if (panel["population"] <= 0).any():
        errors.append("non-positive population")
    if panel[["lat", "lon"]].isna().any().any():
        errors.append("missing coordinates")
    span = panel.groupby("iso3")["date"].agg(["min", "max", "count"])
    gaps = span[(span["max"] - span["min"]).dt.days + 1 != span["count"]]
    if len(gaps):
        errors.append(f"date gaps in {len(gaps)} countries (e.g. {list(gaps.index[:3])})")
    if span["max"].nunique() > 1:
        log.warning("countries end on different dates: %s", span["max"].value_counts().to_dict())
    _fail(errors, "panel")
    log.info("panel validated: %d countries, %d rows", panel["iso3"].nunique(), len(panel))


def validate_features(df: pd.DataFrame, features: list[str]) -> None:
    errors = []
    if not features:
        errors.append("empty feature list")
    sub = df[features]
    if np.isinf(sub.to_numpy(dtype=float)).any():
        errors.append("infinite values in features")
    nan_frac = sub.isna().mean()
    dead = nan_frac[nan_frac > 0.95].index.tolist()
    if dead:
        log.warning("features that are >95%% missing (kept, trees handle NaN): %s", dead)
    if df["y_growth"].notna().sum() < 1000:
        errors.append("fewer than 1000 labelled rows")
    if df.duplicated(["iso3", "date"]).any():
        errors.append("duplicate (iso3, date) rows")
    _fail(errors, "features")


def validate_inference_frame(df: pd.DataFrame, features: list[str]) -> None:
    """The scoring frame must contain every training feature (name-for-name)."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise DataValidationError(f"[inference] missing features: {missing}")
    if df.empty:
        raise DataValidationError("[inference] empty scoring frame")
