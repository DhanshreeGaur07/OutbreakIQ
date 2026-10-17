"""Leakage-safe feature engineering.

Every feature at (country, date t) is computed from information available at or before t:
rolling windows are trailing, forward-fills only look backward, and the spatial / global
features aggregate *other countries on the same date t*. Targets are the only columns that
look forward, and they are prefixed ``y_``. ``tests/test_features.py`` verifies this property by
recomputing features on a truncated history and checking they are unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import Config
from src.data.preprocess import MOBILITY_COLS, STATIC_COLS, TV_COLS
from src.epi.transmission import estimate_rt

META_COLS = ["date", "iso3", "country", "continent", "inc", "active"]
TARGET_COLS = ["y_growth", "y_surge", "y_inc_future"]


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def knn_indices(lat: np.ndarray, lon: np.ndarray, k: int) -> np.ndarray:
    d = haversine_matrix(lat, lon)
    np.fill_diagonal(d, np.inf)
    k = min(k, len(lat) - 1)
    return np.argsort(d, axis=1)[:, :k]


def build_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a long frame with META, feature and target columns for all active country-days."""
    iso = sorted(panel["iso3"].unique())
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    def mat(col: str) -> pd.DataFrame:
        return (panel.pivot(index="date", columns="iso3", values=col)
                .reindex(index=dates, columns=iso))

    inc = mat("inc")
    cum_pm = mat("cum_pm")
    new_smooth = mat("new_smooth")
    active = mat("active").fillna(False).astype(bool)
    log_inc = np.log1p(inc)
    h = cfg.horizon

    f: dict[str, pd.DataFrame] = {"log_inc": log_inc}
    for k in (7, 14, 21, 28):
        f[f"g{k}"] = log_inc - log_inc.shift(k)
    # damped-trend extrapolation of 7-day momentum: baseline *and* anchor for the ML residual
    f["trend_growth"] = 0.5 * f["g7"] * cfg.horizon / 7.0
    f["accel7"] = f["g7"] - (log_inc.shift(7) - log_inc.shift(14))
    f["roll_std14"] = log_inc.rolling(14).std()
    f["from_peak90"] = log_inc - log_inc.rolling(90, min_periods=14).max()
    f["log_cum"] = np.log1p(cum_pm)
    rt = estimate_rt(new_smooth).clip(0, 5)
    f["rt_cori"] = rt
    f["rt_chg7"] = rt - rt.shift(7)

    # spatial importation pressure: mean of the k nearest countries on the same day
    coords = panel.groupby("iso3")[["lat", "lon"]].first().reindex(iso)
    nb = knn_indices(coords["lat"].to_numpy(), coords["lon"].to_numpy(), cfg.n_neighbors)
    lv, gv = log_inc.to_numpy(), f["g7"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN neighbourhoods -> NaN
        f["nb_log_inc"] = pd.DataFrame(np.nanmean(lv[:, nb], axis=2), index=dates, columns=iso)
        f["nb_g7"] = pd.DataFrame(np.nanmean(gv[:, nb], axis=2), index=dates, columns=iso)
    # global momentum: median 7-day growth across the panel (captures worldwide variant waves)
    world = f["g7"].median(axis=1)
    f["world_g7"] = pd.DataFrame(np.tile(world.to_numpy()[:, None], (1, len(iso))),
                                 index=dates, columns=iso)

    # time-varying covariates (trailing forward-fill only)
    for c in [*TV_COLS, *MOBILITY_COLS.values()]:
        if c in panel.columns:
            f[c] = mat(c).ffill()

    doy = dates.dayofyear.to_numpy()
    for name, fn in (("doy_sin", np.sin), ("doy_cos", np.cos)):
        f[name] = pd.DataFrame(np.tile(fn(2 * np.pi * doy / 365.25)[:, None], (1, len(iso))),
                               index=dates, columns=iso)

    # ---- targets (the only forward-looking columns)
    inc_future = inc.shift(-h)
    surge = ((inc_future >= cfg.surge_ratio * inc) & (inc_future >= cfg.surge_min_level))
    t = {
        "y_growth": np.log1p(inc_future) - log_inc,
        "y_surge": surge.astype(float).where(inc_future.notna()),
        "y_inc_future": inc_future,
    }

    index = pd.MultiIndex.from_product([dates, iso], names=["date", "iso3"])
    data = {name: m.reindex(index=dates, columns=iso).to_numpy(dtype="float64").ravel()
            for name, m in {**f, **t}.items()}
    data["inc"] = inc.to_numpy(dtype="float64").ravel()
    data["active"] = active.to_numpy().ravel()
    df = pd.DataFrame(data, index=index).reset_index()

    meta = panel.groupby("iso3").agg(
        country=("country", "first"), continent=("continent", "first"),
        **{c: (c, "first") for c in STATIC_COLS if c in panel.columns})
    for c in meta.columns:
        df[c] = df["iso3"].map(meta[c])
    for c in ("population", "population_density", "gdp_per_capita"):
        if c in df:
            df[c] = np.log1p(df[c])

    df = df[df["active"] & df["g28"].notna()].reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = set(META_COLS) | set(TARGET_COLS)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
