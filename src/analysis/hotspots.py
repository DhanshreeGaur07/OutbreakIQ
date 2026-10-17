"""Spatial hotspot detection with Local Moran's I (LISA) on k-nearest-neighbour weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import knn_indices


def local_moran(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, k: int = 5,
                permutations: int = 999, seed: int = 0) -> pd.DataFrame:
    """Local Moran's I with row-standardised kNN weights and conditional permutation p-values.
    Quadrants: HH = hotspot, LL = coldspot, HL / LH = spatial outliers."""
    rng = np.random.default_rng(seed)
    n = len(values)
    nb = knn_indices(lat, lon, k)
    kk = nb.shape[1]
    z = (values - values.mean()) / (values.std() or 1.0)
    lag = z[nb].mean(axis=1)
    i_obs = z * lag

    # conditional permutation: hold z_i fixed, resample its neighbours from the other countries
    others = rng.integers(0, n - 1, size=(n, permutations, kk))
    others += others >= np.arange(n)[:, None, None]          # skip self
    lag_perm = z[others].mean(axis=2)
    i_perm = z[:, None] * lag_perm
    ge = (i_perm >= i_obs[:, None]).sum(axis=1)
    p = (np.minimum(ge, permutations - ge) + 1) / (permutations + 1) * 2
    p = np.clip(p, 0, 1)

    quad = np.where(z >= 0, np.where(lag >= 0, "HH", "HL"), np.where(lag >= 0, "LH", "LL"))
    cluster = np.where(p < 0.05, quad, "ns")
    return pd.DataFrame({"local_i": i_obs, "z": z, "z_lag": lag, "p_value": p,
                         "cluster": cluster})


def detect_hotspots(latest: pd.DataFrame, value_col: str = "surge_prob", k: int = 5) -> pd.DataFrame:
    """``latest``: one row per country with iso3, lat, lon and ``value_col``."""
    res = local_moran(latest[value_col].to_numpy(float), latest["lat"].to_numpy(float),
                      latest["lon"].to_numpy(float), k=k)
    return pd.concat([latest.reset_index(drop=True), res], axis=1)
