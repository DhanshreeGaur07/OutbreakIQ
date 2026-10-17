"""Mechanistic epidemiology: Cori effective reproduction number and a fitted SEIR model."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.stats import gamma as gamma_dist

log = logging.getLogger(__name__)

SIGMA = 1 / 5.2   # incubation: E -> I
GAMMA = 1 / 7.0   # recovery:   I -> R


# --------------------------------------------------------------------------------------
# Effective reproduction number (Cori et al., 2013), vectorised over countries
# --------------------------------------------------------------------------------------
def serial_interval(mean: float = 5.2, sd: float = 1.7, max_days: int = 21) -> np.ndarray:
    shape, scale = (mean / sd) ** 2, sd ** 2 / mean
    cdf = gamma_dist.cdf(np.arange(0, max_days + 1), a=shape, scale=scale)
    w = np.diff(cdf)
    return w / w.sum()


def estimate_rt(incidence: pd.DataFrame, window: int = 7, prior_shape: float = 1.0,
                prior_scale: float = 1.0, si: np.ndarray | None = None) -> pd.DataFrame:
    """Posterior-mean Rt on a (date x country) matrix of daily case counts. Strictly causal:
    Rt at day t only uses incidence up to and including t."""
    si = serial_interval() if si is None else si
    inc = incidence.fillna(0.0).to_numpy(dtype=float)
    lam = np.zeros_like(inc)
    for s, w in enumerate(si, start=1):
        lam[s:] += w * inc[:-s]
    sum_i = pd.DataFrame(inc).rolling(window, min_periods=window).sum().to_numpy()
    sum_l = pd.DataFrame(lam).rolling(window, min_periods=window).sum().to_numpy()
    rt = (prior_shape + sum_i) / (1.0 / prior_scale + sum_l)
    return pd.DataFrame(rt, index=incidence.index, columns=incidence.columns)


# --------------------------------------------------------------------------------------
# SEIR with under-reporting
# --------------------------------------------------------------------------------------
def _rhs(y, t, beta, n):
    s, e, i, r = y
    new_e = beta * s * i / n
    return [-new_e, new_e - SIGMA * e, SIGMA * e - GAMMA * i, GAMMA * i]


def simulate(beta: float, n: float, days: int, state0: tuple[float, float, float, float]):
    """Integrate the SEIR ODEs; returns states (days+1, 4) and reported-case proxy sigma*E."""
    t = np.arange(days + 1)
    states = odeint(_rhs, state0, t, args=(beta, n))
    return states, SIGMA * states[:, 1]


@dataclass
class SeirFit:
    beta: float
    rho: float            # fraction of infections that are reported
    r_eff: float
    rmse_log: float
    state_end: tuple[float, float, float, float]
    n: float


def fit_seir(obs: np.ndarray, n: float) -> SeirFit | None:
    """Fit (beta, I0, rho) to the last ``len(obs)`` days of smoothed daily reported cases."""
    obs = np.asarray(obs, dtype=float)
    if len(obs) < 14 or np.nanmean(obs) < 5:
        return None
    days = len(obs)

    def sim(theta):
        beta, i0, rho = np.exp(theta)
        st0 = (n - 2.5 * i0, 1.5 * i0, i0, 0.0)
        states, rep = simulate(beta, n, days - 1, st0)
        return states, rho * rep

    def resid(theta):
        _, rep = sim(theta)
        return np.log1p(rep) - np.log1p(obs)

    lo = np.log([0.05, 1.0, 0.01])
    hi = np.log([1.5, 0.2 * n, 1.0])
    best = None
    for beta0 in (0.8 * GAMMA * 1.5, 1.6 * GAMMA * 1.5, 0.5 * GAMMA * 1.5):
        rho0 = 0.3
        i0 = max(obs[0] / (rho0 * SIGMA * 1.5), 2.0)
        x0 = np.clip(np.log([beta0, i0, rho0]), lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(resid, x0, bounds=(lo, hi), x_scale=1.0, max_nfev=60)
        except Exception as exc:  # numerical failure in the ODE solver
            log.debug("SEIR fit failed: %s", exc)
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None
    beta, i0, rho = np.exp(best.x)
    states, _ = sim(best.x)
    s_end = states[-1]
    r_eff = beta / GAMMA * s_end[0] / n
    rmse = float(np.sqrt(np.mean(best.fun ** 2)))
    return SeirFit(float(beta), float(rho), float(r_eff), rmse, tuple(map(float, s_end)), float(n))


def seir_forecast(fit: SeirFit, days: int, beta_multiplier: float = 1.0) -> np.ndarray:
    """Forecast reported daily cases from the fitted end-state; ``beta_multiplier`` = what-if."""
    _, rep = simulate(fit.beta * beta_multiplier, fit.n, days, fit.state_end)
    return fit.rho * rep[1:]


def fit_all(panel: pd.DataFrame, window: int, horizon: int, n_jobs: int = 1):
    """Fit one SEIR per country on the latest ``window`` days. Returns (params, forecasts)."""
    last = panel["date"].max()
    groups = []
    for iso, g in panel.sort_values("date").groupby("iso3"):
        g = g.tail(window)
        if g["date"].max() != last:
            continue
        groups.append((iso, g["country"].iloc[0], g["population"].iloc[0], g["new_smooth"].to_numpy()))

    def work(item):
        iso, name, pop, obs = item
        return item, fit_seir(obs, pop)

    results = Parallel(n_jobs=n_jobs)(delayed(work)(it) for it in groups)
    params, fcs = [], []
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    for (iso, name, pop, _), fit in results:
        if fit is None:
            continue
        params.append({
            "iso3": iso, "country": name, "beta": fit.beta, "gamma": GAMMA, "sigma": SIGMA,
            "rho": fit.rho, "r_eff": fit.r_eff, "rmse_log": fit.rmse_log, "population": fit.n,
            "S": fit.state_end[0], "E": fit.state_end[1], "I": fit.state_end[2],
            "R": fit.state_end[3], "as_of": last,
        })
        fc = seir_forecast(fit, horizon)
        fcs.append(pd.DataFrame({"iso3": iso, "date": future_dates,
                                 "seir_cases": fc, "seir_inc": fc / pop * 1e6}))
    p = pd.DataFrame(params)
    f = pd.concat(fcs, ignore_index=True) if fcs else pd.DataFrame(
        columns=["iso3", "date", "seir_cases", "seir_inc"])
    return p, f
