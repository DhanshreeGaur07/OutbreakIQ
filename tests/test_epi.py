import numpy as np
import pandas as pd

from src.epi.transmission import GAMMA, fit_seir, estimate_rt, seir_forecast, serial_interval, simulate


def test_serial_interval_is_a_distribution():
    w = serial_interval()
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()


def test_rt_is_one_for_constant_incidence_and_above_one_for_growth():
    t = np.arange(120)
    flat = pd.DataFrame({"a": np.full(120, 500.0)})
    grow = pd.DataFrame({"a": 50 * np.exp(0.05 * t)})
    assert abs(estimate_rt(flat).iloc[-1, 0] - 1.0) < 0.05
    assert estimate_rt(grow).iloc[-1, 0] > 1.15


def test_rt_is_causal():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.poisson(200, 100).astype(float)})
    full = estimate_rt(x)
    trunc = estimate_rt(x.iloc[:60])
    assert np.allclose(full.iloc[:60].to_numpy(), trunc.to_numpy(), equal_nan=True)


def test_seir_conserves_population():
    n = 1e6
    states, _ = simulate(0.4, n, 200, (n - 100, 50, 50, 0))
    assert np.allclose(states.sum(axis=1), n, rtol=1e-6)


def test_seir_fit_recovers_transmission_rate():
    n, beta, rho = 5e6, 0.35, 0.25
    states, rep = simulate(beta, n, 59, (n - 4000, 1500, 1000, 0))
    obs = rho * rep
    fit = fit_seir(obs, n)
    assert fit is not None and fit.rmse_log < 0.05
    assert abs(fit.beta / GAMMA - beta / GAMMA) < 0.35        # R0 within ~0.35
    f = seir_forecast(fit, 14)
    assert len(f) == 14 and (f >= 0).all()
    assert seir_forecast(fit, 14, beta_multiplier=0.5)[-1] < f[-1]   # what-if: less transmission
