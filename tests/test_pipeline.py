import numpy as np
import pandas as pd

from src.models.backtest import make_folds, run_backtest, split_fold
from src.models.evaluate import evaluate
from src.config import Config
from src.data.preprocess import build_panel
from src.features.build_features import TARGET_COLS, build_features, feature_columns
from src.analysis.hotspots import local_moran
from src.data.synthetic import write_synthetic


def _cfg(tmp_path, **kw):
    write_synthetic(tmp_path, days=520, seed=3)
    base = dict(raw_dir=str(tmp_path), first_cutoff="2020-11-01", step_days=120, min_total_cases=500,
                min_train_rows=500)
    base.update(kw)
    return Config(**base)


def test_panel_cleaning(tmp_path):
    cfg = _cfg(tmp_path)
    panel = build_panel(cfg)
    assert "Diamond Princess" not in set(panel["country"])          # non-country rows dropped
    assert panel["iso3"].str.len().eq(3).all()
    assert (panel["new_cases"] >= 0).all()
    assert panel.groupby("iso3")["cum_cases"].apply(lambda s: s.is_monotonic_increasing).all()
    assert panel.loc[panel["iso3"] == "CHN"].shape[0] == panel.loc[panel["iso3"] == "USA"].shape[0]


def test_features_have_no_lookahead(tmp_path):
    """Features at day t must not change when future data is removed."""
    cfg = _cfg(tmp_path)
    panel = build_panel(cfg)
    cut = panel["date"].max() - pd.Timedelta(days=60)
    full = build_features(panel, cfg)
    trunc = build_features(panel[panel["date"] <= cut], cfg)
    feats = [c for c in feature_columns(full) if c not in TARGET_COLS]
    a = full[full["date"] <= cut].sort_values(["iso3", "date"]).reset_index(drop=True)
    b = trunc.sort_values(["iso3", "date"]).reset_index(drop=True)
    assert len(a) == len(b)
    assert np.allclose(a[feats].to_numpy(float), b[feats].to_numpy(float), equal_nan=True, atol=1e-9)


def test_folds_are_embargoed(tmp_path):
    cfg = _cfg(tmp_path)
    df = build_features(build_panel(cfg), cfg)
    h = pd.Timedelta(days=cfg.horizon)
    for fold in make_folds(df, cfg):
        fit, calib, test = split_fold(df, fold, cfg)
        if fit.empty or calib.empty or test.empty:
            continue
        assert fit["date"].max() + h <= calib["date"].min() + pd.Timedelta(days=1)   # no target overlap
        assert (fit["date"] + h).max() < fold.cutoff                                  # labels known at cutoff
        assert calib["date"].max() + h <= fold.cutoff
        assert test["date"].min() > fold.cutoff


def test_backtest_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    df = build_features(build_panel(cfg), cfg)
    feats = feature_columns(df)
    pred, models = run_backtest(df, feats, cfg)
    assert pred["is_live"].any() and not pred.loc[pred["is_live"], "y_growth"].notna().any()
    assert pred["surge_prob"].between(0, 1).all()
    assert (pred["q_lo"] <= pred["pred_growth"] + 1e-9).all() and (pred["pred_growth"] <= pred["q_hi"] + 1e-9).all()
    m = evaluate(pred, cfg)
    reg = m["regression"]
    assert reg["gradient_boosting"]["mae"] < reg["persistence"]["mae"]
    # NB: with only ~500 days of history the ML correction cannot reliably beat the damped-trend
    # baseline (regime shifts between folds); that comparison is made on the full backtest.
    assert 0.5 < m["interval"]["coverage"] < 0.97
    assert m["surge_classification"]["conformal_distributional"]["brier_skill_vs_climatology"] > 0


def test_local_moran_finds_a_planted_hotspot():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-40, 60, 80)
    lon = rng.uniform(-120, 120, 80)
    v = rng.normal(0, 0.1, 80)
    centre = 0
    d = np.hypot(lat - lat[centre], lon - lon[centre])
    v[np.argsort(d)[:8]] += 3.0                                    # a tight high-value cluster
    res = local_moran(v, lat, lon, k=4, permutations=499)
    assert (res["cluster"].iloc[np.argsort(d)[:8]] == "HH").sum() >= 3
