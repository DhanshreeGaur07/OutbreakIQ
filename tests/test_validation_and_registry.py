import json

import numpy as np
import pandas as pd

from src.config import Config
from src.data.preprocess import build_panel
from src.data.synthetic import write_synthetic
from src.data.validate import (DataValidationError, validate_features, validate_inference_frame,
                               validate_panel, validate_raw)
from src.features.build_features import build_features, feature_columns
from src.models.registry import load_run


def _cfg(tmp_path):
    write_synthetic(tmp_path / "raw", days=400, seed=5)
    return Config(raw_dir=str(tmp_path / "raw"), min_total_cases=300, first_cutoff="2020-11-01")


def _raises(fn, *a):
    try:
        fn(*a)
    except DataValidationError:
        return True
    return False


def test_validators_accept_good_and_reject_bad_data(tmp_path):
    cfg = _cfg(tmp_path)
    validate_raw(cfg.raw_dir)
    panel = build_panel(cfg)
    validate_panel(panel, cfg)

    dup = pd.concat([panel, panel.head(3)])
    assert _raises(validate_panel, dup, cfg)
    neg = panel.copy(); neg.loc[neg.index[5], "inc"] = -1.0
    assert _raises(validate_panel, neg, cfg)
    gap = panel.drop(panel.index[100:105])
    assert _raises(validate_panel, gap, cfg)
    assert _raises(validate_raw, tmp_path / "nowhere")

    df = build_features(panel, cfg)
    feats = feature_columns(df)
    validate_features(df, feats)
    bad = df.copy(); bad.loc[bad.index[0], feats[0]] = np.inf
    assert _raises(validate_features, bad, feats)
    assert _raises(validate_inference_frame, df.drop(columns=[feats[0]]), feats)


def test_full_pipeline_registry_and_train_serve_parity(tmp_path):
    """Run every stage on a small synthetic world, then check that inference from the saved model
    reproduces the training-time live forecast exactly (no train/serve skew)."""
    from src.pipeline import stages

    cfg = Config(synthetic_days=560, synthetic_seed=11, min_total_cases=300, min_train_rows=500,
                 first_cutoff="2020-11-01", step_days=120, n_jobs=1,
                 raw_dir=str(tmp_path / "raw"), interim_dir=str(tmp_path / "interim"),
                 processed_dir=str(tmp_path / "proc"), models_dir=str(tmp_path / "models"),
                 artifacts_dir=str(tmp_path / "art"), log_dir=str(tmp_path / "logs"))
    stages.run_ingest(cfg, synthetic=True)
    stages.run_preprocess(cfg)
    stages.run_features(cfg)
    run_dir = stages.run_train(cfg)
    stages.run_predict(cfg, skip_seir=True)

    model, meta = load_run(cfg.models_dir)
    assert (run_dir / "model.joblib").exists() and meta["run_id"] == run_dir.name
    assert set(meta["features"]) <= set(pd.read_csv(tmp_path / "proc" / "features.csv.gz", nrows=1).columns)
    assert len(meta["feature_data_sha256"]) == 64

    pred = pd.read_csv(tmp_path / "art" / "predictions.csv.gz", parse_dates=["date"])
    fc = pd.read_csv(tmp_path / "art" / "forecast_latest.csv", parse_dates=["date"])
    last = pred[pred["date"] == pred["date"].max()].set_index("iso3").sort_index()
    fc = fc.set_index("iso3").sort_index()
    assert list(last.index) == list(fc.index)
    for col in ("pred_growth", "surge_prob", "q_lo", "q_hi"):
        assert np.allclose(last[col], fc[col], atol=1e-9), col
    assert fc["surge_prob"].between(0, 1).all()
    assert {"tier", "cluster", "model_run_id"} <= set(fc.columns)
    metrics = json.loads((tmp_path / "art" / "metrics.json").read_text())
    assert metrics["horizon_days"] == cfg.horizon
