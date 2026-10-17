"""Pipeline stages. Each stage reads the previous stage's persisted output and writes its own,
so any stage can be re-run independently:

    ingest -> data/raw  ->  preprocess -> data/interim/panel.csv.gz  ->  features ->
    data/processed/features.csv.gz  ->  train -> models/<run_id>/ + artifacts/  ->  predict -> artifacts/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.analysis.hotspots import detect_hotspots
from src.config import Config
from src.data.ingest import download_all
from src.data.preprocess import build_panel
from src.data.synthetic import write_synthetic
from src.data.validate import validate_features, validate_panel, validate_raw
from src.epi.transmission import fit_all
from src.features.build_features import build_features, feature_columns
from src.models.backtest import run_backtest
from src.models.evaluate import breakdown, eval_subset, evaluate, reliability_table
from src.models.model import risk_tier
from src.models.predict import score
from src.models.registry import file_sha256, load_run, save_run
from src.reporting.report import feature_importance, make_figures, save_metrics, write_results_md

log = logging.getLogger(__name__)

PANEL_FILE = "panel.csv.gz"
FEATURES_FILE = "features.csv.gz"
MANIFEST_FILE = "feature_manifest.json"


def _read_features(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    proc = Path(cfg.processed_dir)
    df = pd.read_csv(proc / FEATURES_FILE, parse_dates=["date"])
    manifest = json.loads((proc / MANIFEST_FILE).read_text())
    return df, manifest["features"]


# ------------------------------------------------------------------------------ 1. ingest
def run_ingest(cfg: Config, synthetic: bool = False, mobility: bool = False, force: bool = False):
    if synthetic:
        write_synthetic(cfg.raw_dir, days=cfg.synthetic_days, seed=cfg.synthetic_seed)
        log.info("wrote synthetic raw data to %s", cfg.raw_dir)
    else:
        download_all(cfg.raw_dir, mobility=mobility, force=force)
    validate_raw(cfg.raw_dir)


# ------------------------------------------------------------------------------ 2. preprocess
def run_preprocess(cfg: Config) -> Path:
    validate_raw(cfg.raw_dir)
    panel = build_panel(cfg)
    validate_panel(panel, cfg)
    out = Path(cfg.interim_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / PANEL_FILE, index=False)
    log.info("wrote %s (%d rows)", out / PANEL_FILE, len(panel))
    return out / PANEL_FILE


# ------------------------------------------------------------------------------ 3. features
def run_features(cfg: Config) -> Path:
    panel = pd.read_csv(Path(cfg.interim_dir) / PANEL_FILE, parse_dates=["date"])
    df = build_features(panel, cfg)
    features = feature_columns(df)
    validate_features(df, features)
    out = Path(cfg.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / FEATURES_FILE, index=False)
    manifest = {"features": features, "n_rows": len(df), "n_countries": int(df["iso3"].nunique()),
                "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
                "sha256": file_sha256(out / FEATURES_FILE)}
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    log.info("features: %d rows x %d features -> %s", len(df), len(features), out)
    return out / FEATURES_FILE


# ------------------------------------------------------------------------------ 4. train
def run_train(cfg: Config) -> Path:
    df, features = _read_features(cfg)
    art = Path(cfg.artifacts_dir)
    (art / "figures").mkdir(parents=True, exist_ok=True)

    pred, models = run_backtest(df, features, cfg)
    metrics = evaluate(pred, cfg)
    by_cont, by_fold = breakdown(pred, cfg, "continent"), breakdown(pred, cfg, "fold")
    importance = feature_importance(models, cfg, features)
    d = eval_subset(pred, cfg)
    calib = reliability_table(d["y_surge"].to_numpy(), d["surge_prob"].to_numpy())
    make_figures(pred, metrics, importance, cfg, art / "figures")

    as_of = df["date"].max()
    metrics["as_of"] = str(as_of.date())
    metrics["config"] = cfg.to_dict()
    pred["tier"] = risk_tier(pred["surge_prob"]).astype(str).to_numpy()
    pred.to_csv(art / "predictions.csv.gz", index=False)
    importance.to_csv(art / "feature_importance.csv", index=False)
    calib.to_csv(art / "calibration.csv", index=False)
    by_cont.to_csv(art / "metrics_by_continent.csv", index=False)
    by_fold.to_csv(art / "metrics_by_fold.csv", index=False)
    save_metrics(metrics, art / "metrics.json")
    write_results_md(metrics, by_cont, importance, art / "RESULTS.md")

    live_model = models[max(models)][0]            # last fold = trained on all labelled data
    sha = json.loads((Path(cfg.processed_dir) / MANIFEST_FILE).read_text())["sha256"]
    run = save_run(cfg.models_dir, live_model, features, cfg.to_dict(), metrics, sha, str(as_of.date()))
    (run / "metrics.json").write_text((art / "metrics.json").read_text())
    return run


# ------------------------------------------------------------------------------ 5. predict
def run_predict(cfg: Config, skip_seir: bool = False, as_of: str | None = None,
                run_id: str | None = None) -> Path:
    """Batch inference with a *registered* model (no training code involved)."""
    model, meta = load_run(cfg.models_dir, run_id)
    df, features = _read_features(cfg)
    if features != meta["features"]:
        raise RuntimeError("Feature set on disk differs from the one the model was trained on; "
                           "re-run `features` and `train`.")
    panel = pd.read_csv(Path(cfg.interim_dir) / PANEL_FILE, parse_dates=["date"])
    day = pd.Timestamp(as_of) if as_of else df["date"].max()
    rows = df[df["date"] == day]
    out = score(model, rows, features, cfg)

    coords = panel.groupby("iso3")[["lat", "lon"]].first()
    latest = out.join(coords, on="iso3")
    latest = detect_hotspots(latest, "surge_prob", k=cfg.n_neighbors)
    art = Path(cfg.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    if skip_seir:
        seir_p, seir_f = pd.DataFrame(), pd.DataFrame()
    else:
        seir_p, seir_f = fit_all(panel[panel["date"] <= day], cfg.seir_window, cfg.seir_horizon, cfg.n_jobs)
        if len(seir_p):
            latest = latest.merge(seir_p[["iso3", "r_eff"]].rename(columns={"r_eff": "seir_r_eff"}),
                                  on="iso3", how="left")
    latest["model_run_id"] = meta["run_id"]
    latest.to_csv(art / "forecast_latest.csv", index=False)
    seir_p.to_csv(art / "seir_params.csv", index=False)
    seir_f.to_csv(art / "seir_forecast.csv", index=False)
    history = panel[["iso3", "country", "continent", "date", "cum_cases", "new_smooth", "inc",
                     "population", "lat", "lon"]].merge(df[["iso3", "date", "rt_cori"]],
                                                        on=["iso3", "date"], how="left")
    history.to_csv(art / "history.csv.gz", index=False)
    log.info("forecast for %s written (%d countries, model %s)", day.date(), len(latest), meta["run_id"])
    return art / "forecast_latest.csv"
