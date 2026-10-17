# OutbreakIQ — Epidemic Spread Prediction (Epidemiology + AI)

Hackathon Track C. An end-to-end, reproducible ML pipeline that forecasts COVID-19 outbreak risk per
country, detects spatial hotspots and serves an interactive risk dashboard.
Data: JHU CSSE (primary), Our World in Data (secondary), Google Mobility (optional).

**Deliverables:** GitHub repo ✔ · outbreak prediction model ✔ · interactive dashboard ✔ · risk map ✔

## Pipeline
```
ingest ─▶ preprocess ─▶ features ─▶ train ─▶ predict ─▶ dashboard
data/raw   data/interim   data/processed   models/<run_id>/   artifacts/
           panel.csv.gz   features.csv.gz  + artifacts/       forecast_latest.csv ...
```
Every stage reads the previous stage's persisted output, validates its data contract, and can be re-run on
its own. Training and inference are separate code paths: `predict` loads a **registered model** from
`models/<run_id>/` (immutable, with metadata: features, config, data hash, library versions, headline
metrics) and scores the feature table. A test asserts inference reproduces the training-time forecast exactly.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py all                  # ingest -> preprocess -> features -> train -> predict
streamlit run app/dashboard.py

python main.py train                # run one stage
python main.py predict --as-of 2022-03-01
python main.py all --synthetic      # offline demo on simulated data (isolated paths)
OUTBREAKIQ_ARTIFACTS=artifacts/synthetic streamlit run app/dashboard.py
```
Docker: `docker compose run --rm pipeline && docker compose up dashboard` → http://localhost:8501

## Modelling
| Component | Approach |
|---|---|
| Outbreak prediction | 14-day-ahead log-change in incidence. Gradient boosting learns a *correction to a damped-trend baseline*; a shrinkage weight chosen on a held-out window falls back to the baseline if the correction stops generalising. |
| Uncertainty | 80% intervals by conformalised quantile regression; surge probability from a normalised conformal predictive distribution. |
| Transmission | Causal Cori Rt feature; per-country SEIR with under-reporting for what-if scenarios. |
| External factors | Stringency, vaccination, testing, demographics, optional mobility, spatial importation pressure from neighbouring countries. |
| Hotspots | Local Moran's I (LISA) with permutation p-values. |

## Engineering practices
- **Leakage control:** trailing-only features (unit-tested by recomputing on truncated history), walk-forward folds with a 14-day embargo; every historical dashboard forecast is out-of-sample.
- **Data contracts** (`src/data/validate.py`): schema, duplicates, negatives, date gaps, infinities, inference-frame columns.
- **Reproducibility:** one typed config (`config/config.yaml`), fixed seeds, pinned dependency ranges, data hash stored with every model.
- **Observability:** console + rotating file logs (`logs/pipeline.log`), per-stage timings, non-zero exit code on failure.
- **Quality gates:** 12 tests (`pytest -q`), `ruff`, pre-commit, GitHub Actions CI (3.10 + 3.12, plus an end-to-end synthetic run), non-root Docker image with healthcheck.

## Layout
```
main.py                   CLI entry point (stages)
config/config.yaml        configuration
src/data/                 ingest.py  preprocess.py  validate.py  synthetic.py
src/features/             build_features.py
src/models/               model.py  backtest.py  evaluate.py  registry.py  predict.py
src/epi/transmission.py   Cori Rt + SEIR          src/analysis/hotspots.py   LISA
src/pipeline/stages.py    stage runners           src/reporting/report.py    figures + RESULTS.md
app/dashboard.py          Streamlit dashboard     tests/   docs/MODEL_CARD.md
```

## Results
`python main.py train` writes `artifacts/RESULTS.md` and `artifacts/figures/*.png`. Paste them here:

<!-- RESULTS: paste artifacts/RESULTS.md and embed artifacts/figures/*.png -->

## Limitations
JHU stopped daily updates in March 2023; reported cases reflect testing behaviour; country-level
aggregation hides sub-national outbreaks; the SEIR fit is a local approximation; Google Mobility ended in
2022. A research project, not a decision tool — see `docs/MODEL_CARD.md`.
