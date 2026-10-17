"""OutbreakIQ interactive dashboard.   streamlit run app/dashboard.py

Reads the artifacts written by `python main.py all` (or `python main.py all --synthetic`).
Point it elsewhere with  OUTBREAKIQ_ARTIFACTS=artifacts/synthetic streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, no packaging needed
from src.epi.transmission import SeirFit, seir_forecast  # noqa: E402
from src.models.model import TIERS  # noqa: E402

st.set_page_config(page_title="OutbreakIQ", page_icon="🦠", layout="wide")

TIER_COLORS = {"Low": "#2a9d8f", "Guarded": "#8ab17d", "Elevated": "#e9c46a",
               "High": "#f4a261", "Critical": "#d1495b"}
TIER_ORDER = [t[1] for t in TIERS]
ART = Path(os.environ.get("OUTBREAKIQ_ARTIFACTS", "artifacts"))


@st.cache_data(show_spinner=False)
def load(art: str):
    a = Path(art)
    need = ["predictions.csv.gz", "history.csv.gz", "forecast_latest.csv", "metrics.json"]
    if not all((a / n).exists() for n in need):
        return None
    out = {
        "pred": pd.read_csv(a / "predictions.csv.gz", parse_dates=["date"]),
        "hist": pd.read_csv(a / "history.csv.gz", parse_dates=["date"]),
        "latest": pd.read_csv(a / "forecast_latest.csv", parse_dates=["date"]),
        "metrics": json.loads((a / "metrics.json").read_text()),
    }
    for name, f in (("seir_p", "seir_params.csv"), ("calib", "calibration.csv"),
                    ("imp", "feature_importance.csv"), ("cont", "metrics_by_continent.csv")):
        try:
            out[name] = pd.read_csv(a / f)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            out[name] = pd.DataFrame()
    return out


def add_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_change_pct"] = ((1 + df["pred_inc"]) / (1 + df["inc"]) - 1) * 100
    return df


data = load(str(ART))
st.title("🦠 OutbreakIQ — epidemic spread prediction")
if data is None:
    st.error(f"No artifacts found in `{ART}`. Run `python main.py all` "
             "(or `python main.py all --synthetic` for an offline demo) first.")
    st.stop()

pred, hist, latest, metrics = data["pred"], data["hist"], data["latest"], data["metrics"]
h = metrics["horizon_days"]
as_of = pd.Timestamp(metrics["as_of"])
if "synthetic" in str(ART):
    st.warning("Showing SYNTHETIC demo data — not real epidemiology.")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Controls")
dates = sorted(pred["date"].dt.date.unique())
sel_date = st.sidebar.select_slider("Forecast issued on", options=dates, value=dates[-1])
continents = sorted(pred["continent"].dropna().unique())
sel_cont = st.sidebar.multiselect("Continents", continents, default=continents)
metric_label = st.sidebar.radio("Map colour", ["Surge probability", "Predicted change (%)", "Rt (Cori)"])
st.sidebar.caption(f"Latest data: {as_of.date()} · horizon: {h} days")

day = add_change(pred[(pred["date"].dt.date == sel_date) & pred["continent"].isin(sel_cont)])
is_live = bool(day["is_live"].any()) if len(day) else False

tab_map, tab_country, tab_hot, tab_model, tab_about = st.tabs(
    ["🗺️ Risk map", "📈 Country explorer", "🔥 Hotspots", "🧪 Model performance", "ℹ️ About"])

# ------------------------------------------------------------------ risk map
with tab_map:
    st.caption(("**Live forecast** — outcome not yet observed." if is_live else
                "**Historical forecast** — produced by a model that had not seen this period "
                "(walk-forward backtest, out-of-sample).")
               + f" Target date: {(pd.Timestamp(sel_date) + pd.Timedelta(days=h)).date()}")
    col, title, scale, rng = {
        "Surge probability": ("surge_prob", "P(surge)", "YlOrRd", (0, 1)),
        "Predicted change (%)": ("pred_change_pct", f"Predicted {h}-day change (%)", "RdYlGn_r", (-100, 200)),
        "Rt (Cori)": ("rt_cori", "Rt", "RdYlGn_r", (0.5, 1.8)),
    }[metric_label]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", len(day))
    c2.metric("Critical / High", int(day["tier"].isin(["Critical", "High"]).sum()))
    c3.metric("Median Rt", f"{day['rt_cori'].median():.2f}" if len(day) else "–")
    c4.metric("Mean P(surge)", f"{day['surge_prob'].mean():.0%}" if len(day) else "–")
    fig = px.choropleth(day, locations="iso3", color=col, hover_name="country",
                        hover_data={"iso3": False, "tier": True, "surge_prob": ":.0%",
                                    "inc": ":.1f", "pred_inc": ":.1f", "rt_cori": ":.2f"},
                        color_continuous_scale=scale, range_color=rng, labels={col: title},
                        projection="natural earth")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520)
    st.plotly_chart(fig)
    st.subheader("Highest-risk countries")
    top = day.sort_values("surge_prob", ascending=False).head(15)[
        ["country", "tier", "surge_prob", "inc", "pred_inc", "pred_inc_lo", "pred_inc_hi", "rt_cori"]]
    top.columns = ["Country", "Tier", "P(surge)", "Cases/M/day now", f"Forecast (+{h}d)",
                   "80% low", "80% high", "Rt"]
    st.dataframe(top.round(2), hide_index=True)

# ------------------------------------------------------------------ country explorer
with tab_country:
    names = sorted(hist["country"].unique())
    default = names.index("India") if "India" in names else 0
    country = st.selectbox("Country", names, index=default)
    iso = hist.loc[hist["country"] == country, "iso3"].iloc[0]
    hc = hist[hist["iso3"] == iso].sort_values("date")
    pc = pred[pred["iso3"] == iso].sort_values("date")
    lat = latest[latest["iso3"] == iso]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cases / M / day", f"{hc['inc'].iloc[-1]:.1f}")
    k2.metric("Rt (Cori)", f"{hc['rt_cori'].iloc[-1]:.2f}" if hc["rt_cori"].notna().any() else "–")
    if len(lat):
        k3.metric("Risk tier", str(lat["tier"].iloc[0]))
        k4.metric("P(surge)", f"{lat['surge_prob'].iloc[0]:.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hc["date"], y=hc["inc"], name="Observed (7-day avg)",
                             line=dict(color="#222", width=2)))
    if len(pc):
        x = pc["date"] + pd.Timedelta(days=h)
        fig.add_trace(go.Scatter(x=x, y=pc["pred_inc_hi"], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=pc["pred_inc_lo"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(209,73,91,0.2)", name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=pc["pred_inc"], name=f"{h}-day-ahead forecast",
                                 line=dict(color="#d1495b", width=1.5)))
    sp = data["seir_p"]
    row = sp[sp["iso3"] == iso] if len(sp) else sp
    if len(row):
        st.markdown("**Transmission scenario (SEIR)** — re-simulate from today's fitted state")
        change = st.slider("Change in transmission rate vs. now (%)", -60, 60, 0, step=5)
        r = row.iloc[0]
        fit = SeirFit(beta=r["beta"], rho=r["rho"], r_eff=r["r_eff"], rmse_log=r["rmse_log"],
                      state_end=(r["S"], r["E"], r["I"], r["R"]), n=r["population"])
        days = 28
        fc = seir_forecast(fit, days, 1 + change / 100) / r["population"] * 1e6
        fx = pd.date_range(pd.Timestamp(r["as_of"]) + pd.Timedelta(days=1), periods=days)
        fig.add_trace(go.Scatter(x=fx, y=fc, name=f"SEIR scenario ({change:+d}% transmission)",
                                 line=dict(color="#2e86ab", dash="dash", width=2)))
        k3.caption(f"SEIR R_eff: {r['r_eff']:.2f}")
    fig.update_layout(height=460, yaxis_title="cases / million / day", legend=dict(orientation="h"),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)

# ------------------------------------------------------------------ hotspots
with tab_hot:
    st.markdown("Spatial clusters of high predicted surge risk, found with **Local Moran's I** "
                "(5 nearest neighbours, 999 permutations, p < 0.05). *HH* = hotspot, *LL* = coldspot, "
                "*HL/LH* = spatial outliers.")
    lh = latest.copy()
    lh["cluster_label"] = lh["cluster"].replace({"ns": "not significant"})
    fig = px.scatter_geo(lh, lat="lat", lon="lon", color="cluster_label", size="surge_prob",
                         hover_name="country", projection="natural earth",
                         color_discrete_map={"HH": "#d1495b", "LL": "#2e86ab", "HL": "#f4a261",
                                             "LH": "#8ab17d", "not significant": "#c7ccd1"})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=480)
    st.plotly_chart(fig)
    hh = lh[lh["cluster"] == "HH"].sort_values("surge_prob", ascending=False)
    if len(hh):
        st.dataframe(hh[["country", "tier", "surge_prob", "p_value"]].round(3), hide_index=True)
    else:
        st.info("No statistically significant high-high clusters at the latest date.")

# ------------------------------------------------------------------ model performance
with tab_model:
    reg, clf, iv = metrics["regression"], metrics["surge_classification"], metrics["interval"]
    st.markdown(f"Walk-forward backtest · **{metrics['n_folds']} folds** · {metrics['n_countries']} countries · "
                f"{metrics['n_rows']:,} out-of-sample country-days · {metrics['date_range'][0]} → "
                f"{metrics['date_range'][1]}")
    m1, m2, m3 = st.columns(3)
    g = reg["gradient_boosting"]
    m1.metric(f"MAE ({h}-day log-growth)", f"{g['mae']:.3f}",
              f"{g['skill_vs_persistence']:+.0%} vs persistence")
    m2.metric("AUROC (surge, fold mean)", f"{clf['conformal_distributional']['auroc_fold_mean']:.3f}")
    m3.metric("80% interval coverage", f"{iv['coverage']:.0%}")
    st.dataframe(pd.DataFrame(reg).T.round(3))
    st.dataframe(pd.DataFrame(clf).T.round(3))
    left, right = st.columns(2)
    if len(data["calib"]):
        f = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="perfect"),
                       go.Scatter(x=data["calib"]["mean_pred"], y=data["calib"]["observed"],
                                  mode="lines+markers", name="model")])
        f.update_layout(title="Reliability of P(surge)", xaxis_title="predicted", yaxis_title="observed", height=380)
        left.plotly_chart(f)
    if len(data["imp"]):
        top = data["imp"].head(12).iloc[::-1]
        f = px.bar(top, x="importance", y="feature", orientation="h", title="Permutation importance")
        f.update_layout(height=380)
        right.plotly_chart(f)
    if len(data["cont"]):
        st.subheader("By continent")
        st.dataframe(data["cont"].round(3), hide_index=True)

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
**Task.** For every country-day, forecast the {h}-day-ahead change in COVID-19 incidence and the
probability of a *surge* (incidence ≥ {metrics['config']['surge_ratio']:.2f}× today **and**
≥ {metrics['config']['surge_min_level']:.0f} cases per million per day).

**Data.** JHU CSSE confirmed cases (primary), Our World in Data covariates (secondary: stringency,
vaccination, testing, demographics) and optionally Google Community Mobility.

**Method.** Leakage-safe features (momentum, acceleration, Cori Rt, spatial importation pressure from
neighbouring countries, global momentum, policy / vaccination / demographics) → gradient boosting that
learns a *correction to a damped-trend baseline*, with shrinkage chosen on a held-out calibration window,
conformalised 80% intervals and a conformal predictive distribution for surge probability. Transmission
is modelled separately with a fitted SEIR model that supports what-if scenarios. Hotspots use Local
Moran's I.

**Limitations.** JHU stopped daily updates in March 2023; reported cases depend on testing and reporting
practice; country-level aggregation hides sub-national outbreaks; this is a research / portfolio project
and not a public-health decision tool.
""")
