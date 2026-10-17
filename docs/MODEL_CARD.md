# Model card — OutbreakIQ

**Intended use:** research / education; ranking countries by short-term COVID-19 surge risk.
**Not for:** operational public-health decisions, sub-national or individual-level inference.

**Targets.** `y_growth = log1p(inc[t+14]) − log1p(inc[t])` (inc = 7-day mean cases / million / day);
`y_surge = inc[t+14] ≥ max(1.25·inc[t], 5)`.

**Data.** JHU CSSE global confirmed (primary); OWID stringency, vaccination, testing, demographics
(secondary); Google Mobility (optional). Countries need ≥ 1M population and ≥ 10k total cases.

**Validation.** Expanding-window walk-forward, 90-day refit cadence (config: `step_days`), 14-day embargo, 120-day
calibration window. Metrics are computed where incidence ≥ 1 case / million / day.

**Known failure modes.** Regime changes (new variants, policy shifts, reporting changes) degrade
accuracy until the next refit; very low-incidence countries are noisy; reporting artefacts
(backlog dumps, weekend effects) are only partly smoothed; with < ~2 years of history the ML correction
cannot reliably beat the damped-trend baseline (the shrinkage guard mitigates but does not eliminate this).

**Fairness / bias.** Under-testing countries appear lower-risk than they are; demographic features are
country-level proxies and may encode reporting capacity rather than biology.
