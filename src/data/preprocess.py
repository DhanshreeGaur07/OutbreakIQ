"""Stage 2 - preprocessing: parse raw files, align names, clean counts, build the country-day panel."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.ingest import JHU_FILE, MOBILITY_FILE, OWID_FILE

log = logging.getLogger(__name__)

STATIC_COLS = [
    "population", "population_density", "median_age", "aged_65_older",
    "gdp_per_capita", "hospital_beds_per_thousand",
]
TV_COLS = ["stringency_index", "people_vaccinated_per_hundred", "new_tests_smoothed_per_thousand"]
MOBILITY_COLS = {
    "retail_and_recreation_percent_change_from_baseline": "mob_retail",
    "transit_stations_percent_change_from_baseline": "mob_transit",
    "workplaces_percent_change_from_baseline": "mob_work",
    "residential_percent_change_from_baseline": "mob_home",
}

# JHU country names -> candidate OWID location names (first hit wins)
NAME_ALTERNATES = {
    "US": ["United States"], "Korea, South": ["South Korea"], "Taiwan*": ["Taiwan"],
    "Burma": ["Myanmar"], "Congo (Kinshasa)": ["Democratic Republic of Congo"],
    "Congo (Brazzaville)": ["Congo"], "Cabo Verde": ["Cape Verde"],
    "West Bank and Gaza": ["Palestine"], "Holy See": ["Vatican"], "Timor-Leste": ["Timor"],
    "Turkey": ["Turkiye"], "Czechia": ["Czech Republic"], "Micronesia": ["Micronesia (country)"],
    "Eswatini": ["Swaziland"], "North Macedonia": ["Macedonia"],
}




# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def load_jhu(path: str | Path) -> pd.DataFrame:
    """JHU wide cumulative-confirmed file -> long frame [country, date, cum_cases, lat, lon].

    Province rows are summed to country level; coordinates use the national row when it
    exists (so overseas territories don't drag the centroid) and the provincial median otherwise.
    """
    raw = pd.read_csv(path)
    id_cols = ["Province/State", "Country/Region", "Lat", "Long"]
    date_cols = [c for c in raw.columns if c not in id_cols]

    cum = raw.groupby("Country/Region")[date_cols].sum()
    cum.columns = pd.to_datetime(cum.columns, format="%m/%d/%y")
    cum = cum.copy()

    med = raw.groupby("Country/Region")[["Lat", "Long"]].median()
    nat = raw[raw["Province/State"].isna()].drop_duplicates("Country/Region")
    nat = nat.set_index("Country/Region")[["Lat", "Long"]]
    coords = med.copy()
    common = nat.index.intersection(coords.index)
    coords.loc[common, ["Lat", "Long"]] = nat.loc[common, ["Lat", "Long"]]

    long = cum.rename_axis("country").reset_index().melt(
        id_vars="country", var_name="date", value_name="cum_cases")
    long["date"] = pd.to_datetime(long["date"])
    coords = coords.rename(columns={"Lat": "lat", "Long": "lon"}).rename_axis("country")
    return long.merge(coords.reset_index(), on="country", how="left")


def load_owid(path: str | Path) -> pd.DataFrame:
    wanted = ["iso_code", "continent", "location", "date", *STATIC_COLS, *TV_COLS]
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in wanted if c in header]
    df = pd.read_csv(path, usecols=cols, parse_dates=["date"])
    df = df[df["iso_code"].astype(str).str.len().eq(3) & ~df["iso_code"].astype(str).str.startswith("OWID")]
    return df


def load_mobility(path: str | Path, locations: set[str]) -> pd.DataFrame:
    """Country-level Google mobility (rows without sub-regions), smoothed with a 7-day mean."""
    keep = ["country_region", "sub_region_1", "metro_area", "iso_3166_2_code", "census_fips_code",
            "date", *MOBILITY_COLS]
    parts = []
    for chunk in pd.read_csv(path, chunksize=500_000, parse_dates=["date"],
                             usecols=lambda c: c in keep, low_memory=False):
        mask = chunk["sub_region_1"].isna() & chunk["country_region"].isin(locations)
        if "metro_area" in chunk:
            mask &= chunk["metro_area"].isna()
        parts.append(chunk[mask])
    mob = pd.concat(parts, ignore_index=True).rename(columns=MOBILITY_COLS)
    mob = mob.rename(columns={"country_region": "location"})
    cols = list(MOBILITY_COLS.values())
    mob = mob[["location", "date", *cols]].sort_values(["location", "date"])
    mob[cols] = mob.groupby("location")[cols].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return mob


def _resolve_location(name: str, owid_names: set[str]) -> str | None:
    for cand in [name, *NAME_ALTERNATES.get(name, [])]:
        if cand in owid_names:
            return cand
    return None


# --------------------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------------------
def build_panel(cfg: Config) -> pd.DataFrame:
    """Country x day panel with cleaned incidence, demographics and optional covariates."""
    raw = Path(cfg.raw_dir)
    jhu = load_jhu(raw / JHU_FILE)
    owid = load_owid(raw / OWID_FILE)

    static_cols = [c for c in STATIC_COLS if c in owid]
    tv_cols = [c for c in TV_COLS if c in owid]
    loc = owid.groupby("iso_code").agg(
        location=("location", "last"), continent=("continent", "last"),
        **{c: (c, "last") for c in static_cols},
    ).reset_index().rename(columns={"iso_code": "iso3"})

    names = set(loc["location"])
    mapping = {c: _resolve_location(c, names) for c in jhu["country"].unique()}
    unmatched = sorted(c for c, m in mapping.items() if m is None)
    if unmatched:
        log.info("JHU entries without an OWID match (dropped): %s", unmatched)
    jhu["location"] = jhu["country"].map(mapping)
    panel = jhu.dropna(subset=["location"]).merge(loc, on="location", how="inner")

    if tv_cols:
        tv = owid[["iso_code", "date", *tv_cols]].rename(columns={"iso_code": "iso3"})
        panel = panel.merge(tv, on=["iso3", "date"], how="left")

    if cfg.use_mobility and (raw / MOBILITY_FILE).exists():
        mob = load_mobility(raw / MOBILITY_FILE, set(panel["location"]))
        panel = panel.merge(mob, on=["location", "date"], how="left")
    elif cfg.use_mobility:
        log.warning("use_mobility=true but %s not found; run `outbreakiq download --mobility`",
                    raw / MOBILITY_FILE)

    panel = panel.sort_values(["iso3", "date"]).reset_index(drop=True)
    # cumulative counts must be monotone; JHU occasionally revises downward
    panel["cum_cases"] = panel.groupby("iso3")["cum_cases"].cummax()
    daily = panel.groupby("iso3")["cum_cases"].diff()
    panel["new_cases"] = daily.fillna(panel["cum_cases"]).clip(lower=0)
    panel["new_smooth"] = panel.groupby("iso3")["new_cases"].transform(
        lambda s: s.rolling(cfg.smooth_window, min_periods=1).mean())

    panel = panel[panel["population"].fillna(0) >= cfg.min_population]
    big = panel.groupby("iso3")["cum_cases"].transform("max") >= cfg.min_total_cases
    panel = panel[big].copy()

    panel["inc"] = panel["new_smooth"] / panel["population"] * 1e6      # cases / million / day
    panel["cum_pm"] = panel["cum_cases"] / panel["population"] * 1e6
    panel["active"] = panel["cum_cases"] >= cfg.start_cases
    panel = panel.rename(columns={"country": "country"}).reset_index(drop=True)
    log.info("panel: %d countries, %s -> %s", panel["iso3"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())
    return panel
