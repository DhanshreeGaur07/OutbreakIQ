"""Synthetic stand-in for JHU + OWID, written in the *real* file formats.

Purpose: CI, unit tests and an offline demo. It simulates coupled multi-wave SEIR epidemics with
policy feedback (stringency reacts to incidence, transmission reacts to stringency), global
variant waves, vaccination, weekday reporting artefacts and noise. Not for scientific claims.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# iso3, JHU name, OWID name, continent, lat, lon, pop (millions), median age, GDP pc, density
COUNTRIES = [
    ("USA", "US", "United States", "North America", 37.1, -95.7, 331, 38.3, 54225, 36),
    ("CAN", "Canada", "Canada", "North America", 56.1, -106.3, 38, 41.4, 44017, 4),
    ("MEX", "Mexico", "Mexico", "North America", 23.6, -102.5, 128, 29.3, 17336, 66),
    ("BRA", "Brazil", "Brazil", "South America", -14.2, -51.9, 213, 33.5, 14103, 25),
    ("ARG", "Argentina", "Argentina", "South America", -38.4, -63.6, 45, 31.9, 18934, 17),
    ("COL", "Colombia", "Colombia", "South America", 4.6, -74.3, 51, 31.2, 13255, 44),
    ("PER", "Peru", "Peru", "South America", -9.2, -75.0, 33, 29.1, 12236, 25),
    ("CHL", "Chile", "Chile", "South America", -35.7, -71.5, 19, 35.4, 22767, 24),
    ("GBR", "United Kingdom", "United Kingdom", "Europe", 55.4, -3.4, 68, 40.8, 39753, 272),
    ("FRA", "France", "France", "Europe", 46.2, 2.2, 65, 42.0, 38606, 123),
    ("DEU", "Germany", "Germany", "Europe", 51.2, 10.5, 83, 46.6, 45229, 238),
    ("ITA", "Italy", "Italy", "Europe", 41.9, 12.6, 60, 47.9, 35220, 206),
    ("ESP", "Spain", "Spain", "Europe", 40.5, -3.7, 47, 45.5, 34272, 94),
    ("POL", "Poland", "Poland", "Europe", 51.9, 19.1, 38, 41.8, 27216, 124),
    ("UKR", "Ukraine", "Ukraine", "Europe", 48.4, 31.2, 44, 41.4, 7894, 77),
    ("TUR", "Turkey", "Turkey", "Asia", 38.9, 35.2, 85, 31.6, 25129, 104),
    ("CZE", "Czechia", "Czechia", "Europe", 49.8, 15.5, 11, 43.3, 32606, 137),
    ("SWE", "Sweden", "Sweden", "Europe", 60.1, 18.6, 10, 41.0, 46949, 24),
    ("RUS", "Russia", "Russia", "Europe", 61.5, 105.3, 146, 39.6, 24766, 9),
    ("IND", "India", "India", "Asia", 20.6, 78.9, 1380, 28.2, 6427, 450),
    ("PAK", "Pakistan", "Pakistan", "Asia", 30.4, 69.3, 221, 23.5, 5034, 255),
    ("BGD", "Bangladesh", "Bangladesh", "Asia", 23.7, 90.4, 165, 27.5, 3524, 1265),
    ("IDN", "Indonesia", "Indonesia", "Asia", -0.8, 113.9, 274, 29.7, 11189, 145),
    ("PHL", "Philippines", "Philippines", "Asia", 12.9, 121.8, 110, 25.2, 7599, 352),
    ("VNM", "Vietnam", "Vietnam", "Asia", 14.1, 108.3, 97, 32.6, 6172, 308),
    ("THA", "Thailand", "Thailand", "Asia", 15.9, 101.0, 70, 40.1, 16278, 135),
    ("JPN", "Japan", "Japan", "Asia", 36.2, 138.3, 126, 48.2, 39002, 347),
    ("KOR", "Korea, South", "South Korea", "Asia", 35.9, 127.8, 51, 43.4, 35938, 528),
    ("IRN", "Iran", "Iran", "Asia", 32.4, 53.7, 84, 32.4, 19083, 50),
    ("SAU", "Saudi Arabia", "Saudi Arabia", "Asia", 23.9, 45.1, 35, 31.9, 49045, 16),
    ("EGY", "Egypt", "Egypt", "Africa", 26.8, 30.8, 102, 25.3, 10551, 98),
    ("NGA", "Nigeria", "Nigeria", "Africa", 9.1, 8.7, 206, 18.1, 5338, 209),
    ("ETH", "Ethiopia", "Ethiopia", "Africa", 9.1, 40.5, 115, 19.5, 1730, 105),
    ("ZAF", "South Africa", "South Africa", "Africa", -30.6, 22.9, 59, 27.3, 12295, 46),
    ("KEN", "Kenya", "Kenya", "Africa", -0.0, 37.9, 54, 20.1, 2993, 82),
    ("MAR", "Morocco", "Morocco", "Africa", 31.8, -7.1, 37, 29.5, 7485, 80),
    ("COD", "Congo (Kinshasa)", "Democratic Republic of Congo", "Africa", -4.0, 21.8, 90, 17.0, 808, 40),
    ("AUS", "Australia", "Australia", "Oceania", -25.3, 133.8, 26, 37.9, 44649, 3),
    ("NZL", "New Zealand", "New Zealand", "Oceania", -40.9, 174.9, 5, 37.9, 36086, 18),
    ("CHN", "China", "China", "Asia", 35.9, 104.2, 1440, 38.7, 15309, 147),
]
WAVES = [(300, 0.9), (480, 0.7), (640, 1.6), (700, 1.2), (900, 0.6)]   # global variant pushes


def _simulate_country(rng, pop, days, seed_day, beta0, waves, rho0):
    s, e, i, r = pop, 0.0, 0.0, 0.0
    sigma, gamma, waning = 1 / 5.2, 1 / 7.0, 1 / 220
    new_i = np.zeros(days)
    stringency = np.zeros(days)
    vacc = np.zeros(days)
    for t in range(days):
        if t == seed_day:
            e += 25.0
            s -= 25.0
        recent = new_i[max(0, t - 7):t].mean() if t else 0.0
        inc_pm = recent / pop * 1e6 * rho0
        stringency[t] = 15 + 65 / (1 + np.exp(-(np.log1p(inc_pm) - 3.2) * 1.3))
        vacc[t] = np.clip((t - 340) / 320, 0, 1) * 0.7
        wave = sum(m * np.exp(-0.5 * ((t - d) / 22) ** 2) for d, m in waves)
        beta = (beta0 * (1 + 0.22 * np.cos(2 * np.pi * (t - 20) / 365)) * (1 + wave)
                * (1 - 0.45 * stringency[t - 7 if t >= 7 else 0] / 100) * (1 - 0.45 * vacc[t]))
        flow = min(beta * s * i / pop * rng.lognormal(0, 0.06), s)
        out_e, out_i, wane = sigma * e, gamma * i, waning * r
        s += -flow + wane
        e += flow - out_e
        i += out_e - out_i
        r += out_i - wane
        new_i[t] = out_e
    weekday = np.array([1.1, 1.15, 1.1, 1.05, 1.0, 0.8, 0.8])[np.arange(days) % 7]
    rho = rho0 * (1 - 0.3 * np.clip((np.arange(days) - 700) / 400, 0, 1))   # testing declines
    reported = rng.poisson(np.maximum(new_i * rho * weekday, 0) * rng.lognormal(0, 0.12, days))
    return reported, stringency, vacc


def write_synthetic(raw_dir: str | Path, days: int = 1100, seed: int = 7,
                    start: str = "2020-01-22") -> Path:
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, periods=days)

    # crude spatial delay: countries closer to the epicentre are seeded earlier
    epi = np.array([30.6, 114.3])
    jhu_rows, owid_rows = [], []
    for iso, jname, oname, cont, lat, lon, popm, age, gdp, dens in COUNTRIES:
        pop = popm * 1e6
        dist = np.hypot(lat - epi[0], (lon - epi[1]) * np.cos(np.radians(lat))) * 111
        seed_day = int(np.clip(20 + dist / 250 + rng.integers(0, 15), 5, 120))
        beta0 = rng.uniform(0.32, 0.52)
        rho0 = rng.uniform(0.08, 0.5)
        waves = [(d + rng.integers(-25, 25), m * rng.uniform(0.6, 1.4)) for d, m in WAVES]
        rep, stringency, vacc = _simulate_country(rng, pop, days, seed_day, beta0, waves, rho0)
        cum = np.cumsum(rep)
        tests = np.clip(rep / 0.08 / pop * 1000, 0, None)
        if jname == "China":   # exercise province aggregation in the loader
            for prov, frac, dla in (("Hubei", 0.6, 0.0), ("Guangdong", 0.4, -8.0)):
                jhu_rows.append((prov, jname, lat + dla, lon, (cum * frac).round().astype(int)))
        else:
            jhu_rows.append((np.nan, jname, lat, lon, cum.astype(int)))
        df = pd.DataFrame({
            "iso_code": iso, "continent": cont, "location": oname, "date": dates,
            "population": pop, "population_density": dens, "median_age": age,
            "aged_65_older": age * 0.45, "gdp_per_capita": gdp, "hospital_beds_per_thousand": 3.0,
            "stringency_index": stringency,
            "people_vaccinated_per_hundred": vacc * 100,
            "new_tests_smoothed_per_thousand": tests,
        })
        owid_rows.append(df)
    # junk rows the loaders must drop
    jhu_rows.append((np.nan, "Diamond Princess", 0.0, 0.0, np.cumsum(np.ones(days, dtype=int))))
    jhu_rows.append((np.nan, "Summer Olympics 2020", 35.7, 139.7, np.zeros(days, dtype=int)))

    cols = [f"{d.month}/{d.day}/{d:%y}" for d in dates]
    jhu = pd.DataFrame([[p, c, la, lo, *v] for p, c, la, lo, v in jhu_rows],
                       columns=["Province/State", "Country/Region", "Lat", "Long", *cols])
    jhu.to_csv(raw / "jhu_confirmed_global.csv", index=False)

    owid = pd.concat(owid_rows, ignore_index=True)
    agg = owid.groupby("date", as_index=False)[["population"]].sum()
    agg["iso_code"], agg["location"], agg["continent"] = "OWID_WRL", "World", np.nan
    pd.concat([owid, agg], ignore_index=True).to_csv(raw / "owid-covid-data.csv", index=False)
    return raw
