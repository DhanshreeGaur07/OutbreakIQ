"""Stage 1 - ingestion: fetch raw files (JHU CSSE, Our World in Data, optional Google Mobility)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/"
    "csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
)
OWID_URLS = [
    "https://covid.ourworldindata.org/data/owid-covid-data.csv",
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
]
MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

JHU_FILE = "jhu_confirmed_global.csv"
OWID_FILE = "owid-covid-data.csv"
MOBILITY_FILE = "Global_Mobility_Report.csv"

def download(url: str, dest: str | Path, retries: int = 3, timeout: int = 180,
             force: bool = False) -> Path:
    """Stream a file to disk with retries; cached unless ``force``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("cached: %s", dest)
        return dest
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
                log.info("downloaded %s", dest)
                return dest
        except requests.RequestException as exc:
            last = exc
            log.warning("download attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def download_all(raw_dir: str | Path, mobility: bool = False, force: bool = False) -> None:
    raw = Path(raw_dir)
    download(JHU_URL, raw / JHU_FILE, force=force)
    err = None
    for url in OWID_URLS:
        try:
            download(url, raw / OWID_FILE, force=force)
            err = None
            break
        except RuntimeError as exc:
            err = exc
    if err:
        raise err
    if mobility:
        download(MOBILITY_URL, raw / MOBILITY_FILE, force=force)


