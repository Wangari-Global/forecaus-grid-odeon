"""RTE eCO2mix ingestion (fallback / cross-check to ENTSO-E load).

Pulls French national consumption / nuclear generation / CO2 intensity from the
public ODRE opendatasoft dataset ``eco2mix-national-cons-def`` (keyless HTTP
API). Used to cross-check the ENTSO-E load series. Falls back to the committed
offline sample when there is no network.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .. import config
from ._io import cached, log

_ODRE_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "eco2mix-national-cons-def/exports/json"
)


def fetch_eco2mix(region: str = config.BIDDING_ZONE,
                  start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Hourly consumption [MW], nuclear [MW] and CO2 intensity [g/kWh] from RTE.

    Cached to data/raw/rte_eco2mix.parquet.
    """
    def download():
        try:
            import requests
        except ImportError:
            return None
        s = pd.Timestamp(start or config.INGEST_START).date()
        e = pd.Timestamp(end or config.INGEST_END).date()
        params = {
            "select": "date_heure,consommation,nucleaire,taux_co2",
            "where": f'date_heure >= "{s}" and date_heure < "{e}"',
            "limit": -1,
            "timezone": "UTC",
        }
        log(f"RTE eco2mix: GET {_ODRE_URL}")
        resp = requests.get(_ODRE_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "date_heure": "time",
            "consommation": "consumption_mw",
            "nucleaire": "nuclear_mw",
            "taux_co2": "co2_rate_g_per_kwh",
        }).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        # eco2mix is published at 15-min resolution -> aggregate to hourly.
        return df.apply(pd.to_numeric, errors="coerce").resample("h").mean()

    return cached("rte_eco2mix", download)
