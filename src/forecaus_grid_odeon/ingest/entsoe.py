"""ENTSO-E ingestion: load, generation mix, day-ahead prices.

Uses the `entsoe-py` client against the ENTSO-E Transparency Platform.
A free API token is required for live downloads (export ``ENTSOE_API_KEY``);
see the README for how to obtain one. Without a token every function falls
back to the committed offline sample via :func:`._io.cached`.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .. import config
from ._io import cached, log

# Map ENTSO-E generation PSR labels onto our canonical, aggregated columns.
_GEN_GROUPS = {
    "solar_mw": ["Solar"],
    "wind_mw": ["Wind Onshore", "Wind Offshore"],
    "hydro_mw": [
        "Hydro Run-of-river and poundage",
        "Hydro Water Reservoir",
        "Hydro Pumped Storage",
    ],
    "nuclear_mw": ["Nuclear"],
    "gas_mw": ["Fossil Gas"],
}


def _client():
    """Return an EntsoePandasClient, or None if token/lib unavailable."""
    if not config.ENTSOE_API_KEY:
        return None
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        log("entsoe-py not installed; cannot query ENTSO-E live")
        return None
    return EntsoePandasClient(api_key=config.ENTSOE_API_KEY)


def _window(start: Optional[str], end: Optional[str]):
    start = pd.Timestamp(start or config.INGEST_START, tz="UTC")
    end = pd.Timestamp(end or config.INGEST_END, tz="UTC")
    return start, end


def fetch_load(zone: str = config.BIDDING_ZONE,
               start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Hourly load [MW] indexed by UTC timestamp. Cached to data/raw/entsoe_load.parquet."""
    def download():
        client = _client()
        if client is None:
            return None
        s, e = _window(start, end)
        df = client.query_load(zone, start=s, end=e)
        s_load = df["Actual Load"] if isinstance(df, pd.DataFrame) else df
        out = s_load.to_frame("load_mw")
        return out.resample("h").mean()

    return cached("entsoe_load", download)


def fetch_generation(zone: str = config.BIDDING_ZONE,
                     start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Hourly generation by aggregated type [MW] (renewables share derivable downstream)."""
    def download():
        client = _client()
        if client is None:
            return None
        s, e = _window(start, end)
        raw = client.query_generation(zone, start=s, end=e)
        # entsoe-py returns MultiIndex columns (type, 'Actual Aggregated'/'Consumption').
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs("Actual Aggregated", axis=1, level=-1, drop_level=True)
        raw = raw.resample("h").mean()
        out = pd.DataFrame(index=raw.index)
        for col, labels in _GEN_GROUPS.items():
            present = [c for c in labels if c in raw.columns]
            out[col] = raw[present].sum(axis=1) if present else 0.0
        known = [c for labels in _GEN_GROUPS.values() for c in labels]
        out["other_mw"] = raw[[c for c in raw.columns if c not in known]].sum(axis=1)
        return out

    return cached("entsoe_generation", download)


def fetch_day_ahead_price(zone: str = config.BIDDING_ZONE,
                          start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Hourly day-ahead price [EUR/MWh]. Cached to data/raw/entsoe_price.parquet."""
    def download():
        client = _client()
        if client is None:
            return None
        s, e = _window(start, end)
        s_price = client.query_day_ahead_prices(zone, start=s, end=e)
        return s_price.to_frame("day_ahead_price_eur_mwh").resample("h").mean()

    return cached("entsoe_price", download)
