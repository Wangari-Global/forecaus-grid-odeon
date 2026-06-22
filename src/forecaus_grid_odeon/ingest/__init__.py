"""Public data ingestion: ENTSO-E, RTE eCO2mix, weather (ERA5) -> parquet cache.

Each source is cached under ``data/raw/*.parquet`` and re-uses the cache on the
next run. With no network / API token, sources transparently fall back to the
committed 1-week sample under ``tests/fixtures`` so the pipeline is reproducible
offline.
"""
from __future__ import annotations

import pandas as pd

from . import entsoe, rte, weather
from ._io import SCHEMA

__all__ = ["entsoe", "rte", "weather", "run", "SCHEMA"]


def run() -> dict[str, pd.DataFrame]:
    """Fetch (or load from cache) every source. Returns name -> DataFrame.

    Weather is aligned to the ENTSO-E load index.
    """
    load = entsoe.fetch_load()
    out = {
        "entsoe_load": load,
        "entsoe_generation": entsoe.fetch_generation(),
        "entsoe_price": entsoe.fetch_day_ahead_price(),
        "rte_eco2mix": rte.fetch_eco2mix(),
        "weather": weather.fetch_weather(index=load.index),
    }
    for name, df in out.items():
        span = f"{df.index.min()} .. {df.index.max()}"
        print(f"[ingest] {name}: {len(df)} rows x {len(df.columns)} cols  [{span}]")
    return out
