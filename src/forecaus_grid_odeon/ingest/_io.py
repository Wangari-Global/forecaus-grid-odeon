"""Shared ingestion plumbing: parquet caching + offline fixture fallback.

Every `fetch_*` function goes through :func:`cached`, which:
  1. returns the cached ``data/raw/<name>.parquet`` if it already exists
     (so re-running ``make ingest`` never re-downloads), else
  2. calls the real downloader, and if that yields nothing (no API token,
     no network, optional dep missing, or an upstream error) falls back to the
     committed 1-week sample under ``tests/fixtures/<name>.parquet``.

The fixture fallback is what makes ``make ingest`` and the test-suite work
fully offline and reproducibly. Real data is used automatically when the
relevant credentials/network are available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .. import config

#: Canonical schema for each cached dataset. The committed fixtures match this
#: exactly; real downloaders normalise their output to it. ``index`` is always a
#: UTC, hourly :class:`~pandas.DatetimeIndex` named ``time``.
SCHEMA: dict[str, list[str]] = {
    "entsoe_load": ["load_mw"],
    "entsoe_generation": [
        "solar_mw", "wind_mw", "hydro_mw", "nuclear_mw", "gas_mw", "other_mw",
    ],
    "entsoe_price": ["day_ahead_price_eur_mwh"],
    "rte_eco2mix": ["consumption_mw", "nuclear_mw", "co2_rate_g_per_kwh"],
    "weather": ["temp_c", "wind_ms", "irradiance_wm2"],
}

INDEX_NAME = "time"


def log(msg: str) -> None:
    print(f"[ingest] {msg}")


def _raw_path(name: str) -> Path:
    return config.RAW / f"{name}.parquet"


def _fixture_path(name: str) -> Path:
    return config.FIXTURES / f"{name}.parquet"


def load_fixture(name: str) -> pd.DataFrame:
    """Load the committed offline sample for ``name``."""
    path = _fixture_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"missing fixture {path}; run scripts/make_fixtures.py to (re)generate"
        )
    return normalise(pd.read_parquet(path), name)


def normalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Enforce the canonical schema: UTC hourly index named ``time``, the
    expected columns in order, float dtype, sorted, de-duplicated."""
    cols = SCHEMA[name]
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df.index.name = INDEX_NAME
    df = df[~df.index.duplicated(keep="first")].sort_index()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing expected columns {missing}")
    return df[cols].astype("float64")


def cached(name: str, downloader: Callable[[], Optional[pd.DataFrame]]) -> pd.DataFrame:
    """Return ``name`` from cache, else download (with fixture fallback) and cache it."""
    path = _raw_path(name)
    if path.exists():
        log(f"{name}: using cached {path}")
        return normalise(pd.read_parquet(path), name)

    df: Optional[pd.DataFrame]
    if config.OFFLINE:
        df = None
    else:
        try:
            df = downloader()
        except Exception as exc:  # network / API / parsing — fall back, don't crash
            log(f"{name}: download failed ({exc!r}); falling back to offline sample")
            df = None

    if df is None or df.empty:
        df = load_fixture(name)
        log(f"{name}: using offline sample ({len(df)} rows)")
    else:
        df = normalise(df, name)
        log(f"{name}: downloaded {len(df)} rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log(f"{name}: cached -> {path}")
    return df
