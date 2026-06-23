"""UK Power Networks LV-feeder smart-meter ingestion (real public dataset).

Source: UK Power Networks Open Data, dataset
``ukpn-smart-meter-consumption-lv-feeder`` ("Smart Meter Consumption - LV
Feeder"), served by the keyless opendatasoft HTTP API at
https://ukpowernetworks.opendatasoft.com . It publishes **half-hourly
aggregated active/reactive energy import** per secondary substation x LV feeder,
with the count of contributing smart meters - i.e. many LV/secondary-substation
feeders, exactly the granularity Challenge 4 targets.

We pull the per-feeder active-import series, convert the half-hourly energy to
average power, and normalise to a tidy UTC-indexed frame with a single
``load_kw`` column - one parquet per feeder under ``data/raw/ss/``. Re-runs hit
the cache and skip the download. With ``FORECAUS_OFFLINE=1`` (or no network) the
committed multi-feeder sample under ``tests/fixtures/ss/`` is used instead, so
ingestion is reproducible with no network and no key.

Units: ``total_consumption_active_import`` is the half-hourly Active Energy
Import in watt-hours (Wh). Average power over the 30-minute interval is
``P[kW] = E[Wh] / 0.5 h / 1000 = E[Wh] / 500``. The factor is a named constant
so the conversion is auditable and easy to retune if UKPN restate the units.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .. import config

# --------------------------------------------------------------------- source --
BASE_URL = "https://ukpowernetworks.opendatasoft.com/api/explore/v2.1"
DATASET_ID = "ukpn-smart-meter-consumption-lv-feeder"
LICENCE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"

# Real field names (confirmed against the dataset schema).
F_SUBSTATION = "secondary_substation_id"
F_FEEDER = "lv_feeder_id"
F_TIME = "data_collection_log_timestamp"
F_ACTIVE_IMPORT = "total_consumption_active_import"   # Wh per half-hour
F_DEVICE_COUNT = "aggregated_device_count_active"

#: Half-hourly Active Energy Import [Wh] -> average power [kW] over the interval.
WH_PER_HH_TO_KW = 500.0

#: Canonical SS schema: one tidy column, UTC half-hourly index named ``time``.
SS_SCHEMA = ["load_kw"]
INDEX_NAME = "time"
FREQ = "30min"

#: Default feeders to ingest offline / when live discovery is unavailable. These
#: match the committed sample under ``tests/fixtures/ss/`` (substation, feeder).
DEFAULT_FEEDERS: list[tuple[str, int]] = [
    ("EPN0012345", 1),
    ("EPN0012345", 2),
    ("LPN0004567", 1),
    ("LPN0004567", 3),
    ("SPN0008901", 1),
]


# ------------------------------------------------------------------- plumbing --
def log(msg: str) -> None:
    print(f"[ingest-ss] {msg}")


def feeder_key(substation_id: str, lv_feeder_id) -> str:
    """Filename-safe, stable key for one substation x feeder series."""
    sub = re.sub(r"[^0-9A-Za-z]+", "-", str(substation_id)).strip("-")
    return f"{sub}__feeder-{lv_feeder_id}"


def _raw_dir() -> Path:
    return config.RAW / "ss"


def _fixture_dir() -> Path:
    return config.FIXTURES / "ss"


def _raw_path(key: str) -> Path:
    return _raw_dir() / f"{key}.parquet"


def _fixture_path(key: str) -> Path:
    return _fixture_dir() / f"{key}.parquet"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the canonical SS schema: UTC index named ``time``, single
    ``load_kw`` float column, sorted and de-duplicated."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df.index.name = INDEX_NAME
    df = df[~df.index.duplicated(keep="first")].sort_index()
    missing = [c for c in SS_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"SS feeder frame missing columns {missing}")
    return df[SS_SCHEMA].astype("float64")


def load_fixture(key: str) -> pd.DataFrame:
    """Load one committed offline feeder sample by key."""
    path = _fixture_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"missing SS fixture {path}; run scripts/make_ss_fixtures.py to (re)generate"
        )
    return normalise(pd.read_parquet(path))


def available_fixture_keys() -> list[str]:
    """Keys of every committed feeder sample under ``tests/fixtures/ss/``."""
    return sorted(p.stem for p in _fixture_dir().glob("*.parquet"))


# ------------------------------------------------------------------- download --
def _download_feeder(
    substation_id: str, lv_feeder_id, start: str, end: str
) -> Optional[pd.DataFrame]:
    """Fetch one feeder's half-hourly active-import series from the UKPN API.

    Returns a ``load_kw`` frame, or ``None`` if requests is unavailable or the
    feeder has no rows in the window (caller falls back to the offline sample).
    """
    try:
        import requests
    except ImportError:
        return None

    where = (
        f'{F_SUBSTATION}="{substation_id}" and {F_FEEDER}={int(lv_feeder_id)} '
        f'and {F_TIME}>="{start}" and {F_TIME}<"{end}"'
    )
    params = {
        "select": f"{F_TIME},{F_ACTIVE_IMPORT}",
        "where": where,
        "order_by": F_TIME,
        "timezone": "UTC",
        "limit": -1,
    }
    url = f"{BASE_URL}/catalog/datasets/{DATASET_ID}/exports/json"
    log(f"GET {url}  [{substation_id} / feeder {lv_feeder_id}]")
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if F_TIME not in df or F_ACTIVE_IMPORT not in df:
        return None
    df[F_TIME] = pd.to_datetime(df[F_TIME], utc=True)
    energy_wh = pd.to_numeric(df[F_ACTIVE_IMPORT], errors="coerce")
    out = pd.DataFrame({"load_kw": energy_wh / WH_PER_HH_TO_KW})
    out.index = df[F_TIME]
    return out.dropna()


def discover_feeders(limit: int = 5) -> list[tuple[str, int]]:
    """List real (substation, feeder) pairs from the API; fall back to defaults.

    Uses the opendatasoft ``group_by`` records query to enumerate distinct
    feeders. Best-effort: any failure (no network, gated records, missing
    requests) returns :data:`DEFAULT_FEEDERS` so offline ingestion still works.
    """
    try:
        import requests

        url = f"{BASE_URL}/catalog/datasets/{DATASET_ID}/records"
        params = {
            "select": f"{F_SUBSTATION},{F_FEEDER}",
            "group_by": f"{F_SUBSTATION},{F_FEEDER}",
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        pairs = [
            (r[F_SUBSTATION], r[F_FEEDER])
            for r in results
            if r.get(F_SUBSTATION) is not None and r.get(F_FEEDER) is not None
        ]
        if pairs:
            return pairs
    except Exception as exc:  # noqa: BLE001 - any failure -> documented fallback
        log(f"feeder discovery unavailable ({exc!r}); using default feeder set")
    return list(DEFAULT_FEEDERS)


# ------------------------------------------------------------------ public API --
def fetch_feeder(
    substation_id: str,
    lv_feeder_id,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Return one feeder's ``load_kw`` series, using cache / download / fixture.

    Caches to ``data/raw/ss/<key>.parquet`` and skips the download if present.
    """
    key = feeder_key(substation_id, lv_feeder_id)
    path = _raw_path(key)
    if path.exists():
        log(f"{key}: using cached {path}")
        return normalise(pd.read_parquet(path))

    df: Optional[pd.DataFrame]
    if config.OFFLINE:
        df = None
    else:
        try:
            df = _download_feeder(
                substation_id,
                lv_feeder_id,
                start or config.INGEST_START,
                end or config.INGEST_END,
            )
        except Exception as exc:  # noqa: BLE001 - network/API/parse -> fall back
            log(f"{key}: download failed ({exc!r}); falling back to offline sample")
            df = None

    if df is None or df.empty:
        df = load_fixture(key)
        log(f"{key}: using offline sample ({len(df)} rows)")
    else:
        df = normalise(df)
        log(f"{key}: downloaded {len(df)} rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log(f"{key}: cached -> {path}")
    return df


def ingest_ss(
    feeders: Optional[Iterable[tuple[str, int]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Ingest every requested feeder to a per-feeder ``load_kw`` parquet.

    Parameters
    ----------
    feeders : iterable of (substation_id, lv_feeder_id), optional
        Feeders to ingest. Defaults to the committed sample feeders offline, or
        live-discovered feeders online.

    Returns ``{feeder_key: DataFrame}``.
    """
    if feeders is None:
        if config.OFFLINE:
            # Offline: ingest exactly the feeders we have committed samples for.
            keys = available_fixture_keys() or [feeder_key(s, f) for s, f in DEFAULT_FEEDERS]
            feeders = [_key_to_feeder(k) for k in keys]
        else:
            feeders = discover_feeders()

    out: dict[str, pd.DataFrame] = {}
    for substation_id, lv_feeder_id in feeders:
        key = feeder_key(substation_id, lv_feeder_id)
        out[key] = fetch_feeder(substation_id, lv_feeder_id, start, end)

    for key, df in out.items():
        span = f"{df.index.min()} .. {df.index.max()}" if len(df) else "empty"
        log(f"{key}: {len(df)} rows  [{span}]")
    log(f"ingested {len(out)} feeders -> {_raw_dir()}")
    return out


def _key_to_feeder(key: str) -> tuple[str, int]:
    """Recover (substation, feeder) from a feeder key (for offline iteration)."""
    sub, _, feeder = key.rpartition("__feeder-")
    return sub, int(feeder) if feeder.isdigit() else feeder
