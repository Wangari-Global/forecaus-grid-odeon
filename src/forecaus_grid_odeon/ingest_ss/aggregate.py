"""Aggregate per-feeder ``load_kw`` up to one SS-total per secondary substation.

A secondary substation generally supplies several LV feeders. This groups the
per-feeder series by their ``secondary_substation_id`` and **SUMS** ``load_kw``
to a single SS-total series per substation, on the same canonical contract as
:mod:`.ukpn` (UTC half-hourly index named ``time``, one ``load_kw`` column) —
written one parquet per substation under ``data/raw/ss_agg/``.

Alignment / gap-handling
------------------------
A substation's feeders are outer-joined on the union of their timestamps. At each
timestamp the SS-total is the SUM of the feeders reporting there, kept **only
where at least ``min_feeders`` are present**; timestamps with fewer reporting
feeders are dropped as gaps rather than silently under-counting the substation.
With ``min_feeders=1`` (default) a single-feeder substation passes through
unchanged and every aligned timestamp is kept.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from .. import config
from . import ukpn

AGG_SUBDIR = "ss_agg"


def log(msg: str) -> None:
    print(f"[aggregate-ss] {msg}")


def _agg_raw_dir() -> Path:
    return config.RAW / AGG_SUBDIR


def _agg_fixture_dir() -> Path:
    return config.FIXTURES / AGG_SUBDIR


def substation_of(feeder_key: str) -> str:
    """The secondary-substation id a feeder key belongs to (drops the feeder)."""
    sub, _ = ukpn._key_to_feeder(feeder_key)
    return sub


def substation_totals(
    feeders: Optional[Mapping[str, pd.DataFrame]] = None, *, min_feeders: int = 1
) -> dict[str, pd.DataFrame]:
    """Group per-feeder ``load_kw`` by substation and SUM to an SS-total series.

    Parameters
    ----------
    feeders : mapping ``{feeder_key: load_kw frame}``, optional
        Defaults to every cached real feeder under ``data/raw/ss/`` (offline:
        the committed synthetic feeder fixtures).
    min_feeders : int
        Require at least this many feeders present at a timestamp for the SS-total
        to be defined there (otherwise that timestamp is dropped, see module doc).

    Returns ``{substation_id: load_kw frame}`` on the canonical SS contract.
    """
    if feeders is None:
        feeders = _load_cached_feeders()
    if not feeders:
        raise RuntimeError("no per-feeder series to aggregate (run `make ingest-ss-real`)")

    groups: dict[str, dict[str, pd.Series]] = {}
    for key, df in feeders.items():
        series = ukpn.normalise(df)["load_kw"]
        groups.setdefault(substation_of(key), {})[key] = series

    totals: dict[str, pd.DataFrame] = {}
    for sub, series_map in groups.items():
        mat = pd.concat(series_map, axis=1).sort_index()      # union index; NaN where a feeder is absent
        present = mat.notna().sum(axis=1)
        total = mat.sum(axis=1, min_count=1).where(present >= min_feeders).dropna()
        if total.empty:
            log(f"{sub}: no timestamp has >= {min_feeders} feeders present — skipped")
            continue
        totals[sub] = ukpn.normalise(pd.DataFrame({"load_kw": total}))
    return totals


def _load_cached_feeders() -> dict[str, pd.DataFrame]:
    """Load every cached real feeder (offline: the committed feeder fixtures)."""
    keys = ukpn.available_fixture_keys() if config.OFFLINE else ukpn.cached_feeder_keys()
    out: dict[str, pd.DataFrame] = {}
    for key in keys:
        sub, fid = ukpn._key_to_feeder(key)
        out[key] = ukpn.fetch_feeder(sub, fid)
    return out


def write_substation_totals(*, min_feeders: int = 1) -> dict[str, pd.DataFrame]:
    """Aggregate the cached real feeders and write one parquet per substation to
    ``data/raw/ss_agg/<substation_id>.parquet``."""
    totals = substation_totals(min_feeders=min_feeders)
    out_dir = _agg_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_feeders = len(_load_cached_feeders())
    for sub, df in totals.items():
        df.to_parquet(out_dir / f"{sub}.parquet")
        span = f"{df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d}"
        log(f"{sub}: {len(df)} half-hours, {span}, {df['load_kw'].min():.1f}-{df['load_kw'].max():.1f} kW")
    log(f"wrote {len(totals)} SS-total series (from {n_feeders} feeders) -> {out_dir}")
    return totals


# ----------------------------------------------------------- offline fixtures --
def available_agg_fixture_keys() -> list[str]:
    """Substation ids of every committed SS-aggregate fixture under
    ``tests/fixtures/ss_agg/`` (clearly-labelled synthetic)."""
    return sorted(p.stem for p in _agg_fixture_dir().glob("*.parquet"))


def load_agg_fixture(substation_id: str) -> pd.DataFrame:
    """Load one committed SS-aggregate (SS-total) fixture by substation id."""
    path = _agg_fixture_dir() / f"{substation_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing SS-aggregate fixture {path}; run scripts/make_ss_fixtures.py to (re)generate"
        )
    return ukpn.normalise(pd.read_parquet(path))
