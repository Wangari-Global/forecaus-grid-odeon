"""ODEON Energy Data Space adapters — secondary-substation level.

Challenge 4 inputs (per Technical Guidelines / Annex 1):
  * SCADA at the MV/LV transformer winding  (power, voltage, current; 1-5 min)
  * LV smart-meter import/export at supply points            (15 min)
  * LV grid topology                                         (all pilots)

The live ODEON API spec is not released pre-submission (help-desk Q1/Q4), so the
**live path raises NotImplementedError**. To let the pipeline run against
"ODEON-shaped" data without real access, each adapter has a concrete interface
CONTRACT pinned to a committed MOCK payload (``tests/fixtures/odeon/``); set
``FORECAUS_ODEON_MOCK=1`` (or pass ``source="mock"``) to route there.

Output contract — every time-series adapter returns the **same tidy frame as
Slice 1** (:data:`forecaus_grid_odeon.ingest_ss.ukpn.SS_SCHEMA`): a single
``load_kw`` float column on a UTC :class:`~pandas.DatetimeIndex` named ``time``,
so ODEON data drops straight into ``features.build`` / ``pipeline``. Topology is
structural and returns a tidy per-feeder frame.

Mock payload contract (see the fixture for a worked example)::

    {
      "scada":          {"secondary_substation_id", "interval_minutes",
                         "readings": [{"timestamp", "active_power_kw",
                                       "voltage_v", "current_a"}, ...]},
      "lv_smart_meter": {"secondary_substation_id", "interval_minutes",
                         "readings": [{"timestamp", "active_import_wh",
                                       "active_export_wh"}, ...]},
      "lv_topology":    {"secondary_substation_id", "transformer_rating_kva",
                         "feeders": [{"lv_feeder_id", "n_supply_points",
                                      "phases"}, ...]},
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config
from .ukpn import INDEX_NAME, SS_SCHEMA, normalise

_PENDING = (
    "ODEON Energy Data Space API spec not released pre-submission "
    "(help-desk Q1/Q4). Use forecaus_grid_odeon.ingest (public proxy) for dev, "
    "or set FORECAUS_ODEON_MOCK=1 / source='mock' to run against the mock payload."
)

#: Committed mock payload that pins the interface contract.
MOCK_PATH = Path(config.FIXTURES) / "odeon" / "odeon_ss_mock.json"


# ----------------------------------------------------------------- plumbing ----
def _resolve_source(source: Optional[str]) -> str:
    """'mock' if requested (arg or FORECAUS_ODEON_MOCK), else 'live'."""
    if source is not None:
        return source
    return "mock" if config.ODEON_MOCK else "live"


def load_mock_payload(path: Optional[Path] = None) -> dict:
    """Load the committed mock ODEON payload (the contract example)."""
    path = Path(path) if path is not None else MOCK_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"missing ODEON mock {path}; run scripts/make_odeon_mock.py to (re)generate"
        )
    return json.loads(path.read_text())


def _empty_load_frame() -> pd.DataFrame:
    return normalise(pd.DataFrame({"load_kw": []},
                                  index=pd.DatetimeIndex([], name=INDEX_NAME, tz="UTC")))


def _readings_index(readings: list[dict]) -> pd.DatetimeIndex:
    return pd.to_datetime([r["timestamp"] for r in readings], utc=True)


def _utc(ts) -> pd.Timestamp:
    """Coerce a date/Timestamp (naive or aware) to a UTC Timestamp."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        df = df.loc[df.index >= _utc(start)]
    if end is not None:
        df = df.loc[df.index < _utc(end)]
    return df


def _ss_matches(block: dict, ss_id: Optional[str]) -> bool:
    return ss_id is None or block.get("secondary_substation_id") == ss_id


# ------------------------------------------------------------------ parsers ----
def parse_scada(payload: dict, ss_id: Optional[str] = None,
                start=None, end=None) -> pd.DataFrame:
    """SCADA winding telemetry -> tidy ``load_kw`` frame (active power = SS load)."""
    block = payload["scada"]
    if not _ss_matches(block, ss_id):
        return _empty_load_frame()
    rec = block["readings"]
    df = pd.DataFrame({"load_kw": [float(r["active_power_kw"]) for r in rec]},
                      index=_readings_index(rec))
    return _window(normalise(df), start, end)


def parse_smart_meter(payload: dict, ss_id: Optional[str] = None,
                      start=None, end=None) -> pd.DataFrame:
    """LV smart-meter import/export (Wh per interval) -> tidy ``load_kw`` frame.

    Net load = (import - export) energy converted to average power over the
    reporting interval: ``kW = Wh / (interval_min/60) / 1000``.
    """
    block = payload["lv_smart_meter"]
    if not _ss_matches(block, ss_id):
        return _empty_load_frame()
    interval_h = float(block.get("interval_minutes", 15)) / 60.0
    rec = block["readings"]
    net_wh = [float(r.get("active_import_wh", 0.0)) - float(r.get("active_export_wh", 0.0))
              for r in rec]
    load_kw = [wh / interval_h / 1000.0 for wh in net_wh]
    df = pd.DataFrame({"load_kw": load_kw}, index=_readings_index(rec))
    return _window(normalise(df), start, end)


def parse_topology(payload: dict, ss_id: Optional[str] = None) -> pd.DataFrame:
    """LV grid topology -> tidy per-feeder frame (one row per LV feeder)."""
    block = payload["lv_topology"]
    sub = block.get("secondary_substation_id")
    rating = block.get("transformer_rating_kva")
    rows = [
        {
            "secondary_substation_id": sub,
            "lv_feeder_id": f["lv_feeder_id"],
            "n_supply_points": f.get("n_supply_points"),
            "phases": f.get("phases"),
            "transformer_rating_kva": rating,
        }
        for f in block["feeders"]
    ]
    return pd.DataFrame(rows).set_index("lv_feeder_id").sort_index()


# ------------------------------------------------------------------ adapters ---
def fetch_ss_scada(ss_id: str, start=None, end=None, *,
                   source: Optional[str] = None, payload: Optional[dict] = None) -> pd.DataFrame:
    """SCADA at the MV/LV winding for one SS (1-5 min) -> Slice-1 ``load_kw`` frame."""
    if _resolve_source(source) != "mock":
        raise NotImplementedError(_PENDING)
    return parse_scada(payload if payload is not None else load_mock_payload(), ss_id, start, end)


def fetch_lv_smart_meter(ss_id: str, start=None, end=None, *,
                         source: Optional[str] = None, payload: Optional[dict] = None) -> pd.DataFrame:
    """LV supply-point smart-meter import/export for one SS (15 min) -> ``load_kw`` frame."""
    if _resolve_source(source) != "mock":
        raise NotImplementedError(_PENDING)
    return parse_smart_meter(payload if payload is not None else load_mock_payload(), ss_id, start, end)


def fetch_lv_topology(ss_id: str, *, source: Optional[str] = None,
                      payload: Optional[dict] = None) -> pd.DataFrame:
    """LV grid topology for one SS (feeder/supply-point table)."""
    if _resolve_source(source) != "mock":
        raise NotImplementedError(_PENDING)
    return parse_topology(payload if payload is not None else load_mock_payload(), ss_id)


__all__ = [
    "fetch_ss_scada", "fetch_lv_smart_meter", "fetch_lv_topology",
    "parse_scada", "parse_smart_meter", "parse_topology",
    "load_mock_payload", "SS_SCHEMA", "MOCK_PATH",
]
