"""REAL secondary-substation / LV-feeder demand ingest.

Tries the documented sources **in order** and uses the first that yields
record-level per-feeder demand without a manual login, writing one parquet per
feeder to ``data/raw/ss/`` under the same contract as :mod:`.ukpn`
(UTC ``time`` index, single ``load_kw`` column). It NEVER falls back to the
synthetic fixtures — if every source fails it STOPs and prints exactly what is
missing (see :func:`ingest_ss_real`).

Source order (per the slice spec):
  1. **SP Energy Networks "LV Monitoring Aggregated Data"** (`lv_monitor`,
     opendatasoft v2.1, keyless) — probed, but it only exposes MONTHLY
     transformer capacity-utilisation aggregates (no per-feeder ``load_kw``
     time series), so it cannot satisfy the contract.
  2. **UK Power Networks "Smart Meter Consumption - LV Feeder"** — half-hourly
     aggregated active-energy import per secondary-substation x LV feeder,
     via the opendatasoft v2.1 API with a registered **API key** (read from
     ``UKPN_API_KEY`` or the local token file; never committed).
  3. **Low Carbon London** household smart-meter data aggregated to feeder
     level — bulk download + an external household->feeder mapping (not a
     keyless record API).
  4. **NREL SMART-DS** feeders — real-derived but **SIMULATED**; opt-in only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config
from . import ukpn

ACCESS_DATE = "2026-06-24"
MIN_FEEDERS = 3          # require "a few" feeders
MIN_DAYS = 14            # require "several weeks" of history


def log(msg: str) -> None:
    print(f"[ingest-ss-real] {msg}")


# --------------------------------------------------------------- token (UKPN) --
def _ukpn_token() -> Optional[str]:
    """API key from ``UKPN_API_KEY`` env, else the local token file. Never logged."""
    env = os.environ.get("UKPN_API_KEY")
    if env and env.strip():
        return env.strip()
    # repo_root = .../forecaus_grid_odeon ; token kept one level above the repo set.
    candidate = config.RAW.parents[1].parents[1] / "_TechReference" / "ukpn_token.txt"
    if candidate.exists():
        return candidate.read_text().strip()
    return None


# ------------------------------------------------------------------ source 1 --
def _try_spen_lv_monitor(*_args, **_kwargs):
    """SPEN lv_monitor (keyless). Returns None — it has no per-feeder load_kw series."""
    try:
        import requests
        url = "https://spenergynetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets/lv_monitor"
        r = requests.get(url, timeout=30)
        fields = {f["name"] for f in r.json().get("fields", [])} if r.ok else set()
    except Exception:
        fields = set()
    has_demand = any(k in fields for k in ("active_power", "load_kw", "demand", "consumption"))
    has_subdaily_time = any(k in fields for k in ("timestamp", "datetime", "date_time", "half_hour"))
    reason = ("SPEN `lv_monitor` exposes only MONTHLY transformer capacity-utilisation "
              "aggregates (year_month, *_capacity_utilisation %, power_factor) — no "
              "per-feeder load_kw time series, so it cannot meet the load_kw contract.")
    if has_demand and has_subdaily_time:  # would adapt if SPEN ever adds a demand series
        reason += " (Unexpected: demand-like fields now present — adapter needs writing.)"
    log("source 1 (SPEN lv_monitor): " + reason)
    return None


# ------------------------------------------------------------------ source 2 --
def _try_ukpn(n_feeders: int, min_nonnull: int, *, start: Optional[str] = None,
              end: Optional[str] = None):
    """UKPN Smart Meter Consumption - LV Feeder via opendatasoft v2.1 + API key."""
    token = _ukpn_token()
    if not token:
        log("source 2 (UKPN): no API key (set UKPN_API_KEY or place ../../_TechReference/"
            "ukpn_token.txt) — skipping.")
        return None
    import requests

    base = f"{ukpn.BASE_URL}/catalog/datasets/{ukpn.DATASET_ID}"
    headers = {"Authorization": f"Apikey {token}"}
    T, IMP, SUB, FDR = (ukpn.F_TIME, ukpn.F_ACTIVE_IMPORT, ukpn.F_SUBSTATION, ukpn.F_FEEDER)
    notnull = f"{IMP} is not null"

    def _get(path, **params):
        resp = requests.get(f"{base}/{path}", headers=headers, params=params, timeout=120)
        resp.raise_for_status()
        return resp.json()

    # Discover the available non-null span if not given.
    if start is None or end is None:
        lo = _get("records", select=T, where=notnull, order_by=T, limit=1)["results"]
        hi = _get("records", select=T, where=notnull, order_by=f"{T} desc", limit=1)["results"]
        if not lo or not hi:
            log("source 2 (UKPN): API key valid but no non-null records found — skipping.")
            return None
        start = start or lo[0][T][:10]
        end = end or (pd.Timestamp(hi[0][T]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    window = f'{notnull} and {T}>="{start}" and {T}<"{end}"'
    log(f"source 2 (UKPN): window {start} .. {end}; discovering best-covered feeders")

    # Discover feeders with the most non-null half-hourly records in the window.
    disc = _get("records", select=f"{SUB},{FDR},count(*) as n", where=window,
                group_by=f"{SUB},{FDR}", order_by="n desc", limit=n_feeders * 3)["results"]
    feeders = [(r[SUB], r[FDR]) for r in disc if r["n"] >= min_nonnull][:n_feeders]
    if len(feeders) < MIN_FEEDERS:
        log(f"source 2 (UKPN): only {len(feeders)} feeders have >= {min_nonnull} records — skipping.")
        return None

    out: dict[str, pd.DataFrame] = {}
    for sub, fid in feeders:
        w = f'{SUB}="{sub}" and {FDR}={int(fid)} and {window}'
        rows = _get("exports/json", select=f"{T},{IMP}", where=w, order_by=T)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df[T] = pd.to_datetime(df[T], utc=True)
        df = df.set_index(T)
        df["load_kw"] = pd.to_numeric(df[IMP], errors="coerce") / ukpn.WH_PER_HH_TO_KW
        out[ukpn.feeder_key(sub, fid)] = ukpn.normalise(df.dropna(subset=["load_kw"]))
        log(f"  {ukpn.feeder_key(sub, fid)}: {len(out[ukpn.feeder_key(sub, fid)])} half-hours")

    if len(out) < MIN_FEEDERS:
        return None
    spans = [(v.index.min(), v.index.max()) for v in out.values()]
    provenance = {
        "source": ('UK Power Networks — "Smart Meter Consumption - LV Feeder" '
                   "(ukpowernetworks.opendatasoft.com), opendatasoft Explore API v2.1, "
                   "registered API key. Real aggregated smart-meter active-energy import "
                   "per secondary-substation x LV feeder, half-hourly; Wh/half-hour "
                   f"converted to average kW (÷{ukpn.WH_PER_HH_TO_KW:g})."),
        "kind": "REAL (measured)",
        "span": f"{min(s for s, _ in spans):%Y-%m-%d %Hh} .. {max(e for _, e in spans):%Y-%m-%d %Hh} UTC",
        "licence": ukpn.LICENCE,
        "access_date": ACCESS_DATE,
    }
    return out, provenance


# ------------------------------------------------------------ sources 3 & 4 --
def _try_lcl(*_a, **_k):
    log("source 3 (Low Carbon London): needs a bulk CSV download from the London "
        "Datastore (data.london.gov.uk) PLUS an external household->LV-feeder mapping "
        "(LCL exposes ACORN groups, not feeder topology) — not a keyless record API. "
        "Skipping.")
    return None


def _try_nrel_smartds(*_a, **_k):
    if os.environ.get("FORECAUS_ALLOW_SIMULATED", "").lower() not in ("1", "true", "yes"):
        log("source 4 (NREL SMART-DS): SIMULATED (real-derived) feeders via OEDI S3 "
            "(s3://oedi-data-lake/SMART-DS/, no login, bulk download). Opt in with "
            "FORECAUS_ALLOW_SIMULATED=1 — output would be labelled SIMULATED. Skipping.")
        return None
    raise NotImplementedError(
        "NREL SMART-DS bulk S3 ingest is not implemented; provide a local extract or "
        "implement the OEDI S3 pull. Output MUST be labelled SIMULATED, not measured.")


# ------------------------------------------------------------------ orchestrate --
def _sanity_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for key, df in frames.items():
        full = pd.date_range(df.index.min(), df.index.max(), freq=ukpn.FREQ, tz="UTC")
        pct_missing = float(df.reindex(full).isna().any(axis=1).mean() * 100.0)
        days = (df.index.max() - df.index.min()).total_seconds() / 86400.0
        rows[key] = {"rows": len(df), "start": df.index.min(), "end": df.index.max(),
                     "days": round(days, 1), "pct_missing": round(pct_missing, 2),
                     "load_kw_min": round(float(df["load_kw"].min()), 1),
                     "load_kw_max": round(float(df["load_kw"].max()), 1)}
    rep = pd.DataFrame(rows).T
    rep.index.name = "feeder"
    return rep


def ingest_ss_real(*, n_feeders: int = 8, min_weeks: int = 2,
                   start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """Try the documented sources in order; write the first that works to
    ``data/raw/ss/`` and print a sanity report. STOPs (SystemExit) if all fail."""
    min_nonnull = int(min_weeks * 7 * 48)
    sources = [
        ("SPEN lv_monitor", lambda: _try_spen_lv_monitor()),
        ("UKPN Smart Meter Consumption - LV Feeder",
         lambda: _try_ukpn(n_feeders, min_nonnull, start=start, end=end)),
        ("Low Carbon London", lambda: _try_lcl()),
        ("NREL SMART-DS (SIMULATED)", lambda: _try_nrel_smartds()),
    ]

    chosen = None
    for name, fn in sources:
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 - record and continue to next source
            log(f"{name}: failed ({type(exc).__name__}: {str(exc)[:160]})")
            res = None
        if res:
            chosen = (name, *res)
            break

    if chosen is None:
        print("\n" + "=" * 72)
        print("STOP — no source yielded real record-level per-feeder demand.")
        print("Fixtures are NOT a substitute and have NOT been relabelled. To proceed:")
        print("  1) SPEN lv_monitor: provides only monthly capacity-utilisation, not a")
        print("     load_kw series — needs a different SPEN dataset with half-hourly demand.")
        print("  2) UKPN: set UKPN_API_KEY (or place ../../_TechReference/ukpn_token.txt) —")
        print("     a registered opendatasoft API key for ukpowernetworks.opendatasoft.com.")
        print("  3) Low Carbon London: download the bulk CSV from the London Datastore +")
        print("     supply a household->LV-feeder mapping.")
        print("  4) NREL SMART-DS: set FORECAUS_ALLOW_SIMULATED=1 (output labelled SIMULATED).")
        print("=" * 72)
        raise SystemExit(2)

    name, frames, provenance = chosen
    raw_dir = config.RAW / "ss"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, df in frames.items():
        df.to_parquet(raw_dir / f"{key}.parquet")

    report = _sanity_report(frames)
    print(f"\n=== real SS ingest — source: {name} [{provenance['kind']}] ===")
    print(report.to_string())
    span_days = (report["end"].max() - report["start"].min()).total_seconds() / 86400.0
    print(f"\nfeeders: {len(frames)} | span: {provenance['span']} | "
          f"licence: {provenance['licence']} | accessed: {provenance['access_date']}")

    problems = []
    if len(frames) < MIN_FEEDERS:
        problems.append(f"only {len(frames)} feeders (< {MIN_FEEDERS})")
    if span_days < MIN_DAYS:
        problems.append(f"span {span_days:.1f} d (< {MIN_DAYS} d)")
    if problems:
        raise AssertionError("real SS ingest did not meet acceptance: " + "; ".join(problems))

    print(f"OK: {len(frames)} real feeders, span {span_days:.0f} days "
          f"(>= {MIN_DAYS}) written to {raw_dir}")
    return {"source": name, "provenance": provenance, "report": report, "frames": frames}
