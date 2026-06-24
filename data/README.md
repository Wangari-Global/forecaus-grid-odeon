# Data

## Real LV-feeder ingest — `make ingest-ss-real` (REAL, measured)

`make ingest-ss-real` (→ `forecaus_grid_odeon.ingest_ss.real`) tries the
documented sources **in order** and uses the first that yields record-level
per-feeder demand without a manual login, writing one parquet per feeder to
`raw/ss/<feeder_key>.parquet` on the same `load_kw` contract as
`ingest_ss.ukpn` (UTC `time` index, single `load_kw` column). **It never
relabels the synthetic fixtures as real** — if every source fails it stops and
prints exactly what is missing.

**Source used (last run): UK Power Networks — "Smart Meter Consumption - LV
Feeder"** (`ukpowernetworks.opendatasoft.com`), via the opendatasoft Explore API
v2.1 records/exports with a **registered API key** (read from `UKPN_API_KEY`, or
a local token file; the key is never committed). Real aggregated smart-meter
active-energy import per secondary-substation × LV feeder, half-hourly;
Wh/half-hour → average kW (÷500, `ukpn.WH_PER_HH_TO_KW`).

- **Kind.** REAL — measured aggregated smart-meter data (not synthetic, not simulated).
- **Feeders.** 8 LV feeders across 6 secondary substations (EPN + LPN), 1440
  half-hours each, 0 % missing (kept: `<` `config.SS_MAX_MISSING` = 5 % gaps).
- **Span.** 2026-04-01 00:00 … 2026-04-30 23:30 UTC (~4.3 weeks). The download
  window is `config.SS_INGEST_START/END` (default `None` → auto-discover the
  **longest contiguous span the dataset publishes**).
- **History limitation (honest).** The target is ≥ 6 months, but the UKPN
  opendatasoft preview currently publishes **only ~1 month** of non-null load
  (April 2026; `records_count` capped at 30 000, all other months empty).
  Reaching ≥ 6 months needs UKPN's **registered bulk historical export**, which
  is *not* available through this keyless/API-key path — so the real series stay
  ~1 month and are **never padded or relabelled**. `make ingest-ss-real` prints a
  HISTORY NOTE whenever feeders fall short of `config.SS_TARGET_HISTORY_DAYS`.
- **Licence.** Creative Commons Attribution 4.0 (CC BY 4.0) — attribute "UK Power Networks".
- **Access date.** 2026-06-24.

**Sources tried, in order:**
1. **SP Energy Networks `lv_monitor`** (keyless opendatasoft) — *rejected*: only
   MONTHLY transformer capacity-utilisation aggregates (`year_month`,
   `*_capacity_utilisation`, `power_factor`) — no per-feeder `load_kw` series.
2. **UKPN "Smart Meter Consumption - LV Feeder"** — ✅ used (registered API key).
3. **Low Carbon London** — not reached (needs a bulk London-Datastore download +
   an external household→LV-feeder mapping; not a keyless record API).
4. **NREL SMART-DS** — not reached (SIMULATED; opt-in `FORECAUS_ALLOW_SIMULATED=1`,
   output would be labelled SIMULATED).

`raw/ss/` is git-ignored — **real data is never committed**; only the synthetic
offline fixtures under `tests/fixtures/ss/` are tracked (see below).

## Real SS-total aggregation (`raw/ss_agg/`)

`make aggregate-ss` (→ `forecaus_grid_odeon.ingest_ss.aggregate.substation_totals`,
also run automatically at the end of `make ingest-ss-real`) groups the per-feeder
`load_kw` series by `secondary_substation_id` and **SUMS** them to a single
**SS-total** `load_kw` series per substation — same contract (UTC `time` index,
one `load_kw` column) — one parquet per substation under `raw/ss_agg/`.

- **Rolled up (last run).** **6 secondary substations** from the **8 real feeders**
  (two substations have 2 feeders each — genuinely summed — the other four are
  single-feeder pass-throughs), half-hourly, span 2026-04-01 … 2026-04-30 UTC.
- **Alignment / gap-handling.** Feeders are outer-joined on the union of their
  timestamps; at each timestamp the SS-total is the SUM of the feeders reporting
  there, kept only where `>=` `min_feeders` are present (default 1) — timestamps
  with fewer reporting feeders are **dropped as gaps**, never under-counted.
- **Span limitation.** Inherits the ~1-month feeder span above (see *History
  limitation*); not yet the ≥ several months a full historical export would give.
- `raw/ss_agg/` is git-ignored (real data). The committed **synthetic**
  SS-aggregate fixtures live under `tests/fixtures/ss_agg/` (3 substations, summed
  from the synthetic feeder fixtures) for offline aggregation tests/CI.

## Secondary-substation / LV-feeder demand (`raw/ss/`)

**Source.** UK Power Networks Open Data — dataset
[`ukpn-smart-meter-consumption-lv-feeder`](https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn-smart-meter-consumption-lv-feeder/)
("Smart Meter Consumption - LV Feeder"), served by the opendatasoft Explore API
at `https://ukpowernetworks.opendatasoft.com/api/explore/v2.1` (dataset
metadata is public; **record-level access needs a free registered API key** —
see "Real LV-feeder ingest" above).

**What it is.** Half-hourly aggregated smart-meter consumption per
*secondary substation × LV feeder* (Active + Reactive Energy Import) plus the
count of contributing meters, across UKPN's three regions (EPN, LPN, SPN) —
i.e. many real LV/secondary-substation feeders, the granularity Challenge 4
targets.

**Ingestion.** `make ingest-ss` (→ `forecaus_grid_odeon.ingest_ss.ukpn`) fetches
the per-feeder active-import series, converts the half-hourly energy to average
power, and writes one tidy parquet per feeder to `raw/ss/<feeder_key>.parquet`
with a single UTC-indexed `load_kw` column. Re-runs reuse the cache and skip the
download.

- **Field mapping.** `secondary_substation_id` + `lv_feeder_id` → feeder key;
  `data_collection_log_timestamp` → UTC `time` index;
  `total_consumption_active_import` (Wh per half-hour) → `load_kw`.
- **Units.** Active Energy Import is in watt-hours per 30-min interval; average
  power `P[kW] = E[Wh] / 0.5 h / 1000 = E[Wh] / 500` (constant
  `ukpn.WH_PER_HH_TO_KW`).
- **Span.** Default download window `config.INGEST_START … config.INGEST_END`
  (2024-01-01 … 2025-06-01, UTC). The published preview is a rolling sample;
  the full historical series (all sites with >5 connected meters) is available
  via UKPN's registered-download instructions on the dataset page.
- **Licence.** Creative Commons Attribution 4.0 (CC BY 4.0) —
  <https://creativecommons.org/licenses/by/4.0/>. Attribute "UK Power Networks".
- **Access date.** 2026-06-22 (dataset schema/metadata confirmed live on this
  date; record-level access on the public portal currently requires
  registration/login).

## Offline sample (`../tests/fixtures/ss/`)

A small committed multi-feeder sample (5 LV feeders across 3 substations, 4 days
of half-hourly `load_kw`) lets `make ingest-ss` and the test-suite run with **no
network and no key** under `FORECAUS_OFFLINE=1`. It matches the real adapter's
schema exactly. Because live record access is gated behind portal login, the
committed sample is a deterministic, schema-accurate stand-in generated by
`scripts/make_ss_fixtures.py` (regenerate with
`python scripts/make_ss_fixtures.py`); swap in real exports once portal
credentials are available — the `load_kw` contract is identical.

## Other raw caches (`raw/`)

`entsoe_*`, `rte_eco2mix`, `weather` parquet files are the public-proxy caches
written by `make ingest` (ENTSO-E / RTE eCO2mix / Open-Meteo); see
`forecaus_grid_odeon.ingest`. All of `raw/` is git-ignored except the committed
fixtures under `tests/fixtures/`.
