# forecaus-grid-odeon — causal-augmented, transparent SS-level load forecasting (ODEON Challenge 4)

[![CI](https://github.com/OWNER/forecaus-grid-odeon/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/forecaus-grid-odeon/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Day-ahead load forecasting at **secondary-substation (SS) / LV-feeder** level —
transparent, **reproducible by third parties**, with a flexibility/congestion
application and a federated-learning path. Forecasts are interpretable
(named coefficients), come with calibrated prediction intervals, and every
narrated number is checked back against a computed value.

> **This repository is a curated, public subset (artifact B1)** — the
> evaluation, benchmark, causal and FL-demo code that makes the results
> reproducible. It is **not** the production engine: the live ODEON FL Engine
> client, pilot-data ingestion, and orchestration/secure-aggregation are out of
> scope here (see [Scope firewall](#scope-firewall) and
> [What is *not* in this repo](#what-is-not-in-this-repo)).

## Results

_Headline numbers below are regenerated from cached data by
`scripts/reproduce_headline.py` and pinned in
[`notebooks/figures/RESULTS.md`](notebooks/figures/RESULTS.md); the deterministic
guard (`validation/`) verifies every figure traces to a computed value._

Across **5 secondary-substation feeders** (UK Power Networks "Smart Meter
Consumption – LV Feeder", offline sample), the **day-ahead** forecast benchmark
gives an aggregate MAPE of **6.51%** (seasonal-naive), **11.01%** (SARIMAX) and
**11.63%** (interpretable structured model); best aggregate MAPE **6.51%**.

Federating the structured model across **5 substation nodes** over **12 rounds**
lifts the aggregate test MAPE from **14.43%** (local-only) to **8.59%**
(federated-global), approaching the **6.41%** of a centralised model trained on
pooled data — **without any raw data leaving a node**. The thin-history feeder
(only **24** training rows) improves from **25.66%** local to **10.34%** under
federation, a cold-start gain of **15.32 pp**.

### ODEON benchmark table — `notebooks/figures/odeon_benchmark.csv`

Dataset: *UKPN Smart Meter Consumption – LV Feeder*. MAPE in %, coverage = share
of actuals inside the 90% conformal interval (point-only for the FL rows).

| model | role | protocol | MAPE | coverage |
|---|---|---|---:|---:|
| seasonal_naive | baseline | day-ahead rolling-origin | 6.51 | 0.921 |
| sarimax | baseline | day-ahead rolling-origin | 11.01 | 0.933 |
| structured_gam | interpretable (non-federated) | day-ahead rolling-origin | 11.63 | 0.854 |
| structured_gam | local-only | last-day holdout, per-node | 14.43 | — |
| structured_gam | federated-global | last-day holdout, per-node | 8.59 | — |
| structured_gam | centralised (pooled) | last-day holdout, per-node | 6.41 | — |

The two `protocol` groups use different evaluation harnesses (rolling-origin
day-ahead vs per-node last-day holdout) and are reported separately rather than
mixed. Figures: `notebooks/figures/` (forecast benchmark, causal DAG + break,
federated convergence, forecast→flex schedule).

> **Honesty note.** On this tiny offline sample the seasonal-naive baseline is
> strong (near-periodic days), so the interpretable model does not "win" on
> MAPE here; its value is transparency + federation + the cold-start gain, which
> grow with real history. Numbers are reported as computed, not cherry-picked.

## One-command reproduction

```bash
make setup                                   # pip install -e '.[notebooks,fl]'  (Python 3.10–3.13)
FORECAUS_OFFLINE=1 python scripts/reproduce_headline.py
```

This recomputes the headline SS-forecast + FL numbers from the committed
fixtures (no network, no keys), prints the table, runs the deterministic guard
over the narrative, and refreshes `notebooks/figures/RESULTS.md`. To regenerate
every figure + the benchmark CSV:

```bash
FORECAUS_OFFLINE=1 jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

Other entry points (all run offline):

```bash
make forecast-ss      # day-ahead SS benchmark: structured GAM vs seasonal-naive vs SARIMAX
make causal-ss        # SS DAG -> DoWhy effect + refuters; causal vs correlational on a regime change
make fl-train         # federate the transparent forecaster across feeders (Flower + FedAvg)
make flex-run         # forecast -> risk-adjusted congestion-relief schedule
make ingest-ss        # per-feeder LV demand parquet (UKPN; falls back to fixtures offline)
make test             # full test suite
python scripts/edge_benchmark.py             # Pi 4-class edge-deployability check
```

## Data sources & licences

All development data is public; pilot data stays on ODEON infrastructure.

- **UK Power Networks — "Smart Meter Consumption – LV Feeder"** (primary SS
  dataset), keyless opendatasoft API, **CC BY 4.0**. See
  [`data/README.md`](data/README.md) for fields, units, span and access date.
  The committed `tests/fixtures/ss/*.parquet` are a tiny multi-feeder sample so
  the pipeline runs offline; the live record endpoint currently requires a
  portal login, so the sample is a deterministic, schema-accurate stand-in.
- **ENTSO-E / RTE eCO2mix / Open-Meteo (ERA5)** — national/regional proxy data
  used by `ingest/` for development only (committed 1-week fixtures under
  `tests/fixtures/`). ENTSO-E needs a free API token (never committed; read from
  the `ENTSOE_API_KEY` environment variable).
- **ODEON Energy Data Space** — `ingest_ss/odeon_api.py` parses a committed
  **mock** payload (`tests/fixtures/odeon/`) into the same tidy frames; the live
  calls raise `NotImplementedError` until the API spec lands.

No API keys or downloaded datasets are committed; `data/` (raw/processed caches)
is git-ignored — only the small fixtures under `tests/fixtures/` are tracked.

## Scope firewall

This is a **separate** repository from `forecaus_grid_engine` (AID4SME C2.3,
microgrid digital-twin sizing/control) and from the public `forecaus` demo
(OpenHorizons OC3, corporate demand forecasting). The three share a
forecasting/causal **core** but are funded under distinct calls and kept
physically separate to avoid any overlap of funded activities. This public
artifact (B1) exposes only the curated eval/benchmark/causal/FL-demo subset.

### Lineage — reused vs new
- **Reused core:** `forecast/` (SARIMAX, DARTS TFT, conformal intervals),
  `causal/` (DAG + DoWhy + refutation), `eval/` (rolling-origin backtest +
  metrics), `features/`, `validation/` (deterministic numeric-claim guard),
  `ingest/` (public ENTSO-E/RTE/weather — development proxy only).
- **ODEON-specific (new):** `ingest_ss/` (SS adapters: UKPN LV-feeder +
  ODEON-mock), `forecast/structured.py` (interpretable edge/federatable model),
  `eval/ss_benchmark.py` + `eval/ss_causal.py`, `flex/` (forecast→flexibility),
  `fl/` (Flower client + deterministic FedAvg reference).

## What is *not* in this repo

Out of scope for this public subset (and by the non-overlap rule):

- the **production ODEON FL Engine** client — consumed via the
  `fl/OdeonFLClient` stub, *not* reimplemented (orchestration / secure
  aggregation are ODEON baseline);
- **real pilot-data ingestion** (`ingest_ss/odeon_api.py` live path is stubbed);
- the **AID4SME microgrid engine** and the **OC3 forecaus** demo.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE). Copyright 2026 Ari Joury / Wangari.
Third-party data retains its own licence (UKPN data is CC BY 4.0; attribute
"UK Power Networks").
