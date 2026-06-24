# forecaus-grid-odeon — causal-augmented, transparent SS-level load forecasting (ODEON Challenge 4)

[![CI](https://github.com/Wangari-Global/forecaus-grid-odeon/actions/workflows/ci.yml/badge.svg)](https://github.com/Wangari-Global/forecaus-grid-odeon/actions/workflows/ci.yml)
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

> ℹ️ **REAL data.** Headline numbers below are from **real UK Power Networks
> "Smart Meter Consumption – LV Feeder"** half-hourly demand (8 LV feeders, one
> recent month; CC BY 4.0) — ingest with `make ingest-ss-real`, provenance (span,
> licence, access date) in [`data/README.md`](data/README.md). The committed
> offline fixtures remain a clearly-labelled synthetic stand-in for tests/CI.

_Headline numbers below are regenerated from cached data by
`scripts/reproduce_headline.py` and pinned in
[`notebooks/figures/RESULTS.md`](notebooks/figures/RESULTS.md); the deterministic
guard (`validation/`) verifies every figure traces to a computed value._

Across **8 secondary-substation feeders** (real UKPN LV-feeder demand), the
**day-ahead** forecast benchmark gives an aggregate MAPE of **36.18%**
(seasonal-naive), **40.92%** (SARIMAX) and **30.51%** (interpretable structured
model). The **structured model beats seasonal-naive** (30.51% vs 36.18%) and is
the best of the three; it stays **above the <5% target (gap 25.51 pp)** because a
single LV feeder (≈tens of homes, half-hourly) is far noisier than the aggregated
load that target was set for — see the honesty note below.

Federating the structured model across **8 substation nodes** over **12 rounds**
lifts the aggregate test MAPE from **30.79%** (local-only) to **23.32%**
(federated-global), approaching the **22.99%** of a centralised model trained on
pooled data — **without any raw data leaving a node**. The thin-history feeder
(only **24** training rows) improves from **111.02%** local to **49.24%** under
federation, a cold-start gain of **61.78 pp**.

### ODEON benchmark table — `notebooks/figures/odeon_benchmark.csv`

Dataset: *real UK Power Networks "Smart Meter Consumption – LV Feeder"* (CC BY
4.0; per-feeder rows + span/access date in the CSV and `data/README.md`). MAPE in
%, coverage = share of actuals inside the 90% conformal interval (point-only for
the FL rows). Aggregate over 8 feeders:

| model | role | protocol | MAPE | coverage |
|---|---|---|---:|---:|
| seasonal_naive | baseline | day-ahead rolling-origin | 36.18 | 0.898 |
| sarimax | baseline | day-ahead rolling-origin | 40.92 | 0.902 |
| structured_gam | interpretable (non-federated) | day-ahead rolling-origin | 30.51 | 0.904 |
| structured_gam | local-only | last-day holdout, per-node | 30.79 | — |
| structured_gam | federated-global | last-day holdout, per-node | 23.32 | — |
| structured_gam | centralised (pooled) | last-day holdout, per-node | 22.99 | — |

The two `protocol` groups use different evaluation harnesses (rolling-origin
day-ahead vs per-node last-day holdout) and are reported separately rather than
mixed. Figures: `notebooks/figures/` (forecast benchmark, causal DAG + break,
federated convergence, forecast→flex schedule, edge-deployability). The
federated-convergence, forecast→flex and edge figures (+ `edge_fit.csv`) are
regenerated from the real data by `make figures-real`
(`scripts/regenerate_real_figures.py`); federation cuts aggregate test MAPE from
**30.79%** (local-only) to **23.32%** (federated-global, approaching the
**22.99%** centralised bound), and the interpretable model fits the documented
Raspberry-Pi-4 edge envelope (≈0.6 KiB artifact, sub-ms inference).

> **Honesty note.** On real per-feeder half-hourly demand the structured model
> beats seasonal-naive (30.51% vs 36.18%) but stays well above the <5% MAPE
> target — that target is an *aggregated*-load figure, and a single LV feeder is
> far spikier (small denominators inflate MAPE), so <5% is not attainable per
> feeder. The model's value is transparency + calibrated intervals + federation
> (which cuts aggregate MAPE from 30.79% to 23.32%) + the cold-start gain. The
> weather join and the 24 h / 168 h lags both help, and the leaky intra-horizon
> lag_1 / rolling features were removed so the day-ahead numbers are honest.
> Numbers are reported as computed, not cherry-picked.

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
