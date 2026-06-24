# PROPOSAL_NUMBERS — real-data figures for the ODEON Challenge 4 application

Drop-in values for the `[[BUILD: …]]` placeholders in
`2607_ODEON_Challenge4_application_draft.md`. **All numbers are REAL** (measured
UK Power Networks LV-feeder smart-meter data) and machine-generated — do not hand-edit;
regenerate with the commands in the reproducibility block below.

- **Data source.** UK Power Networks — *"Smart Meter Consumption – LV Feeder"*
  (`ukpowernetworks.opendatasoft.com`, opendatasoft Explore API v2.1, registered API key).
- **Scope.** 8 LV/secondary-substation feeders (EPN + LPN), half-hourly.
- **Span.** 2026-04-01 → 2026-04-30 UTC (~4.3 weeks), 0 % missing.
- **Licence.** CC BY 4.0 — attribute "UK Power Networks". **Accessed** 2026-06-24.

---

## [[BUILD: ss_forecast_mape_coverage]]

Day-ahead (48-step) load forecasting at LV-feeder level, rolling-origin backtest,
aggregate over 8 real feeders (leak-safe: calendar + weather + 24 h/168 h lags):

| model | MAPE | 90 % interval coverage |
|---|---:|---:|
| seasonal-naive (baseline) | 36.18 % | 0.898 |
| SARIMAX (baseline) | 40.92 % | 0.902 |
| **structured GAM (interpretable, ours)** | **30.51 %** | **0.904** |

> The interpretable structured model is the best of the three and **beats the
> seasonal-naive baseline (30.51 % vs 36.18 %)**, with calibrated prediction
> intervals (90.4 % empirical coverage at the nominal 90 %). It stays above the
> aggregate-load <5 % target (gap 25.5 pp) **because a single LV feeder
> (≈tens of homes, half-hourly) is intrinsically far spikier than aggregated
> demand** — reported honestly; the value is transparency + calibration +
> federation (below), not a headline <5 %.

## [[BUILD: federated_vs_centralised_vs_local]]

Federated learning across the 8 substations (FedAvg, 12 rounds; per-node last-day
holdout). **No raw data leaves a node — only fixed-length parameter vectors
(12 floats/round) are exchanged**:

| setting | aggregate test MAPE |
|---|---:|
| local-only (each node alone) | 30.79 % |
| **federated-global (no data shared)** | **23.32 %** |
| centralised (data pooled — upper bound) | 22.99 % |

> Federation cuts aggregate MAPE from **30.79 % → 23.32 %** (a **7.5 pp** gain),
> reaching within **0.3 pp** of the centralised, data-pooled bound (22.99 %) —
> **without any raw consumption data leaving a substation.**

## [[BUILD: cold_start_gain]]

> A thin-history feeder (only 24 training rows) improves from **111.02 % local to
> 49.24 % under federation — a +61.78 pp cold-start gain** from the shared global
> model, with no data shared.

## [[BUILD: edge_fit]]

Interpretable forecaster on a real-data model (18 features, 1 104 training rows),
measured single-core and scaled to a **Raspberry Pi 4-class** envelope:

| metric | measured (Pi-4 est.) | budget | within budget |
|---|---:|---:|:--:|
| deployable artifact size | 0.59 KiB | 64 KiB | ✅ |
| peak workload memory | 0.53 MiB | 50 MiB | ✅ |
| local training time | 6.5 ms | 1 000 ms | ✅ |
| single-step inference | 0.012 ms | 5 ms | ✅ |

> The model is genuinely **edge-deployable**: a ~0.6 KiB named-coefficient
> artifact, sub-millisecond inference and a few-MiB footprint — fits a Raspberry
> Pi 4-class envelope on every budgeted metric (PASS), at a conservative 12× Pi-4
> single-core slowdown.

## [[BUILD: repo_url]]

> Public code (Apache-2.0): **https://github.com/Wangari-Global/forecaus-grid-odeon**

## [[BUILD: reproducibility_statement]]

> All figures are regenerated from cached **real** UKPN LV-feeder data and every
> cited number is checked against a computed value by a deterministic guard
> (`scripts/reproduce_headline.py` exits non-zero on any unsupported figure).
> Reproduce end-to-end:
> `make ingest-ss-real` (real data → `data/raw/ss/`; needs a free UKPN
> opendatasoft API key) → `python scripts/reproduce_headline.py` (SS + federated
> tables + `RESULTS.md`, guard PASS) → `make figures-real` (federated-convergence,
> forecast→flex and edge figures). The committed offline fixtures are a clearly
> labelled synthetic stand-in for network-free CI; real-data provenance (source,
> span, licence, access date) is recorded in `data/README.md`.

---

_Source artifacts: `notebooks/figures/odeon_benchmark.csv` (per-feeder + aggregate),
`notebooks/figures/edge_fit.csv`, `notebooks/figures/RESULTS.md`,
`notebooks/figures/{03_federated_convergence,04_forecast_to_flex,edge_fit}.png`._
