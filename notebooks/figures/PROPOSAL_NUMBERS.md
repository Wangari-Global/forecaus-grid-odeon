# PROPOSAL_NUMBERS — real-data figures for the ODEON Challenge 4 application

Drop-in values for the `[[BUILD: …]]` placeholders in
`2607_ODEON_Challenge4_application_draft.md`. **All numbers are REAL** (measured
UK Power Networks LV-feeder smart-meter data) and machine-generated — do not hand-edit;
regenerate with the commands in the reproducibility block below.

- **Data source.** UK Power Networks — *"Smart Meter Consumption – LV Feeder"*
  (`ukpowernetworks.opendatasoft.com`, opendatasoft Explore API v2.1, registered API key).
- **Scope.** 8 LV feeders rolled up to **6 secondary substations** (EPN + LPN), half-hourly.
- **Span.** 2026-04-01 → 2026-04-30 UTC (~4.3 weeks), 0 % missing.
- **Licence.** CC BY 4.0 — attribute "UK Power Networks". **Accessed** 2026-06-24.

---

## [[BUILD: ss_forecast_mape_coverage]]

Day-ahead (48-step) load forecasting, rolling-origin backtest, leak-safe features
(calendar + weather + 24 h/168 h lags). **Headline = the substation-total level**
(the Challenge-4 target granularity); the per-feeder level is the harder lower bound.

**Substation-total — Challenge-4 target (HEADLINE), aggregate over 6 substations:**

| model | MAPE | MAE | RMSE | 90 % coverage |
|---|---:|---:|---:|---:|
| seasonal-naive (baseline) | 35.45 % | 3.13 kW | 4.15 kW | 0.902 |
| SARIMAX (baseline) | 39.79 % | 3.25 kW | 3.98 kW | 0.896 |
| **structured GAM (interpretable, ours)** | **29.81 %** | **2.55 kW** | **3.24 kW** | **0.906** |

**Per-feeder — harder lower bound, aggregate over 8 feeders:**

| model | MAPE | MAE | RMSE | 90 % coverage |
|---|---:|---:|---:|---:|
| seasonal-naive | 36.18 % | 2.65 kW | 3.55 kW | 0.898 |
| SARIMAX | 40.92 % | 2.71 kW | 3.31 kW | 0.902 |
| **structured GAM (ours)** | **30.51 %** | **2.15 kW** | **2.75 kW** | **0.904** |

> Headline accuracy is the **substation-total** interpretable model: **29.81 % MAPE
> at 0.906 interval coverage**, **beating seasonal-naive (35.45 %)** and the best of
> the three. The per-feeder level (30.51 %) is the harder lower bound. Both stay
> above the <5 % target — **reported honestly** — because these substations roll up
> only 1–2 LV feeders (≈ tens of homes), so they remain far spikier than the
> whole-network load that target was set for; approaching <5 % needs much larger
> aggregation. The value is the right granularity + transparency + calibrated
> intervals + federation (below), not a headline <5 %.

## [[BUILD: federated_vs_centralised_vs_local]]

Federated learning **across the 6 substations** (one SS-total series per node;
FedAvg, 12 rounds; per-node last-day holdout). **No raw data leaves a node — only
fixed-length parameter vectors (12 floats/round) are exchanged**:

| setting | aggregate test MAPE |
|---|---:|
| local-only (each substation alone) | 33.83 % |
| **federated-global (no data shared)** | **23.85 %** |
| centralised (data pooled — upper bound) | 23.51 % |

> Federation cuts aggregate MAPE from **33.83 % → 23.85 %** (a **≈10 pp** gain),
> reaching within **≈0.3 pp** of the centralised, data-pooled bound (23.51 %) —
> **without any raw consumption data leaving a substation.**

## [[BUILD: cold_start_gain]]

> A thin-history **substation** (only 24 training rows) improves from **111.02 %
> local to 49.95 % under federation — a +61.07 pp cold-start gain** from the shared
> global model, with no data shared.

## [[BUILD: forecast_to_flex]]

> Turning the **SS-total** day-ahead interval forecast into congestion relief: on
> the busiest substation, against an **illustrative** secondary-transformer
> firm-rating limit of **27.4 kW** (the 85th percentile of that substation's own
> historical half-hourly load — a transparent stand-in **until the pilot provides
> the real nameplate rating**), the schedule sizes **34.9 kWh** of risk-adjusted
> **down-flex** (peak **13.3 kW**) across **9** half-hours — vs **14.8 kWh** sized
> against the point forecast alone (the extra headroom the prediction interval buys).

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
