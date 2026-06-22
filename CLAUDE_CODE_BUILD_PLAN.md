# forecaus-grid-odeon — Claude Code Build Plan (prompt slices)

How to use: hand the **slices in order** to Claude Code, one at a time, from the repo root. Each slice is self-contained — paste the fenced prompt, let CC implement + run, review the diff, commit, then move on. One slice = one commit.

**Standing context to give CC once (pin it):**
> This repo (`forecaus-grid-odeon`) builds **TRL 5–6 evidence for ODEON Open Call Challenge 4**: day-ahead **load forecasting at secondary-substation (SS) level**, **transparent and reproducible by third parties**, with a **federated learning** path and a **flexibility/congestion** application. It shares its forecasting/causal/eval/validation core with `forecaus-grid` (AID4SME) but is a SEPARATE, distinctly-funded repo — do not import from or push to that repo, and never touch the public `forecaus` demo. Key ODEON rules: (a) FL must be used natively end-to-end, BUT we **consume ODEON's Federated Learning Engine** — do NOT build a competing federation/orchestration engine (non-overlap rule); (b) integrate by **API**, do not duplicate ODEON infra; (c) edge target ≈ **Raspberry Pi 4**, so prefer a lightweight, **interpretable** model class (GAM / gradient-boosted trees) over heavy nets; (d) forecast target **<5% MAPE** day-ahead at SS level; (e) every cited number must be **reproducible** (deterministic guard). Real ODEON pilot data + the production FL-Engine adapter are **in-project** (not pre-submission) — pre-submission we validate on **public LV/substation data** as a relevant-environment proxy. Stack: pandas, statsmodels, scikit-learn, DARTS, DoWhy/EconML, networkx, Flower (`flwr`). Cache data to `data/` (gitignored). Stubs are marked; keep code small, tested, reproducible, and report results as-found (negative results are fine).

**Dependency order:** 0 → 1 → 2 → 3 → 4 → 5 → 6, then artifacts 7 → 8 → 9 → 10 → 11. Slices 1→3→5 are the critical path (SS data → forecast → federation).

---

### Slice 0 — Environment & CI smoke  *(deps: none)*
```
Make `pip install -e '.[fl]'` work and `make test` pass in a fresh venv. Resolve any resolver conflicts (esp. darts + numpy; flwr). The repo already has tests for flex/ and fl/ plus the inherited core tests — get them green. Add .github/workflows/ci.yml that installs the package and runs `pytest -q` on push. Do not change model logic. Acceptance: clean install; `pytest -q` green; CI file present.
```

### Slice 1 — Secondary-substation data ingestion  *(deps: 0)*  ← critical path; lifts TRL 4→5
```
Implement src/forecaus_grid_odeon/ingest_ss/ for a REAL public LV/secondary-substation dataset with MANY substations/feeders. Primary source: UK Power Networks "Smart Meter Consumption - LV Feeder" (ukpowernetworks.opendatasoft.com, keyless HTTP/CSV API). Fetch per-feeder demand time series for a set of feeders, normalise to a tidy UTC-indexed frame per feeder (column: load_kw), cache to data/raw/ss/*.parquet, skip re-download if present. Add a small committed multi-feeder sample (a few days, a handful of feeders) under tests/fixtures/ss/ so tests run offline (FORECAUS_OFFLINE=1). Wire a `make ingest-ss` target and cli `ingest-ss` to actually run this (replace the current stub print). Add data/README.md noting source, span, licence, access date. Acceptance: `make ingest-ss` writes per-feeder parquet; a unit test loads the fixture and checks schema/freq/feeder count; offline mode needs no network.
```

### Slice 2 — SS features + multi-node panel  *(deps: 1)*
```
Extend features/ and pipeline.py to build an SS-level modelling frame per substation: calendar (cyclical + holidays for the dataset's country), lagged load (e.g. 1, 24, 168 steps adapted to series length), weather join if available (open-meteo, keyless, aligned to the load index), and optional topology features when present. Provide load_ss_frame(feeder_id) and an iterator over feeders for the federated setup. Enforce no train/test leakage around a configurable split. Acceptance: frames are aligned + NA-free; a unit test asserts no future leakage in lag columns and that >1 feeder is available for federation.
```

### Slice 3 — Transparent SS forecaster + benchmark toward <5% MAPE  *(deps: 2)*  ← critical path
```
Add an interpretable, edge-deployable forecaster in forecast/ (a GAM-style structured regressor OR gradient-boosted trees) exposing named parameters/coefficients so it can federate and be audited. Wrap with the existing conformal intervals. Benchmark it day-ahead at SS level against seasonal_naive + SARIMAX via eval/rolling_origin, reporting MAE/RMSE/MAPE + interval coverage per feeder and aggregated. Add cli `forecast-ss`. Acceptance: `make forecast-ss` prints a per-feeder + aggregate metrics table on the SS data; MAPE is finite and trending toward the <5% target; conformal coverage ≈ nominal; the model exposes a parameter dict for Slice 5.
```

### Slice 4 — Causal layer + structural-break robustness at SS level  *(deps: 2)*  ← differentiator
```
Reuse causal/ (DAG → DoWhy estimate → refutation). Define an SS-level DAG (load ← temperature, calendar; downstream PV/EV where present). Run the effect estimate + the placebo/random-common-cause/subset refuters. Reuse eval/causal_break to compare the causal-augmented forecaster vs a purely-correlational ML model on overall vs a regime-change window (use a real documented break if the series has one, else a clearly-labelled injected shift). Add cli `causal-ss`. Acceptance: prints effect + CI + a pass/fail refuter table, and a causal-vs-correlational figure/table on the break window; result reported honestly (incl. negative).
```

### Slice 5 — Federated learning across substations  *(deps: 3)*  ← critical path; retires the FL gap
```
Extend fl/ to federate the Slice-3 transparent forecaster across substations as nodes, using Flower (flwr) on the client side with the existing deterministic fedavg() as the aggregation reference. Each feeder trains locally; only parameters move (assert no raw data leaves a node). Produce: global-vs-local accuracy per feeder, the cold-start benefit for a thin-history feeder, and round-by-round convergence. Add cli `fl-train`. Keep the production path consuming ODEON's FL Engine via the fl/OdeonFLClient stub — do NOT build orchestration/secure-agg yourself (ODEON baseline). Acceptance: `make fl-train` runs N-node federation offline on the fixtures, prints global vs centralised vs local metrics + a convergence trace; a test asserts fedavg determinism and that no node exchanges raw samples.
```

### Slice 6 — Flexibility / congestion application, end-to-end  *(deps: 3)*
```
Extend flex/congestion.py beyond v0: accept the forecast point + interval and an SS limit (thermal and/or contractual), and return a RISK-ADJUSTED flexibility need (volume + timing) — e.g. size flex against the upper interval, not just the point. Add a forecast→flex pipeline and cli `flex-run` that takes a Slice-3 forecast and outputs the congestion-relief schedule. Acceptance: `flex-run` produces a per-timestep flex_up/flex_down schedule with timing on the SS series; unit tests cover the limit-breach and reverse-power cases and the interval-based sizing.
```

### Slice 7 — Edge-fit evidence (Raspberry Pi 4 budget)  *(deps: 3)*
```
Add scripts/edge_benchmark.py that measures the trained interpretable model's artifact size, training time, and single-inference latency under a constrained CPU/RAM budget (document a Pi 4-class envelope; cap threads). Emit a small table to notebooks/figures/edge_fit.csv. Acceptance: script runs and writes the table; numbers show the model fits a Pi 4-class envelope (or, if not, report it honestly and note mitigation).
```

### Slice 8 — ODEON API integration contract  *(deps: 1)*
```
Give ingest_ss/odeon_api.py a concrete interface CONTRACT against a MOCKED ODEON Energy Data Space response (a small JSON fixture mimicking SS SCADA / LV smart-meter / topology payloads). Implement the adapters to parse the mock into the same tidy frames as Slice 1, behind a flag, so the pipeline can run against "ODEON-shaped" data without real access. Keep the live calls raising NotImplementedError until the API spec lands (help-desk Q1/Q4). Acceptance: a test feeds the mock JSON through fetch_* and gets frames matching the Slice-1 schema; live path still clearly stubbed.
```

### Slice 9 — Reproducibility / transparency artifact  *(deps: 3, 5)*  ← Challenge-4-specific
```
Add scripts/reproduce_headline.py that recomputes the headline SS forecast/FL numbers from cached data and prints them, plus run the validation/ deterministic guard over a short narrative of those numbers to show each cited figure traces to a computed value (flag any that don't). Write notebooks/figures/RESULTS.md with the headline numbers in prose. Acceptance: `python scripts/reproduce_headline.py` reproduces the table; the guard passes on the real narrative and flags an injected hallucinated number in a test.
```

### Slice 10 — Reproducible notebooks + benchmark table  *(deps: 3, 4, 5, 6)*
```
Create notebooks that run top-to-bottom offline: 01_ss_forecast_benchmark, 02_ss_causal, 03_federated_across_substations, 04_forecast_to_flex. Each writes its figures to notebooks/figures/. Produce the ODEON benchmark table (federated vs centralised vs local, + baselines, MAPE/coverage on the named public LV dataset). Acceptance: `jupyter nbconvert --execute` runs all notebooks clean; figures + the benchmark CSV regenerated; numbers match RESULTS.md.
```

### Slice 11 — Public-repo hygiene + push  *(deps: 10; 9 if ready)*
```
Make this a clean, citable PUBLIC repo (artifact B1) as a CURATED SUBSET, not the production engine. Add a permissive LICENSE (Apache-2.0) for the eval/benchmark/causal/FL-demo code; confirm .gitignore excludes data/, .venv/, secrets, *.parquet except committed fixtures; scrub any keys/downloaded data. README: add a Results section pulling RESULTS.md headline numbers, a one-command repro path, data-source/licence notes, and the scope-firewall note (separate from AID4SME and the OC3 forecaus demo). Commit in logical chunks and PRINT the suggested `git remote add` + push commands for Ari to run (do not push yourself). Acceptance: clean history, no secrets/data, README Results populated; ready for Ari to push and paste the URL into the application.
```

---

## Maps to the pre-submission build & TRL plan
- Slice 1 → **A1** (SS data) · Slice 2 → **A1/A2** · Slice 3 → **A2** + artifact **B2** · Slice 4 → **A3** · Slice 5 → **A4** + artifact **B3** · Slice 6 → **A5** · Slice 7 → **A6** · Slice 8 → **A7** · Slice 9 → artifact **B4** · Slice 10 → artifacts **B1/B2/B3** · Slice 11 → artifact **B1** (+ **B5** TRL one-pager refs; **B6** Zenodo note is a separate manual step with a DISTINCT DOI from AID4SME).

## Feasibility cut (≈2.5 weeks to 9 July)
**Must-have:** 0 → 1 → 2 → 3 → 5 → 6, then 9 → 10 → 11.
**If time allows:** 4 (cheap, reuses the harness), 7, 8.
**Critical path:** 1 (SS data) → 3 (transparent forecast) → 5 (federate it). flex v0 and fedavg already exist, so 5/6 are extensions, not greenfield.

## Guardrails for every slice
- **Non-overlap:** consume ODEON's FL Engine; never build federation/orchestration/secure-agg or any ODEON baseline infra.
- **Scope firewall:** separate from `forecaus-grid` (AID4SME) and the `forecaus` OC3 demo — no cross-imports, no pushing to those repos, separate Zenodo DOI.
- **TRL honesty:** claim the general-purpose solution at TRL 5–6 (validated on real public LV data); SS-on-pilot-data + production FL adapter are the in-project climb to 7–8.
- **Transparency:** interpretable model + deterministic guard; every cited number reproducible by a script.

## Hand-off to the application
When the must-have slices are done, copy into the ODEON Challenge-4 proposal: the headline SS forecast accuracy + interval coverage and the federated-vs-centralised numbers (from RESULTS.md); the public repo URL; and the TRL claim anchored to etio's production track record. Keep claims honest: public-data relevant-environment validation, TRL 5–6, FL prototype (production adapter in-project).
