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

---

# Phase 2 — Real-data validation & honesty (post-scaffold)

**Why:** Phase-1 produced a complete, reproducible framework + a working FL demonstration, but every headline number came from the **synthetic offline fixtures** (`scripts/make_ss_fixtures.py`) — and the benchmark artifacts were **mislabeled as the real "UKPN Smart Meter Consumption – LV Feeder" dataset**. For ODEON that is two problems: (1) presenting synthetic data under a real dataset's name is a credibility/compliance risk we must remove; (2) synthetic data is TRL-4 (lab), whereas the call wants **TRL 5–6 = validation in a relevant environment**, which needs **real LV/substation data**. These slices fix labeling first, then get real data, then re-run.

**Standing context to add (pin it):**
> Synthetic fixtures must NEVER be presented as real data or under a real dataset's name. If real data cannot be obtained in a slice, STOP and report exactly which credential/registration is missing — do not silently fall back to fixtures and relabel them. Every result figure must state its data source truthfully. The ENTSO-E API key only unlocks national/transmission ENTSO-E proxy data (the `ingest/` core), NOT substation-level data (`ingest_ss/`).

Order: 12 → 13 (hygiene + honesty, do immediately), then 14 → 15 → 16 → 17 (real-data validation).

### Slice 12 — Repo hygiene: remove sync-conflict duplicates  *(deps: none)*
```
Remove the file-sync conflict artifacts that crept in: top-level `.gitignore 2`, `Makefile 2`, `pyproject 2.toml`, `src/forecaus_grid_odeon/cli 2.py`, the empty `src 2/` and `tests 2/` dirs, and every `*<space>N.parquet` / `weather N.parquet` duplicate under data/. Confirm none are git-tracked (they shouldn't be), confirm .gitignore still covers data/ and add a rule for ` [0-9].*` style duplicates if helpful. Acceptance: `git status` clean except intended changes; `python -m compileall -q src` OK; `pytest -q` still green.
```

### Slice 13 — Truth-in-labeling of current (synthetic) results  *(deps: 12)*
```
Make every result artifact state its true source. In scripts/make_notebooks.py (line ~149) STOP inserting the dataset label "UKPN Smart Meter Consumption - LV Feeder" for fixture runs; instead label the source as "SYNTHETIC schema-accurate LV stand-in (tests/fixtures/ss) — illustrative, not real measurements". Regenerate notebooks/figures/odeon_benchmark.csv and notebooks/figures/RESULTS.md so the dataset/source column and the RESULTS.md preamble say SYNTHETIC, with a one-line disclaimer at the top of RESULTS.md. Add the real dataset name back ONLY when a run is actually on real data (Slice 15+). Acceptance: no artifact claims real-data provenance; RESULTS.md carries the synthetic disclaimer; reproduce_headline + guard still pass.
```

### Slice 14 — Real-data ingestion (fallback ladder)  *(deps: 13)*  ← the gating slice
```
Get REAL secondary-substation / LV-feeder demand into data/raw/ss as per-feeder parquet (UTC index, single load_kw column — same contract as ingest_ss.ukpn). Try sources IN ORDER and use the first that yields record-level data without manual login; document which one succeeded in data/README.md (source, span, licence, access date):
  1) SP Energy Networks "LV Monitoring Aggregated Data" (dataset `lv_monitor`) via the opendatasoft Explore API v2.1 records endpoint (keyless if open).
  2) UK Power Networks "Smart Meter Consumption - LV Feeder" — only if a token/registered export is available (Ari provides; do NOT assume).
  3) Low Carbon London household smart-meter data aggregated to feeder level.
  4) NREL SMART-DS feeders — real-derived but simulated; if used, label as SIMULATED, not measured.
Add a `make ingest-ss-real` target and a sanity report (feeder count, span, %-missing); require >= a few feeders and >= several weeks of history. If ALL of 1–4 fail (e.g. every source needs credentials), STOP and print exactly what is needed — do NOT relabel fixtures as real. Acceptance: data/raw/ss holds real (or clearly-labelled simulated) multi-feeder series; sanity report printed; data/README.md updated with the true source.
```

### Slice 15 — Re-run benchmark + improve the model on real data  *(deps: 14)*
```
On the real data from Slice 14, re-run the day-ahead SS benchmark and IMPROVE the interpretable structured model so it at least beats seasonal-naive and trends toward the <5% MAPE target (the synthetic 4-day samples are why it currently underperforms): use longer history, the weather join, proper 24h/168h lags, and light tuning — keep it interpretable + edge-deployable. Regenerate odeon_benchmark.csv + RESULTS.md from real data with the REAL dataset name and per-feeder + aggregate MAPE/coverage. Report honestly: if <5% is not reached, state the achieved number and the gap, and whether it beats naive. Acceptance: benchmark on real data, structured model >= naive (or an honest explanation), RESULTS.md reproduced by `make benchmark`/reproduce_headline.
```

### Slice 16 — Re-run FL + flex + edge on real data  *(deps: 15)*
```
Re-run federated training across the real feeders (local vs federated-global vs centralised + cold-start gain), the forecast->flex congestion schedule, and the edge benchmark, all on real data. Regenerate the federated-convergence, forecast-to-flex, and edge figures + the RESULTS.md federated paragraph with real numbers. Keep the parameters-only / no-raw-data-leaves-node assertion test. Acceptance: all three figures + RESULTS.md regenerated from real data; FL still shows a directional gain (report honestly if not).
```

### Slice 17 — Proposal evidence hand-off  *(deps: 15, 16)*
```
Produce a single notebooks/figures/PROPOSAL_NUMBERS.md block containing the final, real-data figures formatted to drop straight into the application's [[BUILD: …]] placeholders: SS forecast MAPE + coverage; federated vs centralised vs local + cold-start; edge-fit; repo URL; and the reproducibility statement. Confirm the public repo is clean (no secrets/data, fixtures clearly synthetic, README Results section = real numbers). Acceptance: PROPOSAL_NUMBERS.md exists with real figures + provenance; repo ready for Ari to push and for the figures to be pasted into 2607_ODEON_Challenge4_application_draft.md.
```

## Gating note for Ari
Slice 14 may stop if every real LV source needs credentials. Fastest unlocks: try SPEN `lv_monitor` (keyless) first; if UKPN is preferred, register for portal access and hand CC a token/export. ENTSO-E's key does **not** help here — wrong granularity.

## Maps to the application
Slices 15–17 fill the `[[BUILD]]` placeholders in `2607_ODEON_Challenge4_application_draft.md` (SS accuracy, federated results, edge-fit, repo URL, reproducibility). Until then those placeholders stay empty; the FL-mechanism and edge results may be cited only with an explicit "synthetic/illustrative, pending real-data validation" caveat.

---

# Phase 3 — Forecast at the secondary-substation level (the real Challenge-4 target)

**Why:** Phase-2 validated everything on REAL UKPN data, but it forecasts **individual LV feeders** (≈tens of homes) and reports ~30% aggregate MAPE. Challenge 4's **<5%** target is for the **total load at the secondary-substation transformer** (the MV/LV winding), which sums many feeders/supply points and is far smoother. We are currently solving a harder, spikier problem than the call asks, and "30% vs <5%" reads badly in the proposal even with caveats. The UKPN records carry `secondary_substation_id`, so we can build the SS-total series and forecast THAT — the correct granularity — where <5% is realistically reachable. Also pull more history (Phase-2 used ~4.3 weeks; a thin-history node had only 24 rows). Keep the per-feeder results too, framed as the harder lower bound.

**Standing context to add (pin it):**
> Challenge 4's accuracy target (<5% MAPE) is at the SECONDARY-SUBSTATION TOTAL (the MV/LV transformer winding = sum of downstream feeders/supply points), NOT a single LV feeder. Forecast the SS-aggregate as the headline; keep per-feeder numbers as a reported harder case. Never present a per-feeder number as if it were the SS-level result. All real-data honesty rules from Phase 2 still apply.

Order: 18 → 19 → 20 → 21. (Re-uses the Phase-2 ingest/model/FL/flex machinery — these are extensions, not rewrites.)

### Slice 18 — Pull more history + build the SS-aggregate series  *(deps: Phase 2 committed)*
```
Two changes, both on REAL UKPN data via the existing ingest_ss.real / ukpn path (UKPN_API_KEY).
1) HISTORY: extend the download window to the longest contiguous span the dataset offers per feeder — target >= 6 months (ideally ~12) of half-hourly data — via config (e.g. INGEST_START/INGEST_END). Cache per feeder as today. Print per-feeder span + %-missing; keep feeders with >= a few months and < ~5% missing after cleaning.
2) AGGREGATION: add a function (e.g. ingest_ss.aggregate.substation_totals) that groups the per-feeder load_kw by `secondary_substation_id` and SUMS to a single SS-total load_kw series per substation (align timestamps; require >= K feeders present per timestep or document the gap-handling). Write one parquet per substation to data/raw/ss_agg/<substation_id>.parquet on the same load_kw contract. Add a committed small SYNTHETIC multi-substation SS-aggregate fixture under tests/fixtures/ss_agg/ for offline CI (clearly labelled synthetic, same pattern as the feeder fixtures).
Acceptance: data/raw/ss_agg holds >1 real SS-total series with documented span (>= several months); a unit test builds an SS-total from >=2 feeder fixtures and checks the sum + schema/freq; data/README.md updated (span, #substations, #feeders rolled up).
```

### Slice 19 — Re-benchmark at SS level (headline <5% attempt)  *(deps: 18)*
```
Run the day-ahead forecast benchmark on the SS-AGGREGATE series (seasonal_naive, SARIMAX, structured GAM) with the leak-safe calendar + weather + 24h/168h lags now that history is longer. Add cli `forecast-ss-agg` (or a --level=substation flag on forecast-ss). Report per-substation + aggregate MAPE/MAE/RMSE + interval coverage. Regenerate odeon_benchmark.csv with BOTH levels clearly labelled: role="substation-total (Challenge-4 target)" and the existing role="per-feeder (harder case)". Update RESULTS.md + PROPOSAL_NUMBERS.md so the HEADLINE accuracy is the SS-total number, with the per-feeder figure kept as the harder lower bound. Report honestly: if SS-total reaches/approaches <5%, state it; if not, state the achieved number and the gap and why.
Acceptance: SS-total benchmark present and labelled as the Challenge-4 target; structured model >= naive at SS level; RESULTS.md/PROPOSAL_NUMBERS.md headline = SS-total; guard PASS.
```

### Slice 20 — Re-run FL + flex at SS level  *(deps: 19)*
```
Federate the structured forecaster across SUBSTATIONS (each SS-total series = one node) with the existing Flower client + deterministic fedavg; report local vs federated-global vs centralised + cold-start at SS level, parameters-only (no raw data leaves a node). Re-run the forecast->flex congestion schedule on an SS-total forecast against a realistic SS transformer limit (document how the limit is set — e.g. a percentile of historical SS peak, clearly labelled as illustrative until the pilot provides the real rating). Regenerate the federated-convergence and forecast->flex figures + the RESULTS.md/PROPOSAL_NUMBERS.md federated + flex paragraphs at SS level.
Acceptance: FL and flex re-run on SS-total series; figures + numbers regenerated and labelled SS-level; no-raw-data-leaves-node test still passes.
```

### Slice 21 — Refresh evidence + commit  *(deps: 19, 20)*
```
Regenerate notebooks/figures and PROPOSAL_NUMBERS.md so every [[BUILD]] value reflects the SS-aggregate headline (keep per-feeder as the harder case). Re-run `python scripts/reproduce_headline.py` (guard PASS) and `pytest -q`. Ensure data/ stays gitignored, fixtures clearly synthetic, README Results section = SS-level real numbers. Commit the whole Phase-3 change set in logical chunks; print the suggested push command for Ari (do not push).
Acceptance: PROPOSAL_NUMBERS.md headline = SS-total real numbers; tests + guard green; clean commit(s); repo ready to push.
```

## Gating note for Ari
Needs `UKPN_API_KEY` set (already obtained) and that the longer history span is actually served by the dataset for the chosen feeders — if UKPN only exposes a short rolling window for these feeders, Slice 18 should report the real maximum span rather than pad it, and we decide whether to add more substations to compensate.

## Maps to the application
Slice 19 replaces the `[[BUILD: ss_forecast_mape_coverage]]` headline in `2607_ODEON_Challenge4_application_draft.md` with the **substation-total** number (the Challenge-4-comparable figure); Slice 20 refreshes the federated + flex placeholders at SS level; Slice 21 produces the final PROPOSAL_NUMBERS.md to paste. Until Slice 19 lands, cite only FL / edge / reproducibility, not the per-feeder accuracy.
