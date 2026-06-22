# forecaus-grid-odeon — causal-augmented, transparent SS-level load forecasting for ODEON (Challenge 4)

Decision-intelligence layer for **ODEON Open Call Challenge 4**: day-ahead load
forecasting at **secondary-substation (SS)** level, transparent and **reproducible
by third parties**, plus the flexibility/congestion application and a federated
learning path that consumes ODEON's FL Engine.

> **Scope firewall.** This is a *separate* repo from `forecaus_grid_engine`
> (AID4SME C2.3, microgrid digital-twin sizing/control) and from the public
> `forecaus` demo (OpenHorizons OC3, corporate demand forecasting). The three
> share a forecasting/causal **core** but are funded under distinct calls and
> kept physically separate to avoid any overlap of funded activities.

## Lineage — what is reused vs new
**Reused core (from forecaus-grid):** `forecast/` (SARIMAX, DARTS TFT, conformal
intervals), `causal/` (DAG + DoWhy + refutation), `eval/` (rolling-origin
backtest + metrics), `features/`, `validation/` (deterministic numeric-claim
guard), and `ingest/` (public ENTSO-E/RTE/weather — kept as a **development
proxy** only).

**ODEON-specific (new):**
- `ingest_ss/` — secondary-substation adapters for the ODEON Energy Data Space
  (SCADA at MV/LV winding, LV smart-meter, LV topology). **Stub** until the
  ODEON API spec is released (help-desk Q1/Q4).
- `flex/` — the Challenge-4 energy application: forecast → flexibility volume &
  timing to relieve SS congestion. Deterministic v0 implemented.
- `fl/` — federated learning. A transparent **FedAvg** reference over
  interpretable parameters (for a pre-submission prototype), plus a **stub**
  client adapter to ODEON's FL Engine (we consume it; we do not rebuild it).

See `ODEON_Challenge4_FL_Architecture.docx` (in the ODEON project folder) for the design.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[notebooks,fl]'
export FORECAUS_OFFLINE=1            # run on the committed sample, no network
make forecast causal                 # core forecasting + causal evidence (proxy data)
make flex fl-demo                    # ODEON flexibility calc + FedAvg reference
make test
```

## Pre-submission priorities (per the gap analysis)
1. **FL prototype** — federate the transparent forecaster across SS-like nodes
   (extend `fl/`, e.g. with Flower). Highest-value new build; retires the FL gap.
2. **Synthetic SS + flexibility demo** — one SS-level series through the core
   pipeline + `flex/`. Shows the whole Challenge-4 chain.

In-project (grant-funded, not pre-submission): real SS ingestion on pilot data
(`ingest_ss/`) and the production ODEON FL-Engine adapter (`fl/OdeonFLClient`) —
these need pilot data / the API and drive the TRL 5/6 → 7/8 climb.
