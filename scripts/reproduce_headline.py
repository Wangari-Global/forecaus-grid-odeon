#!/usr/bin/env python
"""Recompute the headline SS forecast + FL numbers and prove every cited figure
traces to a computed value; regenerate odeon_benchmark.csv + RESULTS.md.

Data source is chosen truthfully at runtime:
  * if real per-feeder caches exist under ``data/raw/ss/`` (from
    ``make ingest-ss-real``) and we are not forced offline -> **REAL** UKPN
    LV-feeder data, labelled with the real dataset name;
  * otherwise -> the committed **SYNTHETIC** stand-in fixtures, clearly labelled.

The deterministic guard re-parses every number in the prose and checks it
against a computed value, so the artifacts cannot drift or fabricate.

Run:  python scripts/reproduce_headline.py
Exit code 0 iff the table reproduces and the guard passes.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_MD = REPO / "notebooks" / "figures" / "RESULTS.md"
BENCH_CSV = REPO / "notebooks" / "figures" / "odeon_benchmark.csv"

# Force offline (reproducible on fixtures) ONLY when there is no real data — so a
# real ingest is used automatically, but the test-suite / CI stay on fixtures.
_HAS_REAL = any((REPO / "data" / "raw" / "ss").glob("*.parquet"))
if not _HAS_REAL:
    os.environ.setdefault("FORECAUS_OFFLINE", "1")

import sys  # noqa: E402

MAPE_TARGET = 5.0   # aggregate-load day-ahead target (stated, for honest gap reporting)

# Truthful provenance. SYNTHETIC stand-in must NEVER carry the real dataset name;
# the real name is used only when a run is actually on real data.
SYN_LABEL = ("SYNTHETIC schema-accurate LV stand-in (tests/fixtures/ss) "
             "— illustrative, not real measurements")
SYN_DISCLAIMER = ("⚠️ **SYNTHETIC data.** Every figure below comes from a "
                  "schema-accurate LV stand-in (`tests/fixtures/ss`) — illustrative only, "
                  "**not real measurements**. The real UKPN LV-feeder dataset is wired in "
                  "but gated; real-data provenance is shown only on a real-data run.")
REAL_LABEL = ("UK Power Networks 'Smart Meter Consumption - LV Feeder' "
              "(real measured half-hourly LV-feeder demand; CC BY 4.0; "
              "see data/README.md for span and access date)")
REAL_DISCLAIMER = ("**REAL data.** Figures below come from real **UK Power Networks "
                   "'Smart Meter Consumption – LV Feeder'** half-hourly demand; per-feeder "
                   "breakdown, span, licence and access date are in "
                   "`notebooks/figures/odeon_benchmark.csv` and `data/README.md`.")


def compute_headline(rounds: int = 12) -> dict:
    """Recompute the headline numbers; return computed values + narrative + tables."""
    import warnings

    warnings.filterwarnings("ignore")
    from forecaus_grid_odeon.eval import run_ss_benchmark
    from forecaus_grid_odeon.fl import run_fl_training
    from forecaus_grid_odeon.pipeline import using_real_ss_data

    is_real = bool(using_real_ss_data())

    # --- 1. day-ahead SS forecast benchmark (per-feeder + aggregate) ---
    bench = run_ss_benchmark()
    agg = bench["aggregate"]
    r2 = lambda x: round(float(x), 2)
    mape_naive = r2(agg.loc["seasonal_naive", "MAPE"])
    mape_sarimax = r2(agg.loc["sarimax", "MAPE"])
    mape_structured = r2(agg.loc["structured_gam", "MAPE"])
    mape_best = r2(min(mape_naive, mape_sarimax, mape_structured))
    n_feeders = int(len(bench["per_feeder"]))
    beats_naive = mape_structured <= mape_naive
    mape_gap = r2(mape_structured - MAPE_TARGET)

    # --- 2. federation across substations ---
    fl = run_fl_training(rounds=rounds)
    fa = fl["aggregate"]
    fl_local, fl_global, fl_central = r2(fa["local_MAPE"]), r2(fa["global_MAPE"]), r2(fa["centralised_MAPE"])
    cs = fl["cold_start"]
    cs_local, cs_global = r2(cs["local_MAPE"]), r2(cs["global_MAPE"])
    cs_benefit = round(cs_local - cs_global, 2)
    cs_n_train = int(cs["n_train"])
    n_nodes = int(fl["n_nodes"])
    n_rounds = int(len(fl["convergence"]))

    computed = {
        "mape_naive": mape_naive, "mape_sarimax": mape_sarimax,
        "mape_structured": mape_structured, "mape_best": mape_best,
        "mape_target": MAPE_TARGET, "mape_gap": mape_gap, "n_feeders": n_feeders,
        "fl_local": fl_local, "fl_global": fl_global, "fl_central": fl_central,
        "cs_local": cs_local, "cs_global": cs_global, "cs_benefit": cs_benefit,
        "cs_n_train": cs_n_train, "n_nodes": n_nodes, "n_rounds": n_rounds,
    }
    return {
        "computed": computed,
        "is_real": is_real,
        "source_label": REAL_LABEL if is_real else SYN_LABEL,
        "disclaimer": REAL_DISCLAIMER if is_real else SYN_DISCLAIMER,
        "narrative": build_narrative(computed, beats_naive),
        "table": _table(computed),
        "ss_aggregate": agg,
        "ss_per_feeder": bench["per_feeder"],
        "fl_result": fl,
    }


def build_narrative(c: dict, beats_naive: bool) -> str:
    """Prose narrative from computed values (every number traces; honest verdict)."""
    verdict = "beats" if beats_naive else "does not beat"
    return (
        f"Across {c['n_feeders']} secondary-substation feeders, the day-ahead "
        f"forecast benchmark gives an aggregate MAPE of {c['mape_naive']:.2f}% for "
        f"seasonal-naive, {c['mape_sarimax']:.2f}% for SARIMAX and "
        f"{c['mape_structured']:.2f}% for the interpretable structured model; the "
        f"best aggregate MAPE is {c['mape_best']:.2f}%. The structured model "
        f"{verdict} the seasonal-naive baseline ({c['mape_structured']:.2f}% vs "
        f"{c['mape_naive']:.2f}%); it stays above the <{c['mape_target']:.0f}% "
        f"target (gap {c['mape_gap']:.2f} pp) because single-feeder half-hourly "
        f"demand is far noisier than the aggregated load that target was set for.\n\n"
        f"Federating the structured model across {c['n_nodes']} substation nodes "
        f"over {c['n_rounds']} rounds moves the aggregate test MAPE from "
        f"{c['fl_local']:.2f}% (local-only) to {c['fl_global']:.2f}% "
        f"(federated-global), vs {c['fl_central']:.2f}% for a centralised model "
        f"trained on pooled data — without any raw data leaving a node. The "
        f"thin-history feeder (only {c['cs_n_train']} training rows) goes from "
        f"{c['cs_local']:.2f}% local to {c['cs_global']:.2f}% under federation, a "
        f"cold-start change of {c['cs_benefit']:.2f} pp."
    )


def _table(c: dict) -> str:
    lines = [
        "SS day-ahead forecast — aggregate MAPE [%]",
        f"  seasonal_naive : {c['mape_naive']:.2f}",
        f"  sarimax        : {c['mape_sarimax']:.2f}",
        f"  structured_gam : {c['mape_structured']:.2f}   "
        f"(beats naive: {c['mape_structured'] <= c['mape_naive']}; gap to <5%: {c['mape_gap']:.2f} pp)",
        f"  best           : {c['mape_best']:.2f}   (over {c['n_feeders']} feeders)",
        "",
        "Federated learning — aggregate test MAPE [%]",
        f"  local-only          : {c['fl_local']:.2f}",
        f"  federated-global    : {c['fl_global']:.2f}",
        f"  centralised (pooled): {c['fl_central']:.2f}",
        f"  cold start (n_train={c['cs_n_train']}): {c['cs_local']:.2f} -> {c['cs_global']:.2f}  ({c['cs_benefit']:+.2f} pp)",
        f"  nodes={c['n_nodes']}, rounds={c['n_rounds']}",
    ]
    return "\n".join(lines)


def build_benchmark_df(head: dict):
    """ODEON benchmark table — per-feeder AND aggregate rows. The ``dataset``
    column states the TRUE source (real UKPN name only on a real run)."""
    import numpy as np
    import pandas as pd

    agg, per, c = head["ss_aggregate"], head["ss_per_feeder"], head["computed"]
    ROLL, HOLD = "day-ahead rolling-origin", "last-day holdout, per-node"
    role = {"seasonal_naive": "baseline", "sarimax": "baseline",
            "structured_gam": "interpretable"}
    rows = []
    for m in ["seasonal_naive", "sarimax", "structured_gam"]:
        rows.append({"feeder": "ALL (aggregate)", "model": m, "role": role[m], "protocol": ROLL,
                     "MAPE_pct": round(float(agg.loc[m, "MAPE"]), 2),
                     "coverage": round(float(agg.loc[m, "coverage"]), 3)})
    for fid, t in per.items():
        for m in ["seasonal_naive", "sarimax", "structured_gam"]:
            rows.append({"feeder": fid, "model": m, "role": role[m], "protocol": ROLL,
                         "MAPE_pct": round(float(t.loc[m, "MAPE"]), 2),
                         "coverage": round(float(t.loc[m, "coverage"]), 3)})
    for r, key in [("local-only", "fl_local"), ("federated-global", "fl_global"),
                   ("centralised (pooled)", "fl_central")]:
        rows.append({"feeder": "ALL (aggregate)", "model": "structured_gam", "role": r,
                     "protocol": HOLD, "MAPE_pct": c[key], "coverage": np.nan})
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", head["source_label"])
    return df


def write_results_md(narrative: str, disclaimer: str = SYN_DISCLAIMER,
                     path: Path = RESULTS_MD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# Headline results\n\n"
        f"> {disclaimer}\n\n"
        "_Auto-generated by `scripts/reproduce_headline.py`; every number is checked "
        "against a computed value by the deterministic guard._\n\n"
        f"{narrative}\n"
    )
    path.write_text(body)
    return path


def main() -> int:
    from forecaus_grid_odeon.validation import validate_claims

    head = compute_headline()
    print(f"data source: {'REAL — ' if head['is_real'] else 'SYNTHETIC — '}{head['source_label']}\n")
    print(head["table"])
    print("\nNarrative:\n" + head["narrative"])

    res = validate_claims(head["narrative"], head["computed"])
    print(f"\nDeterministic guard: {res['n_numbers']} numbers, "
          f"{len(res['supported'])} supported, {len(res['unsupported'])} unsupported")
    write_results_md(head["narrative"], head["disclaimer"])
    build_benchmark_df(head).to_csv(BENCH_CSV, index=False)
    print(f"wrote {RESULTS_MD}\nwrote {BENCH_CSV}")

    if res["ok"]:
        print("VERDICT: PASS — every cited figure traces to a computed value.")
        return 0
    print("VERDICT: FAIL — unsupported (possibly hallucinated) numbers:")
    for u in res["unsupported"]:
        print(f"  {u['text']} ({u['value']})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
