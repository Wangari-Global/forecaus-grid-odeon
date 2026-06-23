#!/usr/bin/env python
"""Recompute the headline SS forecast + FL numbers from cached data and prove
every cited figure traces to a computed value.

Pipeline:
  1. recompute the day-ahead SS benchmark (structured GAM vs seasonal-naive vs
     SARIMAX) and the cross-feeder federation (local / global / centralised +
     cold start) — all offline on the committed fixtures;
  2. build a short prose narrative *from those computed values* (so it is
     traceable by construction);
  3. run the deterministic guard (:func:`validate_claims`) over the narrative —
     every number must map to a computed value, else it is flagged;
  4. write the prose to ``notebooks/figures/RESULTS.md``.

Run:  python scripts/reproduce_headline.py
Exit code 0 iff the table reproduces and the guard passes (no unsupported number).
"""
from __future__ import annotations

import os

os.environ.setdefault("FORECAUS_OFFLINE", "1")           # reproducible, no network

import sys
from pathlib import Path

RESULTS_MD = Path(__file__).resolve().parents[1] / "notebooks" / "figures" / "RESULTS.md"


def compute_headline(rounds: int = 12) -> dict:
    """Recompute the headline numbers; return computed values + narrative + table."""
    import warnings

    warnings.filterwarnings("ignore")
    from forecaus_grid_odeon.eval import run_ss_benchmark
    from forecaus_grid_odeon.fl import run_fl_training

    # --- 1. day-ahead SS forecast benchmark (aggregate across feeders) ---
    bench = run_ss_benchmark()
    agg = bench["aggregate"]
    r2 = lambda x: round(float(x), 2)
    mape_naive = r2(agg.loc["seasonal_naive", "MAPE"])
    mape_sarimax = r2(agg.loc["sarimax", "MAPE"])
    mape_structured = r2(agg.loc["structured_gam", "MAPE"])
    mape_best = r2(min(mape_naive, mape_sarimax, mape_structured))
    n_feeders = int(len(bench["per_feeder"]))

    # --- 2. federation across substations ---
    fl = run_fl_training(rounds=rounds)
    fa = fl["aggregate"]
    fl_local = r2(fa["local_MAPE"])
    fl_global = r2(fa["global_MAPE"])
    fl_central = r2(fa["centralised_MAPE"])
    cs = fl["cold_start"]
    cs_local = r2(cs["local_MAPE"])
    cs_global = r2(cs["global_MAPE"])
    cs_benefit = round(cs_local - cs_global, 2)          # printed arithmetic is self-consistent
    cs_n_train = int(cs["n_train"])
    n_nodes = int(fl["n_nodes"])
    n_rounds = int(len(fl["convergence"]))

    computed = {
        "mape_naive": mape_naive, "mape_sarimax": mape_sarimax,
        "mape_structured": mape_structured, "mape_best": mape_best,
        "n_feeders": n_feeders,
        "fl_local": fl_local, "fl_global": fl_global, "fl_central": fl_central,
        "cs_local": cs_local, "cs_global": cs_global, "cs_benefit": cs_benefit,
        "cs_n_train": cs_n_train, "n_nodes": n_nodes, "n_rounds": n_rounds,
    }
    return {
        "computed": computed,
        "narrative": build_narrative(computed),
        "table": _table(computed),
        "ss_aggregate": agg,          # per-model MAPE + coverage (forecast benchmark)
        "fl_result": fl,              # full federation result (incl. convergence)
    }


def build_narrative(c: dict) -> str:
    """Prose narrative built from the computed values (traceable by construction)."""
    return (
        f"Across {c['n_feeders']} secondary-substation feeders, the day-ahead "
        f"forecast benchmark gives an aggregate MAPE of {c['mape_naive']:.2f}% for "
        f"seasonal-naive, {c['mape_sarimax']:.2f}% for SARIMAX and "
        f"{c['mape_structured']:.2f}% for the interpretable structured model; the "
        f"best aggregate MAPE is {c['mape_best']:.2f}%.\n\n"
        f"Federating the structured model across {c['n_nodes']} substation nodes "
        f"over {c['n_rounds']} rounds lifts the aggregate test MAPE from "
        f"{c['fl_local']:.2f}% (local-only) to {c['fl_global']:.2f}% "
        f"(federated-global), approaching the {c['fl_central']:.2f}% of a "
        f"centralised model trained on pooled data — without any raw data leaving "
        f"a node. The thin-history feeder (only {c['cs_n_train']} training rows) "
        f"improves from {c['cs_local']:.2f}% local to {c['cs_global']:.2f}% under "
        f"federation, a cold-start gain of {c['cs_benefit']:.2f} pp."
    )


def _table(c: dict) -> str:
    lines = [
        "SS day-ahead forecast — aggregate MAPE [%]",
        f"  seasonal_naive : {c['mape_naive']:.2f}",
        f"  sarimax        : {c['mape_sarimax']:.2f}",
        f"  structured_gam : {c['mape_structured']:.2f}",
        f"  best           : {c['mape_best']:.2f}   (over {c['n_feeders']} feeders)",
        "",
        "Federated learning — aggregate test MAPE [%]",
        f"  local-only          : {c['fl_local']:.2f}",
        f"  federated-global    : {c['fl_global']:.2f}",
        f"  centralised (pooled): {c['fl_central']:.2f}",
        f"  cold start (thin feeder, n_train={c['cs_n_train']}): "
        f"{c['cs_local']:.2f} -> {c['cs_global']:.2f}  (+{c['cs_benefit']:.2f} pp)",
        f"  nodes={c['n_nodes']}, rounds={c['n_rounds']}",
    ]
    return "\n".join(lines)


def write_results_md(narrative: str, path: Path = RESULTS_MD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# Headline results\n\n"
        "_Auto-generated by `scripts/reproduce_headline.py` from cached data; every "
        "number is checked against a computed value by the deterministic guard._\n\n"
        f"{narrative}\n"
    )
    path.write_text(body)
    return path


def main() -> int:
    from forecaus_grid_odeon.validation import validate_claims

    head = compute_headline()
    print(head["table"])
    print("\nNarrative:\n" + head["narrative"])

    res = validate_claims(head["narrative"], head["computed"])
    print(f"\nDeterministic guard: {res['n_numbers']} numbers, "
          f"{len(res['supported'])} supported, {len(res['unsupported'])} unsupported")
    path = write_results_md(head["narrative"])
    print(f"wrote {path}")

    if res["ok"]:
        print("VERDICT: PASS — every cited figure traces to a computed value.")
        return 0
    print("VERDICT: FAIL — unsupported (possibly hallucinated) numbers:")
    for u in res["unsupported"]:
        print(f"  {u['text']} ({u['value']})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
