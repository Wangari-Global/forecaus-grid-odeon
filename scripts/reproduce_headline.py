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
    r2 = lambda x: round(float(x), 2)
    r3 = lambda x: round(float(x), 3)

    # --- 1a. SUBSTATION-TOTAL benchmark (the Challenge-4 target = HEADLINE) ---
    sbench = run_ss_benchmark(level="substation")
    sagg = sbench["aggregate"]
    s_naive = r2(sagg.loc["seasonal_naive", "MAPE"])
    s_sarimax = r2(sagg.loc["sarimax", "MAPE"])
    s_structured = r2(sagg.loc["structured_gam", "MAPE"])
    s_best = r2(min(s_naive, s_sarimax, s_structured))
    s_coverage = r3(sagg.loc["structured_gam", "coverage"])
    n_substations = int(len(sbench["per_feeder"]))
    s_beats = s_structured <= s_naive
    s_gap = r2(s_structured - MAPE_TARGET)

    # --- 1b. PER-FEEDER benchmark (the harder lower bound) ---
    bench = run_ss_benchmark(level="feeder")
    agg = bench["aggregate"]
    mape_naive = r2(agg.loc["seasonal_naive", "MAPE"])
    mape_sarimax = r2(agg.loc["sarimax", "MAPE"])
    mape_structured = r2(agg.loc["structured_gam", "MAPE"])
    mape_best = r2(min(mape_naive, mape_sarimax, mape_structured))
    n_feeders = int(len(bench["per_feeder"]))
    beats_naive = mape_structured <= mape_naive
    mape_gap = r2(mape_structured - MAPE_TARGET)

    # --- 2. federation across SUBSTATIONS (SS-total nodes — Challenge-4 target) ---
    fl = run_fl_training(rounds=rounds, level="substation")
    fa = fl["aggregate"]
    fl_local, fl_global, fl_central = r2(fa["local_MAPE"]), r2(fa["global_MAPE"]), r2(fa["centralised_MAPE"])
    cs = fl["cold_start"]
    cs_local, cs_global = r2(cs["local_MAPE"]), r2(cs["global_MAPE"])
    cs_benefit = round(cs_local - cs_global, 2)
    cs_n_train = int(cs["n_train"])
    n_nodes = int(fl["n_nodes"])
    n_rounds = int(len(fl["convergence"]))

    # --- 3. forecast -> flex congestion schedule on the busiest SS-total ---
    from forecaus_grid_odeon.flex import run_flex
    flex = run_flex(level="substation")
    fs, fsp = flex["summary"], flex["summary_point"]
    flex_limit = r2(flex["binding_limit"])
    flex_peak = r2(fs["peak_flex_down"])
    flex_energy = r2(fs["energy_flex_down_kwh"])
    flex_steps = int(fs["n_breach_down"])
    flex_energy_point = r2(fsp["energy_flex_down_kwh"])

    computed = {
        # HEADLINE: substation-total (Challenge-4 target)
        "s_naive": s_naive, "s_sarimax": s_sarimax, "s_structured": s_structured,
        "s_best": s_best, "s_coverage": s_coverage, "s_gap": s_gap,
        "n_substations": n_substations,
        # per-feeder (harder lower bound)
        "mape_naive": mape_naive, "mape_sarimax": mape_sarimax,
        "mape_structured": mape_structured, "mape_best": mape_best,
        "mape_gap": mape_gap, "n_feeders": n_feeders,
        "mape_target": MAPE_TARGET,
        # federation
        "fl_local": fl_local, "fl_global": fl_global, "fl_central": fl_central,
        "cs_local": cs_local, "cs_global": cs_global, "cs_benefit": cs_benefit,
        "cs_n_train": cs_n_train, "n_nodes": n_nodes, "n_rounds": n_rounds,
        # forecast -> flex (busiest SS-total, illustrative transformer limit)
        "flex_limit": flex_limit, "flex_peak": flex_peak, "flex_energy": flex_energy,
        "flex_steps": flex_steps, "flex_energy_point": flex_energy_point,
    }
    return {
        "computed": computed,
        "is_real": is_real,
        "source_label": REAL_LABEL if is_real else SYN_LABEL,
        "disclaimer": REAL_DISCLAIMER if is_real else SYN_DISCLAIMER,
        "narrative": build_narrative(computed, s_beats),
        "table": _table(computed),
        "ss_aggregate": agg,                 # per-feeder aggregate (notebook back-compat)
        "ss_per_feeder": bench["per_feeder"],
        "feeder_bench": bench,
        "agg_bench": sbench,
        "fl_result": fl,
    }


def build_narrative(c: dict, s_beats: bool) -> str:
    """Prose narrative from computed values (every number traces; honest verdict).

    HEADLINE is the substation-total level (the network-operator target); the
    per-feeder level is reported as the harder lower bound. (No bare digits in the
    prose other than computed values, so the deterministic guard stays clean.)"""
    verdict = "beats" if s_beats else "does not beat"
    return (
        f"At the secondary-substation-total level — the granularity the challenge "
        f"targets — the day-ahead benchmark across {c['n_substations']} substations "
        f"gives an aggregate MAPE of {c['s_naive']:.2f}% for seasonal-naive, "
        f"{c['s_sarimax']:.2f}% for SARIMAX and {c['s_structured']:.2f}% for the "
        f"interpretable structured model, at {c['s_coverage']:.3f} interval "
        f"coverage. The structured model {verdict} the seasonal-naive baseline "
        f"({c['s_structured']:.2f}% vs {c['s_naive']:.2f}%). It stays above the "
        f"<{c['mape_target']:.0f}% target (gap {c['s_gap']:.2f} pp) because these "
        f"substations roll up only a handful of LV feeders (tens of homes), so they "
        f"are still far spikier than whole-network load — the level that target was "
        f"set for; much larger aggregation would be needed to approach it.\n\n"
        f"At the harder per-feeder level ({c['n_feeders']} feeders, a lower bound) "
        f"the structured model gives {c['mape_structured']:.2f}% MAPE versus "
        f"{c['mape_naive']:.2f}% for seasonal-naive and {c['mape_sarimax']:.2f}% for "
        f"SARIMAX.\n\n"
        f"Federating the structured model across {c['n_nodes']} substation nodes "
        f"(one SS-total series each) over {c['n_rounds']} rounds moves the aggregate "
        f"test MAPE from {c['fl_local']:.2f}% (local-only) to {c['fl_global']:.2f}% "
        f"(federated-global), vs {c['fl_central']:.2f}% for a centralised model "
        f"trained on pooled data — without any raw data leaving a node (only "
        f"parameter vectors are exchanged). The thin-history substation (only "
        f"{c['cs_n_train']} training rows) goes from {c['cs_local']:.2f}% local to "
        f"{c['cs_global']:.2f}% under federation, a cold-start change of "
        f"{c['cs_benefit']:.2f} pp.\n\n"
        f"Turning the SS-total day-ahead interval forecast into congestion relief: "
        f"on the busiest substation, against an illustrative transformer firm-rating "
        f"limit of {c['flex_limit']:.2f} kW (a percentile of its own historical "
        f"load, pending the pilot's real rating), the schedule sizes "
        f"{c['flex_energy']:.2f} kWh of risk-adjusted down-flex (peak "
        f"{c['flex_peak']:.2f} kW) across {c['flex_steps']} half-hours — versus "
        f"{c['flex_energy_point']:.2f} kWh sized against the point forecast alone, "
        f"the headroom the prediction interval buys."
    )


def _table(c: dict) -> str:
    lines = [
        f"HEADLINE — substation-total day-ahead MAPE [%]  (over {c['n_substations']} substations, Challenge-4 target)",
        f"  seasonal_naive : {c['s_naive']:.2f}",
        f"  sarimax        : {c['s_sarimax']:.2f}",
        f"  structured_gam : {c['s_structured']:.2f}   "
        f"(beats naive: {c['s_structured'] <= c['s_naive']}; coverage {c['s_coverage']:.3f}; gap to <5%: {c['s_gap']:.2f} pp)",
        "",
        f"per-feeder day-ahead MAPE [%]  (over {c['n_feeders']} feeders, harder lower bound)",
        f"  seasonal_naive : {c['mape_naive']:.2f}",
        f"  sarimax        : {c['mape_sarimax']:.2f}",
        f"  structured_gam : {c['mape_structured']:.2f}",
        "",
        "Federated learning — aggregate test MAPE [%]",
        f"  local-only          : {c['fl_local']:.2f}",
        f"  federated-global    : {c['fl_global']:.2f}",
        f"  centralised (pooled): {c['fl_central']:.2f}",
        f"  cold start (n_train={c['cs_n_train']}): {c['cs_local']:.2f} -> {c['cs_global']:.2f}  ({c['cs_benefit']:+.2f} pp)",
        f"  nodes={c['n_nodes']}, rounds={c['n_rounds']}",
    ]
    return "\n".join(lines)


SS_LEVEL = "substation-total (Challenge-4 target)"
FEEDER_LEVEL = "per-feeder (harder case)"


def build_benchmark_df(head: dict):
    """ODEON benchmark table — BOTH levels, per-unit AND aggregate rows, with
    MAE/RMSE/MAPE + coverage. The ``level`` column labels substation-total (the
    Challenge-4 target / headline) vs per-feeder (harder lower bound); the
    ``dataset`` column states the TRUE source (real UKPN name only on a real run)."""
    import numpy as np
    import pandas as pd

    c = head["computed"]
    ROLL, HOLD = "day-ahead rolling-origin", "last-day holdout, per-node"
    role = {"seasonal_naive": "baseline", "sarimax": "baseline",
            "structured_gam": "interpretable"}

    def _row(level, unit, m, s):
        return {"level": level, "feeder": unit, "model": m, "role": role[m], "protocol": ROLL,
                "MAPE_pct": round(float(s["MAPE"]), 2), "MAE_kw": round(float(s["MAE"]), 3),
                "RMSE_kw": round(float(s["RMSE"]), 3), "coverage": round(float(s["coverage"]), 3)}

    def _level_rows(bench, level):
        out = [_row(level, "ALL (aggregate)", m, bench["aggregate"].loc[m])
               for m in ["seasonal_naive", "sarimax", "structured_gam"]]
        for uid, t in bench["per_feeder"].items():
            out += [_row(level, uid, m, t.loc[m]) for m in ["seasonal_naive", "sarimax", "structured_gam"]]
        return out

    rows = _level_rows(head["agg_bench"], SS_LEVEL) + _level_rows(head["feeder_bench"], FEEDER_LEVEL)
    for r, key in [("local-only", "fl_local"), ("federated-global", "fl_global"),
                   ("centralised (pooled)", "fl_central")]:
        rows.append({"level": SS_LEVEL, "feeder": "ALL (aggregate)", "model": "structured_gam",
                     "role": r, "protocol": HOLD, "MAPE_pct": c[key],
                     "MAE_kw": np.nan, "RMSE_kw": np.nan, "coverage": np.nan})
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
