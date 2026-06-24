"""CLI: ingest | forecast | causal | benchmark | ingest-ss | flex | fl-demo.

ingest/forecast/causal/benchmark run on the public proxy data (shared core).
ingest-ss/flex/fl-demo are the ODEON-specific layers.
"""
import argparse

HORIZON = 24  # 1-day-ahead hourly forecast


def _cmd_ingest():
    from .ingest import run
    run()


def _cmd_forecast():
    from .eval import format_table, summary
    from .forecast import default_models
    from .pipeline import TARGET, load_frame

    frame = load_frame()
    train, test = frame.iloc[:-HORIZON], frame.iloc[-HORIZON:]
    exog = [c for c in frame.columns if c != TARGET]
    print(f"[forecast] frame={frame.shape}, train={len(train)}, horizon={len(test)}")
    rows = {}
    for name, model in default_models(exog_cols=exog).items():
        pred = model(train, test, TARGET)
        rows[name] = summary(test[TARGET], pred["yhat"], pred["lower"], pred["upper"])
    import pandas as pd
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    print(format_table(table))


def _cmd_causal():
    import warnings
    from .causal import run_causal
    from .causal.graph import OUTCOME, TREATMENT
    from .pipeline import load_frame

    warnings.filterwarnings("ignore")
    frame = load_frame()
    print(f"[causal] frame={frame.shape}; effect of {TREATMENT} -> {OUTCOME}")
    result, table = run_causal(frame, TREATMENT, OUTCOME)
    lo, hi = result["ci"]
    print(f"  effect: {result['effect']:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  ({result['method']})")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"Overall: {int(table['passed'].sum())}/{len(table)} refuters passed")


def _cmd_benchmark():
    from .eval import format_table, rolling_origin
    from .forecast import benchmark_models
    from .pipeline import TARGET, load_frame

    frame = load_frame()
    exog = [c for c in frame.columns if c != TARGET]
    print(f"[benchmark] frame={frame.shape}, horizon={HORIZON}")
    table = rolling_origin(frame, benchmark_models(exog_cols=exog), horizon=HORIZON, target=TARGET)
    print(format_table(table))


def _cmd_ingest_ss():
    """Ingest real per-feeder LV demand (UKPN smart-meter LV-feeder dataset)."""
    from .ingest_ss import ingest_ss

    out = ingest_ss()
    print(f"[ingest-ss] wrote {len(out)} per-feeder parquet files under data/raw/ss/")


def _cmd_ingest_ss_real():
    """REAL SS/LV-feeder demand -> data/raw/ss (tries documented sources in order).

    Writes per-feeder parquet on the load_kw contract, prints a sanity report, and
    STOPs (printing exactly what is needed) if no source yields real record-level
    data — fixtures are never relabelled as real.
    """
    from .ingest_ss.real import ingest_ss_real

    ingest_ss_real()


def _cmd_forecast_ss():
    """Day-ahead SS-level benchmark: structured GAM vs seasonal_naive vs SARIMAX."""
    import warnings

    from .eval import format_table, run_ss_benchmark

    warnings.filterwarnings("ignore")
    print("[forecast-ss] day-ahead per-feeder benchmark on the UKPN LV-feeder data")
    result = run_ss_benchmark()

    for fid, table in result["per_feeder"].items():
        print(f"\n--- feeder {fid} ---")
        print(format_table(table))

    print("\n=== AGGREGATE (mean across feeders) ===")
    print(format_table(result["aggregate"]))

    best = result["aggregate"]["MAPE"].min()
    print(f"\nbest aggregate MAPE: {best:.2f}%  (target < 5%)")

    fid, params = result["params"]
    print(f"\nstructured_gam parameters (feeder {fid}) — auditable & federatable (Slice 5):")
    print(f"  {len(params)} named coefficients")
    for k in list(params)[:8]:
        print(f"    {k:24s}: {params[k]:+.4f}")


def _cmd_causal_ss():
    """SS-level causal effect (temp -> load) + refuters + causal-vs-correlational
    on a (clearly-labelled, injected) regime change. Reported honestly."""
    import warnings

    from .causal import SS_OUTCOME, SS_TREATMENT
    from .eval import make_break_figure, run_ss_break_experiment, run_ss_causal_effect

    warnings.filterwarnings("ignore")

    # 1) Effect estimate + refuters on the SS DAG.
    print(f"[causal-ss] SS DAG effect of {SS_TREATMENT} -> {SS_OUTCOME} "
          "(backdoor-adjusted for the diurnal calendar)")
    result, table = run_ss_causal_effect(num_simulations=50)
    lo, hi = result["ci"]
    effect = result["effect"]
    print(f"  effect: {effect:+.3f} kW/degC   90% CI [{lo:+.3f}, {hi:+.3f}]   ({result['method']})")
    print(f"  backdoor adjustment set: {sorted(result['backdoor_set'])}")
    sign = "negative (colder -> more load; heating-dominated)" if effect < 0 else \
           "positive (warmer -> more load; cooling-dominated)"
    print(f"  sign: {sign}")
    print("\n  refuter table (pass = effect survives the stress test):")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    n_pass = int(table["passed"].sum())
    print(f"  refuters passed: {n_pass}/{len(table)}"
          + ("" if n_pass == len(table) else "  <-- reported honestly (not all passed)"))

    # 2) Causal-augmented vs correlational across the regime change.
    print("\n[causal-ss] causal-augmented vs correlational across an INJECTED "
          "PV/EV-penetration regime change (no documented sub-feeder break in the sample)")
    exp = run_ss_break_experiment()
    htable = exp["table"]
    cols = [c for c in ["MAPE_overall", "MAPE_normal", "MAPE_break", "MAPE_degradation",
                        "coverage_break"] if c in htable.columns]
    print(htable[cols].to_string(float_format=lambda x: f"{x:.3f}"))

    path = make_break_figure(exp, exp["break_start"], "data/ss_causal_break.png",
                             ylabel="load [kW]")
    print(f"\n  figure -> {path}")

    caus, corr = htable.loc["causal_augmented"], htable.loc["correlational_ml"]
    print("\n  honest read-out:")
    print(f"    normal regime : correlational MAPE {corr['MAPE_normal']:.2f}% "
          f"vs causal {caus['MAPE_normal']:.2f}%  "
          f"({'correlational wins in-distribution' if corr['MAPE_normal'] < caus['MAPE_normal'] else 'causal wins'})")
    print(f"    break  regime : correlational MAPE {corr['MAPE_break']:.2f}% "
          f"vs causal {caus['MAPE_break']:.2f}%  "
          f"(degradation x{corr['MAPE_degradation']:.1f} vs x{caus['MAPE_degradation']:.1f})")
    verdict = ("causal-augmented degrades LESS on the break"
               if caus["MAPE_degradation"] < corr["MAPE_degradation"]
               else "causal did NOT help here")
    print(f"    verdict: {verdict}.")


def _cmd_flex():
    """Demo the deterministic flexibility-need calc on a synthetic SS day."""
    import pandas as pd
    from .flex import flexibility_need

    idx = pd.date_range("2026-01-01", periods=24, freq="h")
    forecast = pd.Series([300, 290, 285, 280, 285, 300, 340, 420, 520, 560, 540, 510,
                          500, 505, 520, 560, 640, 700, 680, 600, 500, 430, 380, 330],
                         index=idx, dtype=float)
    out = flexibility_need(forecast, ss_limit=600.0)
    need = out[out["flex_down"] > 0]
    print("[flex] synthetic SS day; limit=600 kW")
    print(f"  hours needing relief : {len(need)}")
    print(f"  peak flex_down       : {out['flex_down'].max():.0f} kW")
    print(f"  total flex energy    : {out['flex_down'].sum():.0f} kWh")


def _cmd_fl_demo():
    """Demo the deterministic FedAvg reference over toy node parameters."""
    from .fl import fedavg
    nodes = [
        {"intercept": 10.0, "temp_coef": -2.0, "lag24": 0.8},
        {"intercept": 12.0, "temp_coef": -1.6, "lag24": 0.7},
        {"intercept": 11.0, "temp_coef": -1.8, "lag24": 0.9},
    ]
    glob = fedavg(nodes, weights=[100, 200, 100])
    print("[fl-demo] FedAvg over 3 SS-node parameter sets (sample-weighted):")
    for k, v in glob.items():
        print(f"  {k:10s}: {v:.4f}")
    print("  (production path consumes ODEON FL Engine — see fl/federated.py)")


def _cmd_fl_train():
    """Federate the transparent forecaster across feeders (Flower client + the
    deterministic fedavg reference). Prints global vs centralised vs local +
    a convergence trace. Production path = ODEON FL Engine (fl/OdeonFLClient)."""
    import warnings

    from .fl import run_fl_training

    warnings.filterwarnings("ignore")
    res = run_fl_training(rounds=12)

    print(f"[fl-train] {res['n_nodes']}-node offline federation on the SS fixtures "
          f"(Flower clients + deterministic fedavg); {res['n_params']} parameters/node")
    print(f"  privacy: only parameter vectors cross the boundary — each uplink is "
          f"{res['uplink_param_floats']} floats vs >= {res['min_node_samples']} local samples")

    print("\n  per-feeder test MAPE [%] (local-only vs federated-global vs centralised):")
    print(res["per_feeder"].to_string(float_format=lambda x: f"{x:.2f}"))

    agg = res["aggregate"]
    print(f"\n  aggregate: local {agg['local_MAPE']:.2f}%  |  "
          f"global(federated) {agg['global_MAPE']:.2f}%  |  "
          f"centralised(pooled, reference) {agg['centralised_MAPE']:.2f}%")

    cs = res["cold_start"]
    if cs:
        verdict = "helps" if cs["benefit"] > 0 else "does NOT help (reported honestly)"
        print(f"\n  cold start — thin-history feeder {cs['feeder']} (n_train={cs['n_train']}): "
              f"local {cs['local_MAPE']:.2f}% -> global {cs['global_MAPE']:.2f}%  "
              f"(benefit {cs['benefit']:+.2f} pp; federation {verdict})")

    trace = res["convergence"]
    print("\n  round-by-round convergence (aggregate global MAPE [%]):")
    print("    " + "  ".join(f"r{i+1}:{m:.2f}" for i, m in enumerate(trace)))


def _cmd_flex_run():
    """Forecast -> risk-adjusted congestion-relief schedule for one SS feeder."""
    import warnings

    from .flex import run_flex

    warnings.filterwarnings("ignore")
    res = run_flex()
    sched, s, sp = res["schedule"], res["summary"], res["summary_point"]

    print(f"[flex-run] feeder {res['feeder_id']} — day-ahead Slice-3 interval forecast "
          "-> risk-adjusted flexibility")
    print(f"  limits: thermal {res['thermal_limit']:.1f} kW | contractual "
          f"{res['contractual_limit']:.1f} kW | binding {res['binding_limit']:.1f} kW")
    print(f"  horizon: {len(sched)} steps ({s['dt_hours']:.2f} h each)")

    breaches = sched[(sched["flex_down"] > 0) | (sched["flex_up"] > 0)]
    print(f"\n  congestion-relief schedule (timing x volume; {len(breaches)} active steps):")
    if len(breaches):
        view = breaches[["forecast", "upper", "ss_limit", "flex_down", "flex_up"]]
        print(view.to_string(float_format=lambda x: f"{x:.1f}"))
    else:
        print("    (no limit breach in the forecast horizon)")

    print(f"\n  down-flex (curtailment): peak {s['peak_flex_down']:.1f} kW, "
          f"energy {s['energy_flex_down_kwh']:.1f} kWh over {s['n_breach_down']} steps")
    if s["first_breach_down"] is not None:
        print(f"    window: {s['first_breach_down']} .. {s['last_breach_down']}")
    print(f"  up-flex (reverse-power relief): peak {s['peak_flex_up']:.1f} kW, "
          f"energy {s['energy_flex_up_kwh']:.1f} kWh over {s['n_breach_up']} steps")

    extra = s["energy_flex_down_kwh"] - sp["energy_flex_down_kwh"]
    print(f"\n  risk adjustment: sizing against the UPPER interval procures "
          f"{s['energy_flex_down_kwh']:.1f} kWh vs {sp['energy_flex_down_kwh']:.1f} kWh "
          f"point-only (+{extra:.1f} kWh headroom for forecast uncertainty)")


_COMMANDS = {
    "ingest": _cmd_ingest, "forecast": _cmd_forecast, "causal": _cmd_causal,
    "benchmark": _cmd_benchmark, "ingest-ss": _cmd_ingest_ss,
    "ingest-ss-real": _cmd_ingest_ss_real,
    "forecast-ss": _cmd_forecast_ss, "causal-ss": _cmd_causal_ss,
    "flex": _cmd_flex, "flex-run": _cmd_flex_run,
    "fl-demo": _cmd_fl_demo, "fl-train": _cmd_fl_train,
}


def main():
    p = argparse.ArgumentParser(prog="forecaus-grid-odeon")
    p.add_argument("cmd", choices=list(_COMMANDS))
    args = p.parse_args()
    _COMMANDS[args.cmd]()


if __name__ == "__main__":
    main()
