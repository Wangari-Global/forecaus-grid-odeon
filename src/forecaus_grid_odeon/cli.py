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
    print("[ingest-ss] ODEON SS-level adapters are stubs (API pending, help-desk Q1/Q4).")
    print("            See forecaus_grid_odeon/ingest_ss/odeon_api.py")


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


_COMMANDS = {
    "ingest": _cmd_ingest, "forecast": _cmd_forecast, "causal": _cmd_causal,
    "benchmark": _cmd_benchmark, "ingest-ss": _cmd_ingest_ss, "flex": _cmd_flex,
    "fl-demo": _cmd_fl_demo,
}


def main():
    p = argparse.ArgumentParser(prog="forecaus-grid-odeon")
    p.add_argument("cmd", choices=list(_COMMANDS))
    args = p.parse_args()
    _COMMANDS[args.cmd]()


if __name__ == "__main__":
    main()
