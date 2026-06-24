#!/usr/bin/env python
"""Regenerate the federated-convergence, forecast->flex and edge figures from the
REAL UKPN LV-feeder data in ``data/raw/ss/`` (from ``make ingest-ss-real``).

Unlike the offline notebooks (which run on the committed synthetic fixtures for a
network-free CI), this regenerates the *committed* figures from real measured
data. It STOPs if no real data is present — it never relabels fixtures as real.

Writes:
  * notebooks/figures/03_federated_convergence.png
  * notebooks/figures/04_forecast_to_flex.png
  * notebooks/figures/edge_fit.csv + notebooks/figures/edge_fit.png

The federated paragraph of RESULTS.md is regenerated (with the same real numbers)
by ``scripts/reproduce_headline.py``; run it after this for a consistent set.

Run:  python scripts/regenerate_real_figures.py
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "notebooks" / "figures"
BLUE, RED = "#4c78a8", "#e45756"


def _require_real() -> None:
    from forecaus_grid_odeon.pipeline import using_real_ss_data
    if not using_real_ss_data():
        print("STOP — no real LV-feeder data found (and/or FORECAUS_OFFLINE is set).")
        print("Run `make ingest-ss-real` first (needs UKPN_API_KEY); this script never")
        print("relabels the synthetic fixtures as real.")
        raise SystemExit(2)


def federated_convergence() -> dict:
    """FL across the real feeders -> convergence figure; return the headline dict."""
    from forecaus_grid_odeon.fl import run_fl_training

    fl = run_fl_training(rounds=12)
    agg, cs = fl["aggregate"], fl["cold_start"]
    trace = fl["convergence"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(trace) + 1), trace, marker="o", color=BLUE, label="federated global")
    ax.axhline(agg["local_MAPE"], ls=":", color="grey", label=f"local-only ({agg['local_MAPE']:.1f}%)")
    ax.axhline(agg["centralised_MAPE"], ls="--", color=RED,
               label=f"centralised ({agg['centralised_MAPE']:.1f}%)")
    ax.set_xlabel("FedAvg round")
    ax.set_ylabel("aggregate test MAPE [%]")
    ax.set_title(f"Federated convergence — {fl['n_nodes']} real LV feeders")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_federated_convergence.png", dpi=130)
    plt.close(fig)
    print("saved", FIG / "03_federated_convergence.png")
    print(f"  FL: local {agg['local_MAPE']:.2f}% -> global {agg['global_MAPE']:.2f}% "
          f"(centralised {agg['centralised_MAPE']:.2f}%); "
          f"cold start {cs['local_MAPE']:.2f}% -> {cs['global_MAPE']:.2f}% "
          f"(+{cs['benefit']:.2f} pp); uplink floats {fl['uplink_param_floats']} "
          f"(no raw data leaves a node)")
    return {"aggregate": agg, "cold_start": cs}


def forecast_to_flex() -> dict:
    """Forecast -> risk-adjusted congestion schedule on a real feeder -> figure."""
    from forecaus_grid_odeon.flex import run_flex

    res = run_flex()
    sched, s = res["schedule"], res["summary"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sched.index, sched["forecast"], color=BLUE, label="forecast")
    ax.fill_between(sched.index, sched["lower"], sched["upper"], alpha=0.2, color=BLUE, label="interval")
    ax.plot(sched.index, sched["ss_limit"], ls="--", color="red", label="binding limit")
    ax2 = ax.twinx()
    ax2.bar(sched.index, sched["flex_down"], width=0.015, color=RED, alpha=0.6)
    ax.set_ylabel("load [kW]")
    ax2.set_ylabel("flex_down [kW]")
    ax.set_title(f"Forecast vs limit -> risk-adjusted down-flex (real feeder {res['feeder_id']})")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "04_forecast_to_flex.png", dpi=130)
    plt.close(fig)
    print("saved", FIG / "04_forecast_to_flex.png")
    print(f"  down-flex: peak {s['peak_flex_down']:.1f} kW, energy {s['energy_flex_down_kwh']:.1f} kWh "
          f"over {s['n_breach_down']} steps; risk-adjusted {s['energy_flex_down_kwh']:.1f} kWh vs "
          f"{res['summary_point']['energy_flex_down_kwh']:.1f} kWh point-only")
    return {"summary": s, "feeder": res["feeder_id"]}


def edge_benchmark() -> None:
    """Run the edge benchmark (real-aware subprocess) -> edge_fit.csv + edge_fit.png."""
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "edge_benchmark.py")],
                          capture_output=True, text=True)
    print(proc.stdout.strip().splitlines()[-1] if proc.stdout else "(no edge output)")
    if proc.returncode != 0:
        print("edge benchmark failed:\n", proc.stderr[-1500:])
        raise SystemExit(proc.returncode)

    table = pd.read_csv(FIG / "edge_fit.csv")
    budg = table[table["within_budget"].isin(["True", "False", True, False])].copy()
    budg["value"] = pd.to_numeric(budg["value"], errors="coerce")
    budg["pi4_budget"] = pd.to_numeric(budg["pi4_budget"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4))
    y = range(len(budg))
    ax.barh([i + 0.2 for i in y], budg["value"], height=0.4, color=BLUE, label="measured (Pi-4 est.)")
    ax.barh([i - 0.2 for i in y], budg["pi4_budget"], height=0.4, color=RED, alpha=0.6, label="Pi-4 budget")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{m} [{u}]" for m, u in zip(budg["metric"], budg["unit"])], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("value (log scale)")
    ax.set_title("Edge-deployability vs Raspberry Pi 4 budget (real-data model)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "edge_fit.png", dpi=130)
    plt.close(fig)
    print("saved", FIG / "edge_fit.png", "and", FIG / "edge_fit.csv")


def main() -> int:
    _require_real()
    print("=== regenerating real-data figures ===")
    federated_convergence()
    forecast_to_flex()
    edge_benchmark()
    print("\nDone. Run `python scripts/reproduce_headline.py` to refresh the RESULTS.md "
          "federated paragraph with the same real numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
