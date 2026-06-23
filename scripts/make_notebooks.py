"""Generate the four offline result notebooks under ``notebooks/``.

Each notebook runs top-to-bottom offline on the committed fixtures and writes its
figures to ``notebooks/figures/``. Notebook 03 also (re)builds the ODEON
benchmark CSV and RESULTS.md from a single deterministic computation, so the
figures, the benchmark table and RESULTS.md always agree.

Run from the repo root:  python scripts/make_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """\
import os
os.environ["FORECAUS_OFFLINE"] = "1"   # run fully offline on the committed fixtures
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
# Resolve figures/ whether executed from the repo root or from notebooks/.
FIG = Path("notebooks/figures") if Path("notebooks").is_dir() else Path("figures")
FIG.mkdir(parents=True, exist_ok=True)
print("offline:", os.environ["FORECAUS_OFFLINE"], "| figures ->", FIG)
"""


def _nb(cells):
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ---------------------------------------------------------------- notebook 01 --
NB01 = _nb([
    new_markdown_cell(
        "# 01 · SS day-ahead forecast benchmark\n\n"
        "Day-ahead (one-day-ahead) load forecasting at LV/secondary-substation level on the\n"
        "**UK Power Networks 'Smart Meter Consumption - LV Feeder'** public dataset "
        "(committed offline sample). Compares the interpretable **structured GAM** against the\n"
        "**seasonal-naive** and **SARIMAX** baselines via rolling-origin, reporting MAPE and\n"
        "conformal interval coverage. Runs fully offline."
    ),
    new_code_cell(HEADER),
    new_code_cell(
        "from forecaus_grid_odeon.eval import run_ss_benchmark, format_table\n"
        "bench = run_ss_benchmark()\n"
        "agg = bench['aggregate']\n"
        "print('Per-feeder count:', len(bench['per_feeder']))\n"
        "print('\\nAggregate (mean across feeders):')\n"
        "print(format_table(agg))"
    ),
    new_code_cell(
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))\n"
        "agg['MAPE'].plot.bar(ax=ax1, color='#4c78a8'); ax1.set_ylabel('MAPE [%]')\n"
        "ax1.set_title('Day-ahead MAPE by model'); ax1.tick_params(axis='x', rotation=20)\n"
        "agg['coverage'].plot.bar(ax=ax2, color='#54a24b'); ax2.axhline(0.9, ls='--', c='k', lw=1)\n"
        "ax2.set_ylabel('interval coverage'); ax2.set_title('Coverage (target 0.90)')\n"
        "ax2.tick_params(axis='x', rotation=20)\n"
        "fig.tight_layout(); fig.savefig(FIG / '01_ss_forecast_benchmark.png', dpi=130); plt.close(fig)\n"
        "print('saved', FIG / '01_ss_forecast_benchmark.png')"
    ),
])

# ---------------------------------------------------------------- notebook 02 --
NB02 = _nb([
    new_markdown_cell(
        "# 02 · SS causal layer\n\n"
        "The SS-level causal DAG (`temperature -> load`, confounded by the diurnal calendar),\n"
        "a DoWhy backdoor effect estimate with placebo / random-common-cause / subset refuters,\n"
        "and the causal-augmented vs correlational comparison across a (clearly-labelled,\n"
        "injected) regime change. Reported honestly, including the negative sign."
    ),
    new_code_cell(HEADER),
    new_code_cell(
        "from forecaus_grid_odeon.causal import build_ss_dag, draw_dag\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "draw_dag(build_ss_dag(), ax=ax)\n"
        "fig.savefig(FIG / '02_ss_dag.png', dpi=130, bbox_inches='tight'); plt.close(fig)\n"
        "print('saved', FIG / '02_ss_dag.png')"
    ),
    new_code_cell(
        "from forecaus_grid_odeon.eval import run_ss_causal_effect\n"
        "result, refute = run_ss_causal_effect(num_simulations=30)\n"
        "lo, hi = result['ci']\n"
        "print(f\"temp_c -> load_kw effect: {result['effect']:+.3f} kW/degC   90% CI [{lo:+.3f}, {hi:+.3f}]\")\n"
        "print('backdoor adjustment set:', sorted(result['backdoor_set']))\n"
        "print('\\nRefuters:')\n"
        "print(refute[['original_effect', 'new_effect', 'p_value', 'passed']].round(3).to_string())\n"
        "assert refute['passed'].all(), 'a refuter failed'"
    ),
    new_code_cell(
        "from forecaus_grid_odeon.eval import run_ss_break_experiment, make_break_figure\n"
        "exp = run_ss_break_experiment()\n"
        "print(exp['table'][['MAPE_normal', 'MAPE_break', 'MAPE_degradation']].round(3).to_string())\n"
        "make_break_figure(exp, exp['break_start'], str(FIG / '02_ss_causal_break.png'), ylabel='load [kW]')\n"
        "print('saved', FIG / '02_ss_causal_break.png')"
    ),
])

# ---------------------------------------------------------------- notebook 03 --
NB03 = _nb([
    new_markdown_cell(
        "# 03 · Federated learning across substations\n\n"
        "Federates the interpretable forecaster across feeders (Flower client + deterministic\n"
        "FedAvg) and builds the **ODEON benchmark table** (federated vs centralised vs local,\n"
        "plus the seasonal-naive / SARIMAX baselines) with MAPE/coverage on the named public\n"
        "LV dataset. Writes `figures/odeon_benchmark.csv` and regenerates `figures/RESULTS.md`\n"
        "from the same computation, so the figures, the table and the prose always agree."
    ),
    new_code_cell(HEADER),
    new_code_cell(
        "# Single deterministic recompute (same path as scripts/reproduce_headline.py).\n"
        "import sys, forecaus_grid_odeon\n"
        "REPO = Path(forecaus_grid_odeon.__file__).resolve().parents[2]\n"
        "sys.path.insert(0, str(REPO / 'scripts'))\n"
        "import reproduce_headline as rh\n"
        "head = rh.compute_headline()\n"
        "c, agg_ss, fl = head['computed'], head['ss_aggregate'], head['fl_result']\n"
        "print(head['table'])"
    ),
    new_code_cell(
        "# ODEON benchmark table: baselines (rolling-origin) + the structured model under the\n"
        "# federation protocol (local / federated-global / centralised). Protocol column keeps\n"
        "# the two evaluation harnesses honestly distinct.\n"
        "ROLL = 'day-ahead rolling-origin'\n"
        "HOLD = 'last-day holdout, per-node'\n"
        "rows = [\n"
        "    {'model': 'seasonal_naive', 'role': 'baseline', 'protocol': ROLL,\n"
        "     'MAPE_pct': float(agg_ss.loc['seasonal_naive', 'MAPE']), 'coverage': float(agg_ss.loc['seasonal_naive', 'coverage'])},\n"
        "    {'model': 'sarimax', 'role': 'baseline', 'protocol': ROLL,\n"
        "     'MAPE_pct': float(agg_ss.loc['sarimax', 'MAPE']), 'coverage': float(agg_ss.loc['sarimax', 'coverage'])},\n"
        "    {'model': 'structured_gam', 'role': 'interpretable (non-federated)', 'protocol': ROLL,\n"
        "     'MAPE_pct': float(agg_ss.loc['structured_gam', 'MAPE']), 'coverage': float(agg_ss.loc['structured_gam', 'coverage'])},\n"
        "    {'model': 'structured_gam', 'role': 'local-only', 'protocol': HOLD, 'MAPE_pct': c['fl_local'], 'coverage': np.nan},\n"
        "    {'model': 'structured_gam', 'role': 'federated-global', 'protocol': HOLD, 'MAPE_pct': c['fl_global'], 'coverage': np.nan},\n"
        "    {'model': 'structured_gam', 'role': 'centralised (pooled)', 'protocol': HOLD, 'MAPE_pct': c['fl_central'], 'coverage': np.nan},\n"
        "]\n"
        "odeon = pd.DataFrame(rows)\n"
        "odeon['MAPE_pct'] = odeon['MAPE_pct'].round(2)\n"
        "odeon['coverage'] = odeon['coverage'].round(3)\n"
        "odeon.insert(0, 'dataset', 'UKPN Smart Meter Consumption - LV Feeder')\n"
        "odeon.to_csv(FIG / 'odeon_benchmark.csv', index=False)\n"
        "print(odeon.to_string(index=False))\n"
        "print('\\nsaved', FIG / 'odeon_benchmark.csv')"
    ),
    new_code_cell(
        "# Cold start + convergence.\n"
        "print(f\"cold start - thin feeder (n_train={c['cs_n_train']}): \"\n"
        "      f\"{c['cs_local']:.2f}% local -> {c['cs_global']:.2f}% federated (+{c['cs_benefit']:.2f} pp)\")\n"
        "trace = fl['convergence']\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(range(1, len(trace) + 1), trace, marker='o', color='#4c78a8')\n"
        "ax.set_xlabel('FedAvg round'); ax.set_ylabel('aggregate global MAPE [%]')\n"
        "ax.set_title('Federated convergence'); ax.grid(alpha=0.3)\n"
        "fig.tight_layout(); fig.savefig(FIG / '03_federated_convergence.png', dpi=130); plt.close(fig)\n"
        "print('saved', FIG / '03_federated_convergence.png')"
    ),
    new_code_cell(
        "# Regenerate RESULTS.md from the SAME computed values and check every cited number\n"
        "# traces (deterministic guard) -> the CSV, the figures and RESULTS.md agree.\n"
        "from forecaus_grid_odeon.validation import validate_claims\n"
        "rh.write_results_md(head['narrative'], REPO / 'notebooks' / 'figures' / 'RESULTS.md')\n"
        "res = validate_claims(head['narrative'], head['computed'])\n"
        "assert res['ok'], res['unsupported']\n"
        "print('RESULTS.md regenerated; guard PASS -', res['n_numbers'], 'numbers all supported')"
    ),
])

# ---------------------------------------------------------------- notebook 04 --
NB04 = _nb([
    new_markdown_cell(
        "# 04 · Forecast -> flexibility\n\n"
        "Turns a Slice-3 day-ahead **interval** forecast for one feeder into a **risk-adjusted**\n"
        "congestion-relief schedule: down-flex sized against the UPPER forecast bound, under the\n"
        "binding (min of thermal / contractual) limit. Plots the schedule and reports volume +\n"
        "timing. Runs fully offline."
    ),
    new_code_cell(HEADER),
    new_code_cell(
        "from forecaus_grid_odeon.flex import run_flex\n"
        "res = run_flex()\n"
        "sched, s = res['schedule'], res['summary']\n"
        "print(f\"feeder {res['feeder_id']}  |  thermal {res['thermal_limit']:.1f} kW  |  \"\n"
        "      f\"contractual {res['contractual_limit']:.1f} kW  |  binding {res['binding_limit']:.1f} kW\")\n"
        "breach = sched[(sched['flex_down'] > 0) | (sched['flex_up'] > 0)]\n"
        "print(f\"\\n{len(breach)} active relief steps:\")\n"
        "print(breach[['forecast', 'upper', 'ss_limit', 'flex_down', 'flex_up']].round(1).to_string())"
    ),
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.plot(sched.index, sched['forecast'], color='#4c78a8', label='forecast')\n"
        "ax.fill_between(sched.index, sched['lower'], sched['upper'], alpha=0.2, color='#4c78a8', label='interval')\n"
        "ax.plot(sched.index, sched['ss_limit'], ls='--', color='red', label='binding limit')\n"
        "ax2 = ax.twinx()\n"
        "ax2.bar(sched.index, sched['flex_down'], width=0.015, color='#e45756', alpha=0.6)\n"
        "ax.set_ylabel('load [kW]'); ax2.set_ylabel('flex_down [kW]')\n"
        "ax.set_title('Forecast vs limit -> risk-adjusted down-flex'); ax.legend(loc='upper left')\n"
        "fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / '04_forecast_to_flex.png', dpi=130); plt.close(fig)\n"
        "print('saved', FIG / '04_forecast_to_flex.png')\n"
        "print(f\"down-flex: peak {s['peak_flex_down']:.1f} kW, energy {s['energy_flex_down_kwh']:.1f} kWh over {s['n_breach_down']} steps\")\n"
        "print(f\"risk adjustment: {s['energy_flex_down_kwh']:.1f} kWh vs {res['summary_point']['energy_flex_down_kwh']:.1f} kWh point-only\")"
    ),
])

NOTEBOOKS = {
    "01_ss_forecast_benchmark.ipynb": NB01,
    "02_ss_causal.ipynb": NB02,
    "03_federated_across_substations.ipynb": NB03,
    "04_forecast_to_flex.ipynb": NB04,
}


def main() -> None:
    (NB_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        path = NB_DIR / name
        nbformat.write(nb, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
