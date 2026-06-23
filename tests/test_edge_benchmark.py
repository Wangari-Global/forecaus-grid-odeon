"""Smoke test for scripts/edge_benchmark.py: it runs, writes the table, and
reports the interpretable model within a Pi 4-class envelope."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "edge_benchmark.py"


def test_edge_benchmark_runs_and_fits_envelope(tmp_path):
    out_csv = tmp_path / "edge_fit.csv"
    env = {"PATH": __import__("os").environ.get("PATH", ""),
           "EDGE_FIT_CSV": str(out_csv), "FORECAUS_OFFLINE": "1"}

    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(REPO),
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert out_csv.exists()

    table = pd.read_csv(out_csv)
    assert {"metric", "value", "unit", "pi4_budget", "within_budget"} <= set(table.columns)
    metrics = set(table["metric"])
    assert {"artifact_size", "train_time_pi4_est",
            "inference_latency_pi4_est", "peak_workload_mem"} <= metrics

    # Every budgeted metric is within the Pi 4-class envelope -> PASS.
    budgeted = table[table["within_budget"].isin(["True", "False", True, False])]
    assert len(budgeted) >= 4
    assert (budgeted["within_budget"].astype(str) == "True").all()
    assert "VERDICT: PASS" in proc.stdout
