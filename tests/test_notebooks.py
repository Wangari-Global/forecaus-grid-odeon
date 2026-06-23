"""Acceptance: the result notebooks execute top-to-bottom clean via nbconvert
and regenerate their figures + the ODEON benchmark CSV. Skipped when the
``[notebooks]`` extra (jupyter/nbconvert) is not installed."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
NOTEBOOKS = [
    "01_ss_forecast_benchmark.ipynb",
    "02_ss_causal.ipynb",
    "03_federated_across_substations.ipynb",
    "04_forecast_to_flex.ipynb",
]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nbconvert") is None
    or importlib.util.find_spec("ipykernel") is None,
    reason="requires the [notebooks] extra (jupyter/nbconvert/ipykernel)",
)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executes_clean(name, tmp_path):
    assert (NB_DIR / name).exists(), f"missing {name} (run scripts/make_notebooks.py)"
    # Ensure the 'python3' kernelspec (argv: bare 'python') resolves to THIS
    # interpreter (the one with the package installed) — as it does inside an
    # activated venv, which is how the acceptance command is run.
    bindir = str(Path(sys.executable).parent)
    env = dict(os.environ, FORECAUS_OFFLINE="1",
               PATH=bindir + os.pathsep + os.environ.get("PATH", ""))
    proc = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--output-dir", str(tmp_path), str(NB_DIR / name)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / name).exists()


def test_odeon_benchmark_csv_regenerated_and_matches_results():
    """After running nb 03, the ODEON table exists and its numbers appear in
    RESULTS.md (figures/table/prose agree)."""
    import re

    import pandas as pd

    csv = NB_DIR / "figures" / "odeon_benchmark.csv"
    md = NB_DIR / "figures" / "RESULTS.md"
    assert csv.exists() and md.exists()

    table = pd.read_csv(csv)
    assert {"dataset", "model", "role", "MAPE_pct", "coverage"} <= set(table.columns)
    roles = set(table["role"])
    assert {"baseline", "local-only", "federated-global", "centralised (pooled)"} <= roles
    assert table["dataset"].str.contains("LV Feeder").all()

    md_numbers = set(re.findall(r"\d+\.\d+", md.read_text()))
    for v in table["MAPE_pct"].round(2):
        assert f"{v:.2f}" in md_numbers, f"MAPE {v} not reflected in RESULTS.md"
