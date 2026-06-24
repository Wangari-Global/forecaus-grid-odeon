"""scripts/reproduce_headline.py: the guard passes on the real (computed)
narrative, and flags an injected hallucinated number."""
import importlib.util
from pathlib import Path

import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.validation import validate_claims

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce_headline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("reproduce_headline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    _r, _o = config.RAW, config.OFFLINE
    config.RAW, config.OFFLINE = raw, True
    try:
        return _load_script().compute_headline(rounds=6)
    finally:
        config.RAW, config.OFFLINE = _r, _o


def test_table_reproduces_headline_numbers(head):
    c = head["computed"]
    # The computed values are sane forecast/FL metrics, and the printed table
    # cites them.
    assert 0 < c["mape_best"] <= c["mape_structured"] + 1e-9
    assert c["fl_central"] <= c["fl_global"] <= c["fl_local"] + 1e-9     # federation ordering
    assert c["cs_benefit"] > 0 and c["n_feeders"] > 1 and c["n_nodes"] > 1
    assert f"{c['fl_global']:.2f}" in head["table"]
    assert f"{c['mape_structured']:.2f}" in head["table"]


def test_guard_passes_on_real_narrative(head):
    res = validate_claims(head["narrative"], head["computed"])
    assert res["ok"] is True, res["unsupported"]
    assert res["unsupported"] == []
    assert res["n_numbers"] >= 10          # the narrative cites the headline figures


def test_guard_flags_injected_hallucination(head):
    """Acceptance: a fabricated number with no computed support is flagged."""
    tampered = head["narrative"] + " An audit also reported EUR 9,999,999 of savings."
    res = validate_claims(tampered, head["computed"])
    assert res["ok"] is False
    flagged = [u["value"] for u in res["unsupported"]]
    assert 9999999.0 in flagged
    # The genuine figures are still supported (only the fabricated one is flagged).
    assert len(flagged) == 1


def test_results_md_written_and_traceable(head, tmp_path):
    mod = _load_script()
    path = mod.write_results_md(head["narrative"], head["disclaimer"], tmp_path / "RESULTS.md")
    assert path.exists()
    body = path.read_text()
    assert f"{head['computed']['fl_global']:.2f}" in body
    # The prose body (title carries no result figures) is fully traceable.
    assert validate_claims(body, head["computed"])["ok"] is True
