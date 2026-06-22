"""Headline result: causal-augmented degrades less than correlational ML on a
structural break. Seeded, so the claim is reproducible."""
import warnings

import pytest

from forecaus_grid_odeon.causal.graph import causal_parents
from forecaus_grid_odeon.eval.causal_break import (SPURIOUS_FEATURE, break_dag,
                                             make_break_figure,
                                             run_break_experiment)

warnings.filterwarnings("ignore")


def test_causal_feature_selection_excludes_effect():
    """The causal feature set = parents of demand; it must drop the downstream
    spurious feature (an effect of demand, not a cause)."""
    parents = causal_parents(break_dag(), "load_mw")
    assert "temp_c" in parents and "hour_sin" in parents
    assert SPURIOUS_FEATURE not in parents       # gas_proxy is a child, excluded


@pytest.fixture(scope="module")
def result():
    return run_break_experiment(seed=6)


def test_headline_claim_causal_degrades_less(result):
    t = result["table"]
    caus, corr = t.loc["causal_augmented"], t.loc["correlational_ml"]

    # 1. The break genuinely hurts the correlational model...
    assert corr["MAPE_degradation"] > 3.0
    # 2. ...but barely touches the causal-augmented one.
    assert caus["MAPE_degradation"] < 1.5
    # 3. Headline: causal degrades LESS than correlational on the break window.
    assert caus["MAPE_degradation"] < corr["MAPE_degradation"]
    assert caus["MAPE_break"] < corr["MAPE_break"]


def test_correlational_wins_normally(result):
    """Honesty check: the spurious feature really does help in-distribution, so
    correlational is best in the normal regime — the trade-off is real."""
    t = result["table"]
    assert t.loc["correlational_ml", "MAPE_normal"] < t.loc["causal_augmented", "MAPE_normal"]


def test_causal_keeps_better_break_coverage(result):
    t = result["table"]
    assert t.loc["causal_augmented", "coverage_break"] > t.loc["correlational_ml", "coverage_break"]


def test_table_and_folds_structure(result):
    t = result["table"]
    assert set(t.index) == {"naive", "correlational_ml", "causal_augmented"}
    for col in ["MAPE_overall", "MAPE_normal", "MAPE_break", "MAPE_degradation"]:
        assert col in t.columns
    assert (result["fold_df"]["regime"] == "break").any()


def test_figure_is_written(result, tmp_path):
    path = make_break_figure(result, result["break_start"], str(tmp_path / "break.png"))
    assert (tmp_path / "break.png").exists()
