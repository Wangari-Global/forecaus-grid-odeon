"""SS-level causal layer: DAG structure, DoWhy effect + refuters, and the
causal-vs-correlational structural-break experiment (reported honestly)."""
import warnings

import networkx as nx
import pytest

from forecaus_grid_odeon.causal import build_ss_dag
from forecaus_grid_odeon.causal.graph import (SS_DOWNSTREAM, SS_OUTCOME, SS_TREATMENT,
                                              causal_parents)
from forecaus_grid_odeon.eval import make_break_figure
from forecaus_grid_odeon.eval.ss_causal import (TRUE_TEMP_COEF, run_ss_break_experiment,
                                                run_ss_causal_effect)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ SS DAG ----
def test_ss_dag_structure_and_feature_selection():
    g = build_ss_dag()
    assert nx.is_directed_acyclic_graph(g)
    # temperature -> load (the effect), calendar -> {load, temperature} (confounder).
    assert g.has_edge(SS_TREATMENT, SS_OUTCOME)
    assert g.has_edge("hour_sin", SS_TREATMENT) and g.has_edge("hour_sin", SS_OUTCOME)
    # load -> pv_ev_proxy: downstream effect, not a cause.
    assert g.has_edge(SS_OUTCOME, SS_DOWNSTREAM)

    parents = causal_parents(g, SS_OUTCOME)
    assert SS_TREATMENT in parents and "hour_sin" in parents
    assert SS_DOWNSTREAM not in parents          # the downstream proxy is excluded


# ----------------------------------------------- DoWhy estimate + refuters ----
@pytest.fixture(scope="module")
def effect():
    return run_ss_causal_effect(seed=6, num_simulations=20)


def test_effect_recovers_known_temperature_coefficient(effect):
    result, _ = effect
    # The injected heating effect is recovered (negative: colder -> more load).
    assert result["effect"] == pytest.approx(TRUE_TEMP_COEF, abs=0.4)
    assert result["effect"] < 0                  # honest: a NEGATIVE coefficient
    lo, hi = result["ci"]
    assert hi < 0                                # CI excludes zero -> significant
    # Backdoor adjustment is the diurnal calendar confounder only.
    assert set(result["backdoor_set"]) == {"hour_sin", "hour_cos"}


def test_refuter_table_passes(effect):
    _, table = effect
    assert list(table.index) == ["placebo_treatment", "random_common_cause", "data_subset"]
    assert {"original_effect", "new_effect", "p_value", "passed"} <= set(table.columns)
    # Placebo collapses toward zero; the other two stay close to the original.
    assert abs(table.loc["placebo_treatment", "new_effect"]) < 0.25 * abs(table.loc["placebo_treatment", "original_effect"])
    assert table["passed"].all()


# ------------------------------------------- causal vs correlational break ----
@pytest.fixture(scope="module")
def break_result():
    return run_ss_break_experiment(seed=6)


def test_break_table_structure_and_regimes(break_result):
    t = break_result["table"]
    assert set(t.index) == {"naive", "correlational_ml", "causal_augmented"}
    for col in ["MAPE_overall", "MAPE_normal", "MAPE_break", "MAPE_degradation"]:
        assert col in t.columns
    assert (break_result["fold_df"]["regime"] == "break").any()
    assert (break_result["fold_df"]["regime"] == "normal").any()


def test_causal_degrades_less_on_break(break_result):
    t = break_result["table"]
    caus, corr = t.loc["causal_augmented"], t.loc["correlational_ml"]
    # The break genuinely hurts the correlational model (it leaned on pv_ev_proxy)...
    assert corr["MAPE_degradation"] > 3.0
    # ...but barely touches the causal-augmented one.
    assert caus["MAPE_degradation"] < 1.5
    # Headline: causal degrades LESS and is more accurate on the break window.
    assert caus["MAPE_degradation"] < corr["MAPE_degradation"]
    assert caus["MAPE_break"] < corr["MAPE_break"]


def test_correlational_wins_in_distribution(break_result):
    """Honesty: the spurious proxy really does help in the normal regime, so
    correlational is best there — the trade-off is real, not cherry-picked."""
    t = break_result["table"]
    assert t.loc["correlational_ml", "MAPE_normal"] < t.loc["causal_augmented", "MAPE_normal"]


def test_break_figure_written(break_result, tmp_path):
    path = make_break_figure(break_result, break_result["break_start"],
                             str(tmp_path / "ss_break.png"), ylabel="load [kW]")
    assert (tmp_path / "ss_break.png").exists()
