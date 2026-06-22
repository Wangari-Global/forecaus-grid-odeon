"""Causal-layer tests: DAG structure, effect recovery, refutation pass/fail."""
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon.causal import (build_dag, estimate_effect, refute,
                                  run_causal)

warnings.filterwarnings("ignore")  # statsmodels condition-number chatter


# ------------------------------------------------------------------- DAG ----
def test_dag_is_acyclic_and_has_effect_edge():
    g = build_dag()
    assert nx.is_directed_acyclic_graph(g)
    assert g.has_edge("temp_c", "load_mw")          # treatment -> outcome
    assert g.has_edge("hour_sin", "temp_c")          # confounder fork
    assert g.has_edge("hour_sin", "load_mw")
    # demand is a sink (no outgoing edges) — it is caused, it causes nothing here.
    assert g.out_degree("load_mw") == 0


# --------------------------------------- controlled recovery of an effect ----
@pytest.fixture
def confounded_data():
    """u confounds temp and load; the true temp->load effect is 500."""
    rng = np.random.default_rng(7)
    n = 800
    u = rng.normal(0, 1, n)                       # e.g. time-of-day driver
    temp = 2.0 * u + rng.normal(0, 1, n)
    load = 500.0 * temp + 300.0 * u + rng.normal(0, 30, n)
    df = pd.DataFrame({"u": u, "temp_c": temp, "load_mw": load})
    g = nx.DiGraph([("u", "temp_c"), ("u", "load_mw"), ("temp_c", "load_mw")])
    return df, g


def test_backdoor_recovers_known_effect(confounded_data):
    df, g = confounded_data
    res = estimate_effect(df, "temp_c", "load_mw", graph=g)
    # Adjusting for the confounder u recovers the true coefficient (~500).
    assert res["effect"] == pytest.approx(500, abs=40)
    lo, hi = res["ci"]
    assert lo > 0 and hi < 1000                      # significant, sane CI
    assert "u" in res["backdoor_set"]


def test_naive_correlation_is_biased(confounded_data):
    """Without adjustment, the temp~load slope is biased away from 500."""
    df, _ = confounded_data
    biased = np.polyfit(df["temp_c"], df["load_mw"], 1)[0]
    assert abs(biased - 500) > 40                    # confounding shifts it


def test_refuters_pass_on_valid_effect(confounded_data):
    df, g = confounded_data
    res = estimate_effect(df, "temp_c", "load_mw", graph=g)
    table = refute(res["model"], res["estimand"], res["estimate"], random_seed=0, num_simulations=20)
    assert list(table.index) == ["placebo_treatment", "random_common_cause", "data_subset"]
    assert {"original_effect", "new_effect", "p_value", "passed"} <= set(table.columns)
    # Placebo effect collapses toward zero; the other two stay close to original.
    assert abs(table.loc["placebo_treatment", "new_effect"]) < 0.25 * abs(res["effect"])
    assert table["passed"].all()


# ------------------------------------------------- end-to-end on fixtures ----
def test_run_causal_on_offline_sample(monkeypatch):
    from forecaus_grid_odeon import config
    monkeypatch.setattr(config, "OFFLINE", True)
    from forecaus_grid_odeon.pipeline import load_frame

    frame = load_frame()
    result, table = run_causal(frame, num_simulations=20)   # temp_c -> load_mw, seed 0
    assert result["effect"] > 100                  # the injected effect is detected
    assert result["ci"][0] > 0                     # significant (CI excludes 0)
    assert set(result["backdoor_set"]) == {"hour_sin", "hour_cos"}
    assert len(table) == 3
