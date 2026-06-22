"""Causal effect estimation with DoWhy (identify -> estimate -> CI).

Given the structural DAG, DoWhy derives an identification strategy (a backdoor
adjustment set that blocks confounding) and estimates the effect by linear
regression on treatment + adjustment set. Returns the point effect with a
confidence interval, plus the model/estimand/estimate objects needed for the
refutation tests in :mod:`.refute`.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .graph import OUTCOME, TREATMENT, build_dag, required_columns

# DoWhy is chatty; keep CLI output focused on the result.
for _name in ("dowhy", "dowhy.causal_model", "dowhy.causal_estimator"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def estimate_effect(
    df: pd.DataFrame,
    treatment: str = TREATMENT,
    outcome: str = OUTCOME,
    graph=None,
    method_name: str = "backdoor.linear_regression",
) -> dict:
    """Identify + estimate the causal effect of ``treatment`` on ``outcome``.

    Returns a dict with the point ``effect``, ``ci`` (low, high), the backdoor
    adjustment set, and the DoWhy ``model`` / ``estimand`` / ``estimate`` objects.
    """
    from dowhy import CausalModel

    graph = graph if graph is not None else build_dag(treatment, outcome)
    missing = [c for c in required_columns(graph) if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing graph columns: {missing}")

    data = df[required_columns(graph)].astype(float).reset_index(drop=True)
    # DoWhy 0.14 accepts a networkx DiGraph directly.
    model = CausalModel(data=data, treatment=treatment, outcome=outcome, graph=graph)

    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name=method_name,
        test_significance=True, confidence_intervals=True,
    )

    try:
        ci = np.ravel(estimate.get_confidence_intervals())
        ci = (float(ci[0]), float(ci[-1]))
    except Exception:
        ci = (float("nan"), float("nan"))

    backdoor = list(getattr(estimand, "get_backdoor_variables", lambda: [])() or [])
    return {
        "treatment": treatment,
        "outcome": outcome,
        "effect": float(estimate.value),
        "ci": ci,
        "backdoor_set": backdoor,
        "method": method_name,
        "model": model,
        "estimand": estimand,
        "estimate": estimate,
    }
