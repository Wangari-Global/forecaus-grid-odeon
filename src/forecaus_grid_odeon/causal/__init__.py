"""Causal layer: explicit DAG -> DoWhy effect estimation -> refutation tests."""
from __future__ import annotations

import pandas as pd

from .estimate import estimate_effect
from .graph import (OUTCOME, SS_DOWNSTREAM, SS_OUTCOME, SS_TREATMENT, TREATMENT,
                    build_dag, build_ss_dag, draw_dag, to_dot)
from .refute import refute

__all__ = [
    "build_dag", "build_ss_dag", "draw_dag", "to_dot", "estimate_effect", "refute",
    "run_causal", "TREATMENT", "OUTCOME", "SS_TREATMENT", "SS_OUTCOME", "SS_DOWNSTREAM",
]


def run_causal(df: pd.DataFrame, treatment: str = TREATMENT, outcome: str = OUTCOME,
               random_seed: int = 0, num_simulations: int = 100,
               graph=None) -> tuple[dict, pd.DataFrame]:
    """Estimate the effect and run all refuters. Returns (result, refute_table).

    Pass ``graph`` to use an explicit DAG (e.g. :func:`build_ss_dag` for the
    SS-level effect); otherwise the default national DAG is built.
    """
    result = estimate_effect(df, treatment, outcome, graph=graph)
    table = refute(result["model"], result["estimand"], result["estimate"],
                   random_seed=random_seed, num_simulations=num_simulations)
    return result, table
