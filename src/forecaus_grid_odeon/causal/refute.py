"""Refutation / sensitivity tests — the scientific-rigour evidence (fixes OC1).

Three standard DoWhy refuters, each with an explicit pass criterion:

* **placebo_treatment** — replace the treatment with random noise; a valid effect
  should collapse toward zero. PASS when |placebo effect| is small vs the
  original (the effect *vanishes* when the cause is fake).
* **random_common_cause** — add an independent random confounder; a robust effect
  should barely move. PASS when the relative change is small.
* **data_subset** — re-estimate on a random 80% subset; a stable effect should
  barely move. PASS when the relative change is small.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.getLogger("dowhy").setLevel(logging.WARNING)

# Pass thresholds (documented, deliberately generous to avoid flakiness).
PLACEBO_MAX_RATIO = 0.25     # placebo effect must be <25% of original
STABILITY_MAX_REL = 0.30     # added-confounder / subset effect within 30%


def _p_value(refutation) -> float:
    res = getattr(refutation, "refutation_result", None)
    if isinstance(res, dict) and res.get("p_value") is not None:
        return float(np.ravel(res["p_value"])[0])
    return float("nan")


def refute(model, estimand, estimate, random_seed: int = 0,
           num_simulations: int = 100) -> pd.DataFrame:
    """Run the three refuters; return a per-refuter pass/fail table.

    Columns: original_effect, new_effect, p_value, criterion, passed.
    ``num_simulations`` trades runtime for the stability of the refuters'
    significance tests (lower is faster; the default is the DoWhy default).
    """
    orig = float(estimate.value)
    denom = abs(orig) if abs(orig) > 1e-9 else np.nan
    sims = dict(num_simulations=num_simulations)

    specs = [
        ("placebo_treatment", dict(method_name="placebo_treatment_refuter",
                                   placebo_type="permute", random_seed=random_seed, **sims)),
        ("random_common_cause", dict(method_name="random_common_cause",
                                     random_seed=random_seed, **sims)),
        ("data_subset", dict(method_name="data_subset_refuter",
                             subset_fraction=0.8, random_seed=random_seed, **sims)),
    ]

    rows = []
    for name, kwargs in specs:
        r = model.refute_estimate(estimand, estimate, **kwargs)
        new = float(r.new_effect)
        if name == "placebo_treatment":
            ratio = abs(new) / denom if denom == denom else np.nan
            passed = bool(ratio < PLACEBO_MAX_RATIO)
            criterion = f"|new|/|orig| < {PLACEBO_MAX_RATIO}"
        else:
            rel = abs(new - orig) / denom if denom == denom else np.nan
            passed = bool(rel < STABILITY_MAX_REL)
            criterion = f"|new-orig|/|orig| < {STABILITY_MAX_REL}"
        rows.append({
            "original_effect": orig, "new_effect": new,
            "p_value": _p_value(r), "criterion": criterion, "passed": passed,
        })

    table = pd.DataFrame(rows, index=[s[0] for s in specs])
    table.index.name = "refuter"
    return table
