"""Federated-averaging reference + ODEON FL Engine client adapter (stub).

The reference fedavg() works on plain parameter dicts (e.g. the coefficients of
the transparent forecaster), so federation stays auditable and reproducible -
the "auditable federated forecasting" pitch. The production path runs through
ODEON's FL Engine; that adapter is stubbed until the API is released.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def fedavg(
    params: Sequence[Mapping[str, float]],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Weighted average of per-node parameter dicts (one FedAvg round).

    Parameters
    ----------
    params : list of {param_name: value}
        Locally trained parameters from each substation node. Keys must match.
    weights : list of float, optional
        Per-node weights (e.g. #samples). Defaults to equal weighting.

    Deterministic: same inputs -> same output, no RNG. Raises if keys differ.
    """
    if not params:
        raise ValueError("no parameter sets to aggregate")
    keys = set(params[0])
    for p in params[1:]:
        if set(p) != keys:
            raise ValueError("parameter keys differ across nodes")
    n = len(params)
    w = [1.0 / n] * n if weights is None else [x / sum(weights) for x in weights]
    return {k: sum(w[i] * params[i][k] for i in range(n)) for k in keys}


class OdeonFLClient:
    """Client adapter to the ODEON Federated Learning Engine (STUB).

    Intended flow: register model -> receive global params -> local fit ->
    push update -> receive aggregated global. Implement once the FL Engine API
    is published (help-desk Q1). Do NOT reimplement federation/orchestration -
    that is ODEON baseline (non-overlap rule).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ODEON FL Engine API not released pre-submission (help-desk Q1). "
            "Prototype federation with fedavg() + an open lib (e.g. Flower)."
        )
