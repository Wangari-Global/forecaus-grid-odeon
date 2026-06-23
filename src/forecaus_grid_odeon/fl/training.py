"""Offline federation driver: N substation nodes, FedAvg rounds, honest metrics.

This is the **pre-submission prototype**: Flower clients (one per feeder) train
locally and the deterministic :func:`fedavg` reference aggregates each round. It
deliberately does NOT implement orchestration, scheduling or secure aggregation
— that is ODEON's FL Engine baseline (consumed via
:class:`..federated.OdeonFLClient`, the production path).

What it produces:
  * **global vs local vs centralised** test MAPE per feeder (federated consensus
    vs each node's own local-only fit vs an all-data-pooled reference),
  * the **cold-start benefit** for a thin-history feeder,
  * the **round-by-round convergence** trace of the global model.

Privacy invariant: only fixed-length parameter vectors cross the client
boundary. :func:`federate` checks every uplink payload has exactly ``n_params``
floats (independent of, and far smaller than, a node's sample count) and records
it, so "no raw data leaves a node" is verifiable.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .federated import fedavg
from .flower_client import (TransparentFLClient, fit_scaler, params_to_vector,
                            predict_standardized, ridge_standardized, vector_to_params)


def build_ss_nodes(
    feeders: Optional[Sequence[str]] = None,
    *,
    target: str = "load_kw",
    test_steps: int = 48,
    lags: Sequence[int] = (1, 48),
    rolling: Sequence[int] = (48,),
    thin_feeder: Optional[str] = None,
    thin_train: int = 24,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], list[str], str]:
    """Build per-feeder (train, test) frames with an identical feature set.

    A common (lags, rolling) spec is used for every node so the parameter
    vectors line up. One feeder is designated thin-history (its train window is
    truncated to ``thin_train`` rows) to demonstrate cold start.

    Returns ``(nodes, feature_order, thin_feeder)``.
    """
    from ..features import build_ss_frame
    from ..ingest_ss import ukpn
    from ..pipeline import ss_feeders

    feeders = list(feeders) if feeders is not None else ss_feeders()
    if thin_feeder is None:
        thin_feeder = feeders[0]

    nodes: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    feature_order: list[str] = []
    for fid in feeders:
        sub, fdr = ukpn._key_to_feeder(fid)
        series = ukpn.fetch_feeder(sub, fdr)[target]
        frame = build_ss_frame(series, lags=list(lags), rolling_windows=list(rolling))
        if not feature_order:
            feature_order = [c for c in frame.columns if c != target]
        train, test = frame.iloc[:-test_steps], frame.iloc[-test_steps:]
        if fid == thin_feeder:
            train = train.iloc[-thin_train:]          # cold-start: thin history
        nodes[fid] = (train, test)
    return nodes, feature_order, thin_feeder


def _weighted_mean(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    tot = sum(weights.values())
    return float(sum(values[k] * weights[k] for k in values) / tot)


def federate(
    nodes: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_order: Sequence[str],
    *,
    target: str = "load_kw",
    rounds: int = 12,
    lr: float = 0.5,
    l2: float = 1.0,
    thin_feeder: Optional[str] = None,
) -> dict:
    """Run ``rounds`` of FedAvg over the nodes; return metrics + convergence.

    Aggregation uses the deterministic :func:`fedavg` reference. No RNG, so the
    whole run is reproducible.
    """
    feature_order = list(feature_order)
    n_params = len(feature_order)                       # standardised betas (no intercept)
    clients = {
        fid: TransparentFLClient(tr, te, target, feature_order, l2=l2, lr=lr)
        for fid, (tr, te) in nodes.items()
    }
    test_w = {fid: c.n_test for fid, c in clients.items()}

    # ---- round-by-round FedAvg, starting from a zero global model ----
    global_vec = np.zeros(n_params, dtype=float)
    trace, uplink_floats = [], []
    for _ in range(rounds):
        updates, weights = [], []
        for c in clients.values():
            new_vec, n_samples, _ = c.fit([global_vec], {"lr": lr})
            # Privacy check: the uplink is a parameter vector, never samples.
            assert len(new_vec) == 1 and new_vec[0].shape == (n_params,), "non-parameter uplink"
            uplink_floats.append(int(new_vec[0].size))
            updates.append(vector_to_params(new_vec[0], feature_order))
            weights.append(n_samples)
        global_dict = fedavg(updates, weights)              # <-- deterministic reference
        global_vec = params_to_vector(global_dict, feature_order)
        per_round = {fid: c.evaluate([global_vec], {})[0] for fid, c in clients.items()}
        trace.append(_weighted_mean(per_round, test_w))

    global_params = vector_to_params(global_vec, feature_order)

    # ---- global vs local per feeder ----
    global_mape, local_mape = {}, {}
    for fid, c in clients.items():
        global_mape[fid] = c.evaluate([global_vec], {})[0]
        local_mape[fid] = c.evaluate([c.local_optimum], {})[0]

    # ---- centralised reference (data POOLED in one place — yardstick only) ----
    from ..eval.metrics import mape
    pooled = pd.concat([tr for tr, _ in nodes.values()])
    Xp = pooled[feature_order].to_numpy(float)
    yp = pooled[target].to_numpy(float)
    central_scaler = fit_scaler(Xp, yp)
    central_beta = ridge_standardized(Xp, yp, central_scaler, l2)
    central_mape = {}
    for fid, (_, te) in nodes.items():
        yhat = predict_standardized(te[feature_order].to_numpy(float), central_beta, central_scaler)
        central_mape[fid] = float(mape(te[target].to_numpy(), yhat))

    per_feeder = pd.DataFrame({
        "n_train": {fid: clients[fid].n_train for fid in clients},
        "local_MAPE": local_mape,
        "global_MAPE": global_mape,
        "centralised_MAPE": central_mape,
    })
    per_feeder.index.name = "feeder"

    cold_start = None
    if thin_feeder in local_mape:
        cold_start = {
            "feeder": thin_feeder,
            "n_train": clients[thin_feeder].n_train,
            "local_MAPE": local_mape[thin_feeder],
            "global_MAPE": global_mape[thin_feeder],
            "benefit": local_mape[thin_feeder] - global_mape[thin_feeder],  # +ve => helps
        }

    return {
        "per_feeder": per_feeder,
        "aggregate": {
            "local_MAPE": _weighted_mean(local_mape, test_w),
            "global_MAPE": _weighted_mean(global_mape, test_w),
            "centralised_MAPE": _weighted_mean(central_mape, test_w),
        },
        "convergence": trace,
        "cold_start": cold_start,
        "global_params": global_params,
        "n_nodes": len(clients),
        "n_params": n_params,
        "uplink_param_floats": sorted(set(uplink_floats)),
        "min_node_samples": min(c.n_train for c in clients.values()),
    }


def run_fl_training(rounds: int = 12, lr: float = 0.5, l2: float = 1.0,
                    thin_train: int = 24) -> dict:
    """End-to-end offline federation on the committed SS fixtures."""
    nodes, feature_order, thin = build_ss_nodes(thin_train=thin_train)
    return federate(nodes, feature_order, rounds=rounds, lr=lr, l2=l2, thin_feeder=thin)
