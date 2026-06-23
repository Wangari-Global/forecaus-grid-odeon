"""Federated learning layer for Challenge 4.

ODEON mandates FL "natively end-to-end" and provides the Federated Learning
Engine; the non-overlap rule means we CONSUME it rather than build one. This
package holds:

* :func:`fedavg` — a transparent, deterministic FedAvg reference over the
  interpretable forecaster's named coefficients (the aggregation reference);
* :class:`OdeonFLClient` — the stub adapter to ODEON's FL Engine (production
  path; API pending, help-desk Q1);
* the offline prototype (:func:`federate` / :func:`run_fl_training` +
  :class:`TransparentFLClient`) that federates the Slice-3 transparent
  forecaster across substations using Flower on the client side. It does NOT
  build orchestration/secure-agg — that is ODEON baseline.

The Flower-dependent pieces are imported lazily so ``fedavg``/``OdeonFLClient``
stay importable without the ``[fl]`` extra.
See ODEON_Challenge4_FL_Architecture.docx.
"""
from .federated import OdeonFLClient, fedavg

__all__ = [
    "fedavg", "OdeonFLClient",
    "federate", "run_fl_training", "build_ss_nodes", "TransparentFLClient",
]

_LAZY = {
    "federate": ("training", "federate"),
    "run_fl_training": ("training", "run_fl_training"),
    "build_ss_nodes": ("training", "build_ss_nodes"),
    "TransparentFLClient": ("flower_client", "TransparentFLClient"),
}


def __getattr__(name):  # PEP 562: import Flower-dependent symbols on first use
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(f"{__name__}.{module}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
