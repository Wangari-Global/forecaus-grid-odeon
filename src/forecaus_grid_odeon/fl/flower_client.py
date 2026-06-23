"""Flower client wrapping the Slice-3 transparent forecaster, one per substation.

Each LV feeder is a federated node. The client holds that feeder's data
**privately** and only ever emits/accepts the model's named coefficients (a tiny
fixed-length vector) over the Flower ``NumPyClient`` interface — raw smart-meter
samples never leave the node. The server side is ODEON's FL Engine in production
(see :class:`..federated.OdeonFLClient`); for the offline prototype the rounds
are aggregated by the deterministic :func:`..federated.fedavg` reference.

What federates. The transparent forecaster is a linear additive (ridge) model.
We federate it in **fully standardised space**: each node z-scores its own
features and target with a *private, local* scaler, so the shared coefficients
``beta`` are dimensionless standardised partial effects — one per named feature.
These are scale-free, so they are comparable and safe to average across feeders
of different sizes (averaging the raw original-unit coefficients is not: the
large offsetting intercepts live in different bases and the mean is nonsense).
Federation thus shares the *relationship shape*; each node re-applies it at its
own scale via its private scaler (and its private bias = local target mean).

Local update (transparent + deterministic): given the global ``beta``, one local
"epoch" returns a damped step toward the node's closed-form local optimum
``beta + lr*(beta* - beta)`` — a damped Newton/gradient step on the local
standardised MSE. FedAvg-averaging these converges geometrically to the weighted
mean of the local optima (the consensus), giving a clean trace with no RNG.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    from flwr.client import NumPyClient
except Exception as exc:  # pragma: no cover - flwr is the [fl] extra
    raise ImportError(
        "federated training needs Flower: install with `pip install -e '.[fl]'`"
    ) from exc


# --------------------------------------------------- standardised ridge helpers --
def fit_scaler(X: np.ndarray, y: np.ndarray) -> dict:
    """Per-feature and target mean/std. Stays PRIVATE on the node."""
    sd_x = X.std(axis=0)
    sd_x[sd_x < 1e-12] = 1.0
    sd_y = float(y.std()) or 1.0
    return {"mu_x": X.mean(axis=0), "sd_x": sd_x, "mu_y": float(y.mean()), "sd_y": sd_y}


def ridge_standardized(X: np.ndarray, y: np.ndarray, scaler: dict, l2: float) -> np.ndarray:
    """Closed-form ridge on standardised (X, y); returns dimensionless betas."""
    Z = (X - scaler["mu_x"]) / scaler["sd_x"]
    yc = (y - scaler["mu_y"]) / scaler["sd_y"]
    p = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + l2 * np.eye(p), Z.T @ yc)


def predict_standardized(X: np.ndarray, beta: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply standardised betas at the node's own scale: y = mu_y + sd_y * (Z @ beta)."""
    Z = (X - scaler["mu_x"]) / scaler["sd_x"]
    return scaler["mu_y"] + scaler["sd_y"] * (Z @ np.asarray(beta, dtype=float))


def params_to_vector(params: dict, feature_order: Sequence[str]) -> np.ndarray:
    """Named-coefficient dict -> ordered beta vector (one entry per feature)."""
    return np.array([params[f] for f in feature_order], dtype=float)


def vector_to_params(vec: np.ndarray, feature_order: Sequence[str]) -> dict:
    """Inverse of :func:`params_to_vector` (keeps the coefficients named)."""
    vec = np.asarray(vec, dtype=float)
    return {f: float(v) for f, v in zip(feature_order, vec)}


class TransparentFLClient(NumPyClient):
    """One substation node: trains the transparent forecaster locally.

    Parameters
    ----------
    train, test : pandas.DataFrame
        The node's PRIVATE local data (never transmitted).
    target : str
        Target column (``load_kw``).
    feature_order : sequence of str
        Shared, fixed feature ordering (identical across nodes so the parameter
        vectors line up for FedAvg).
    l2, lr : float
        Local ridge penalty and the damped-step learning rate.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str,
                 feature_order: Sequence[str], *, l2: float = 1.0, lr: float = 0.5):
        self.target = target
        self.feature_order = list(feature_order)
        self.lr = float(lr)
        self.n_train = int(len(train))
        self.n_test = int(len(test))

        Xtr = train[self.feature_order].to_numpy(dtype=float)
        ytr = train[target].to_numpy(dtype=float)
        # PRIVATE: scaler and held-out test never leave the node.
        self._scaler = fit_scaler(Xtr, ytr)
        self._Xte = test[self.feature_order].to_numpy(dtype=float)
        self._yte = test[target].to_numpy(dtype=float)
        # Local optimum beta* (closed-form standardised ridge), computed once.
        self.local_optimum = ridge_standardized(Xtr, ytr, self._scaler, l2)

    # ----- Flower NumPyClient interface (only parameters cross this boundary) --
    def get_parameters(self, config=None) -> list:
        return [self.local_optimum.copy()]

    def fit(self, parameters, config=None):
        """One local round: damped step from global beta toward local beta*.

        Returns ``([updated_beta], num_local_samples, metrics)`` — a parameter
        vector only; no data, no scaler.
        """
        lr = float((config or {}).get("lr", self.lr))
        g = np.asarray(parameters[0], dtype=float)
        step = g + lr * (self.local_optimum - g)
        return [step], self.n_train, {}

    def evaluate(self, parameters, config=None):
        """Apply the given global betas at the node's local scale; score on the
        node's private test set. Returns ``(mape, n_test, {"mape": ...})``."""
        from ..eval.metrics import mape
        yhat = predict_standardized(self._Xte, parameters[0], self._scaler)
        m = float(mape(self._yte, yhat))
        return m, self.n_test, {"mape": m}
