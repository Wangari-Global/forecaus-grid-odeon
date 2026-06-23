"""Interpretable, edge-deployable structured forecaster (GAM-style additive linear).

This is the model meant to run *on the substation edge* and to *federate*:

* **Structured / GAM-style.** The prediction is an additive sum of named feature
  contributions ``yhat = b0 + sum_j w_j * x_j``. With the engineered SS features
  (cyclical hour/dow/month pairs = smooth seasonal terms, load lags = the
  autoregressive part, weather + topology as additive effects) this is a
  generalised additive model with a transparent, inspectable shape.
* **Interpretable / auditable.** :meth:`StructuredForecaster.parameters` returns a
  flat ``{feature_name: coefficient}`` dict (plus ``intercept``) in *original
  feature units*, so each driver's effect can be read off and reviewed.
* **Federatable.** Those named coefficients are exactly the parameter dicts
  :func:`forecaus_grid_odeon.fl.fedavg` averages - every feeder/node exposes the
  same keys, so a FedAvg round is well-defined (this is the Slice-5 hook).
* **Edge-deployable.** Pure-numpy ridge fit and a dot-product predict; no heavy
  runtime deps, a few dozen floats of state.

Ridge regularisation is applied on standardised features (scale-fair penalty),
then folded back to original-unit coefficients so the reported parameters are
comparable across feeders and directly usable for federation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .intervals import make_conformal

INTERCEPT = "intercept"


class StructuredForecaster:
    """Additive ridge regressor over named features (fit / predict / parameters).

    Parameters
    ----------
    features : sequence of str, optional
        Feature columns to use. Defaults to every column except the target at
        fit time.
    l2 : float
        Ridge penalty on the standardised coefficients (>= 0).
    """

    def __init__(self, features: Optional[Sequence[str]] = None, *, l2: float = 1.0):
        self.features = list(features) if features is not None else None
        self.l2 = float(l2)
        self.coef_: dict[str, float] = {}
        self.intercept_: float = 0.0

    # ------------------------------------------------------------------ fit --
    def fit(self, train: pd.DataFrame, target: str) -> "StructuredForecaster":
        feats = self.features if self.features is not None else [
            c for c in train.columns if c != target
        ]
        self.features = list(feats)
        X = train[self.features].to_numpy(dtype=float)
        y = train[target].to_numpy(dtype=float)

        mu_x = X.mean(axis=0)
        sd_x = X.std(axis=0)
        sd_x[sd_x < 1e-12] = 1.0                      # constant column -> no scaling
        Xs = (X - mu_x) / sd_x
        mu_y = y.mean()
        yc = y - mu_y

        p = Xs.shape[1]
        # Ridge on standardised features; intercept handled by centring.
        gram = Xs.T @ Xs + self.l2 * np.eye(p)
        w_std = np.linalg.solve(gram, Xs.T @ yc)

        w = w_std / sd_x                               # back to original units
        self.coef_ = {f: float(c) for f, c in zip(self.features, w)}
        self.intercept_ = float(mu_y - w @ mu_x)
        return self

    # -------------------------------------------------------------- predict --
    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if not self.coef_ and self.intercept_ == 0.0:
            raise RuntimeError("StructuredForecaster.predict called before fit")
        X = test[self.features].to_numpy(dtype=float)
        w = np.array([self.coef_[f] for f in self.features], dtype=float)
        return X @ w + self.intercept_

    # ----------------------------------------------------- federate / audit --
    def parameters(self) -> dict[str, float]:
        """Flat named-coefficient dict (incl. ``intercept``) for audit/FedAvg."""
        return {INTERCEPT: self.intercept_, **self.coef_}

    def set_parameters(self, params: dict[str, float]) -> "StructuredForecaster":
        """Load a (e.g. federated-averaged) parameter dict back into the model."""
        params = dict(params)
        self.intercept_ = float(params.pop(INTERCEPT, self.intercept_))
        self.coef_ = {k: float(v) for k, v in params.items()}
        self.features = list(self.coef_)
        return self


def make_structured(
    features: Optional[Sequence[str]] = None,
    *,
    l2: float = 1.0,
    conformal: bool = True,
    alpha: float = 0.1,
    calib_frac: float = 0.3,
):
    """Adapt :class:`StructuredForecaster` to the backtest model interface.

    Returns ``model(train, test, target) -> DataFrame[yhat, lower, upper]``,
    conformal-wrapped by default so it emits calibrated intervals.
    """
    def point(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
        fitted = StructuredForecaster(features, l2=l2).fit(train, target)
        return pd.DataFrame({"yhat": fitted.predict(test)}, index=test.index)

    return make_conformal(point, alpha=alpha, calib_frac=calib_frac) if conformal else point
