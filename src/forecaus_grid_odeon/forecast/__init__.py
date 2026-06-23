"""Forecasters + a uniform model interface for the backtest harness.

A *model* is a callable ``model(train_df, test_df, target) -> DataFrame`` whose
result is indexed like ``test_df`` with columns ``yhat``, ``lower``, ``upper``.
Factories below adapt the raw forecasters in :mod:`.baselines` to that shape.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import pandas as pd

from .baselines import sarimax_forecast, seasonal_naive
from .intervals import conformal_intervals, conformal_width, make_conformal
from .structured import StructuredForecaster, make_structured

Model = Callable[[pd.DataFrame, pd.DataFrame, str], pd.DataFrame]

__all__ = [
    "seasonal_naive", "sarimax_forecast", "conformal_intervals", "conformal_width",
    "make_seasonal_naive", "make_sarimax", "make_conformal", "make_tft", "make_linear",
    "make_structured", "StructuredForecaster",
    "default_models", "benchmark_models", "Model",
]


def make_linear(features: Sequence[str], *, conformal: bool = True,
                alpha: float = 0.1, calib_frac: float = 0.3) -> Model:
    """Ordinary-least-squares forecaster over a fixed feature set.

    Used for the causal-augmented vs purely-correlational comparison: pass the
    causal parent set for the causal model, or the full column set for the
    correlational model. Conformal-wrapped by default so it emits intervals.
    """
    feats = list(features)

    def point(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(train[feats].to_numpy(float), train[target].to_numpy(float))
        yhat = lr.predict(test[feats].to_numpy(float))
        return pd.DataFrame({"yhat": yhat}, index=test.index)

    return make_conformal(point, alpha=alpha, calib_frac=calib_frac) if conformal else point


def make_seasonal_naive(season: int = 24, alpha: float = 0.1) -> Model:
    """Seasonal-naive point forecast + conformal intervals from in-sample residuals."""
    def model(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
        y = train[target]
        point = seasonal_naive(y, len(test), season).to_numpy()
        resid = (y - y.shift(season)).dropna().to_numpy()
        lower, upper = conformal_intervals(point, resid, alpha)
        return pd.DataFrame({"yhat": point, "lower": lower, "upper": upper}, index=test.index)
    return model


def make_sarimax(
    exog_cols: Optional[Sequence[str]] = None,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    alpha: float = 0.1,
) -> Model:
    """SARIMAX with exogenous regressors (calendar/weather/price/lags) + intervals."""
    def model(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
        cols = list(exog_cols) if exog_cols is not None else [c for c in train.columns if c != target]
        ex_tr = train[cols] if cols else None
        ex_fut = test[cols] if cols else None
        fc = sarimax_forecast(
            train[target], ex_tr, len(test), future_exog=ex_fut,
            order=order, seasonal_order=seasonal_order, alpha=alpha,
        )
        fc.index = test.index
        return fc
    return model


def make_tft(cov_cols: Optional[Sequence[str]] = None, *, alpha: float = 0.1,
             n_epochs: int = 30, input_chunk_length: int = 72,
             hidden_size: int = 16, random_state: int = 42) -> Model:
    """DARTS Temporal Fusion Transformer adapter (future covariates + quantiles)."""
    def model(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
        from .neural import tft_forecast
        cols = list(cov_cols) if cov_cols is not None else [c for c in train.columns if c != target]
        full = pd.concat([train, test])
        fc = tft_forecast(
            train[target], full[cols] if cols else full[[]], len(test),
            alpha=alpha, n_epochs=n_epochs, input_chunk_length=input_chunk_length,
            hidden_size=hidden_size, random_state=random_state,
        )
        fc.index = test.index
        return fc
    return model


def default_models(exog_cols: Optional[Sequence[str]] = None,
                   season: int = 24, alpha: float = 0.1) -> dict[str, Model]:
    """The baseline panel: seasonal-naive + SARIMAX."""
    return {
        "seasonal_naive": make_seasonal_naive(season=season, alpha=alpha),
        "sarimax": make_sarimax(exog_cols=exog_cols, alpha=alpha),
    }


def benchmark_models(exog_cols: Optional[Sequence[str]] = None, season: int = 24,
                     alpha: float = 0.1, *, neural: bool = True,
                     n_epochs: int = 30) -> dict[str, Model]:
    """Full comparison panel: baselines + conformal-wrapped naive + (optional) TFT."""
    models = default_models(exog_cols=exog_cols, season=season, alpha=alpha)
    models["seasonal_naive_conformal"] = make_conformal(
        make_seasonal_naive(season=season), alpha=alpha
    )
    if neural:
        models["tft"] = make_tft(cov_cols=exog_cols, alpha=alpha, n_epochs=n_epochs)
    return models
