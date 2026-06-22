"""Baseline forecasters: seasonal-naive and SARIMAX (with exog + intervals).

Both return a forecast indexed by the future timestamps. ``sarimax_forecast``
returns point + prediction-interval columns; ``seasonal_naive`` returns the point
path (conformal intervals can be layered on via :mod:`.intervals`).
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _future_index(train_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Build the next ``horizon`` timestamps following ``train_index``."""
    freq = pd.infer_freq(train_index) if len(train_index) >= 3 else None
    if freq is None:
        step = train_index[-1] - train_index[-2]
        return pd.DatetimeIndex([train_index[-1] + step * (i + 1) for i in range(horizon)])
    return pd.date_range(train_index[-1], periods=horizon + 1, freq=freq)[1:]


def seasonal_naive(train: pd.Series, horizon: int, season: int = 24) -> pd.Series:
    """Repeat the last full season forward: yhat[i] = train[-season + (i mod season)]."""
    if len(train) < season:
        raise ValueError(f"need >= {season} observations for seasonal_naive, got {len(train)}")
    last = train.to_numpy()[-season:]
    vals = np.array([last[i % season] for i in range(horizon)], dtype=float)
    return pd.Series(vals, index=_future_index(train.index, horizon), name="yhat")


def sarimax_forecast(
    train: pd.Series,
    exog: Optional[pd.DataFrame],
    horizon: int,
    future_exog: Optional[pd.DataFrame] = None,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    alpha: float = 0.1,
) -> pd.DataFrame:
    """statsmodels SARIMAX point + (1-alpha) prediction interval forecast.

    Returns a DataFrame indexed by the future timestamps with columns
    ``yhat``, ``lower``, ``upper``. Zero-variance exog columns are dropped
    (and matched in ``future_exog``) so tiny windows stay non-singular.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cols: Sequence[str] = []
    ex_tr = ex_fut = None
    if exog is not None and exog.shape[1]:
        # Drop (near-)constant columns: they are collinear with the trend='c'
        # intercept and make the design singular / raise in statsmodels.
        def _varies(c):
            a = exog[c].to_numpy(float)
            return np.nanstd(a) > 1e-8 * (1.0 + abs(np.nanmean(a)))
        cols = [c for c in exog.columns if _varies(c)]
        if cols:
            ex_tr = exog[cols].astype(float)
            if future_exog is not None:
                ex_fut = future_exog[cols].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence / non-stationarity chatter
        res = SARIMAX(
            train.astype(float), exog=ex_tr, order=order, seasonal_order=seasonal_order,
            trend="c", enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = res.get_forecast(steps=horizon, exog=ex_fut)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)

    idx = _future_index(train.index, horizon)
    return pd.DataFrame(
        {"yhat": mean.to_numpy(), "lower": ci.iloc[:, 0].to_numpy(), "upper": ci.iloc[:, 1].to_numpy()},
        index=idx,
    )
