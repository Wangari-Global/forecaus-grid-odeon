"""DARTS Temporal Fusion Transformer forecaster (quantile / probabilistic).

The TFT consumes the target history plus **future covariates** — calendar,
weather and price, which are known at forecast time — and is trained with a
``QuantileRegression`` likelihood so it emits a full predictive distribution.
Point forecast = the 0.5 quantile; intervals come straight from the predicted
quantiles.

torch/lightning are imported lazily (inside the functions) so importing
``forecaus_grid_odeon`` never requires the heavy DL stack; install it via
``pip install 'darts[torch]'`` (already in this project's dependencies).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """DARTS needs a tz-naive, regular index; keep wall-clock, drop the tz."""
    return idx.tz_localize(None) if idx.tz is not None else idx


def tft_forecast(
    train_target: pd.Series,
    covariates: pd.DataFrame,
    horizon: int,
    *,
    alpha: float = 0.1,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    input_chunk_length: int = 72,
    hidden_size: int = 16,
    n_epochs: int = 30,
    batch_size: int = 32,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a TFT on ``train_target`` with ``covariates`` (must span the train
    window **and** the ``horizon`` steps after it) and forecast ``horizon`` steps.

    Returns a DataFrame indexed by the future timestamps with columns
    ``yhat`` (median), ``lower`` (alpha/2 quantile) and ``upper`` (1-alpha/2).
    """
    import torch
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TFTModel
    from darts.utils.likelihood_models import QuantileRegression

    qlo, qhi = alpha / 2, 1 - alpha / 2
    qset = sorted(set([*quantiles, qlo, 0.5, qhi]))  # ensure interval quantiles exist

    icl = min(input_chunk_length, len(train_target) - horizon)
    if icl < 8:
        raise ValueError(f"train too short for TFT: need >= {horizon + 8}, got {len(train_target)}")

    torch.manual_seed(random_state)
    target = TimeSeries.from_times_and_values(
        _naive_index(train_target.index), train_target.to_numpy(dtype=float)
    )
    cov = TimeSeries.from_times_and_values(
        _naive_index(covariates.index), covariates.to_numpy(dtype=float)
    )

    # Scale target (fit on train) and covariates (fit on the train span only).
    y_scaler = Scaler()
    target_s = y_scaler.fit_transform(target)
    cov_scaler = Scaler()
    cov_scaler.fit(cov[: len(train_target)])
    cov_s = cov_scaler.transform(cov)

    model = TFTModel(
        input_chunk_length=icl,
        output_chunk_length=horizon,
        hidden_size=hidden_size,
        lstm_layers=1,
        num_attention_heads=2,
        dropout=0.1,
        batch_size=batch_size,
        n_epochs=n_epochs,
        likelihood=QuantileRegression(quantiles=qset),
        random_state=random_state,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
    )
    model.fit(target_s, future_covariates=cov_s)

    pred_s = model.predict(n=horizon, future_covariates=cov_s, num_samples=300)
    pred = y_scaler.inverse_transform(pred_s)

    def q(level: float) -> np.ndarray:
        return pred.quantile(level).values().ravel()

    future = covariates.index[len(train_target): len(train_target) + horizon]
    return pd.DataFrame({"yhat": q(0.5), "lower": q(qlo), "upper": q(qhi)}, index=future)
