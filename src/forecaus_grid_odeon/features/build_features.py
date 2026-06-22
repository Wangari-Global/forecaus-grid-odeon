"""Assemble the modelling frame: target + lags + rolling + weather + calendar + price.

Leakage policy (the whole point of this module):
  * Target **lag** and **rolling** features use strictly PAST values
    (``shift(k>=1)``; rolling is computed on ``shift(1)`` so the current step is
    excluded). A row at time *t* therefore never embeds the target at *t* or later.
  * Exogenous covariates (weather, day-ahead price) are treated as KNOWN at
    forecast time (future covariates) and joined contemporaneously; their gaps
    are forward-filled only (past -> future), never back-filled or interpolated.
  * Every transform is causal, so building features on the full series and then
    splitting at ``config.TRAIN_END`` / ``config.TEST_START`` introduces no
    train/test leakage. ``train_test_split`` does the date cut without recompute.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .. import config
from .calendar import add_calendar

DEFAULT_LAGS: tuple[int, ...] = (1, 24, 168)            # hour-, day-, week-ago
DEFAULT_ROLLING: tuple[int, ...] = (24, 168)            # daily / weekly windows


def build(
    target: pd.Series,
    weather: Optional[pd.DataFrame] = None,
    price: Optional[pd.DataFrame] = None,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING,
    country: str = "FR",
    dropna: bool = True,
) -> pd.DataFrame:
    """Join sources and engineer features; return an aligned, NA-free frame.

    The frame's first column is the target ``y``; remaining columns are
    predictors. Rows in the warm-up window (where the longest lag/rolling
    feature is undefined) are dropped when ``dropna`` is set.
    """
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target must be indexed by a DatetimeIndex")
    if any(k < 1 for k in lags) or any(w < 1 for w in rolling_windows):
        raise ValueError("lags and rolling windows must be >= 1 (past-only)")

    target = target.sort_index()
    name = target.name or "y"
    frame = target.rename(name).to_frame()

    # --- exogenous covariates (known at forecast time): join then ffill gaps ---
    for src in (weather, price):
        if src is not None:
            frame = frame.join(src, how="left")
    exog = [c for c in frame.columns if c != name]
    if exog:
        frame[exog] = frame[exog].ffill()  # past-only fill; no look-ahead

    # --- calendar (pointwise; cannot leak) ---
    frame = add_calendar(frame, country=country)

    # --- target lags: strictly past ---
    for k in lags:
        frame[f"{name}_lag_{k}"] = target.shift(k)

    # --- rolling stats over PAST values only (shift(1) excludes current step) ---
    past = target.shift(1)
    for w in rolling_windows:
        frame[f"{name}_rollmean_{w}"] = past.rolling(w).mean()
        frame[f"{name}_rollstd_{w}"] = past.rolling(w).std()

    if dropna:
        frame = frame.dropna()
    return frame


def train_test_split(
    frame: pd.DataFrame,
    train_end: str = config.TRAIN_END,
    test_start: str = config.TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an already-built (causal) frame by date. No feature recompute, so
    no leakage is introduced by the split itself."""
    tz = frame.index.tz
    end = pd.Timestamp(train_end, tz=tz)
    start = pd.Timestamp(test_start, tz=tz)
    if start <= end:
        raise ValueError("test_start must be after train_end")
    train = frame.loc[frame.index <= end]
    test = frame.loc[frame.index >= start]
    return train, test
