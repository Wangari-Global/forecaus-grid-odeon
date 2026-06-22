"""Calendar features: cyclical hour/day-of-week/month + FR public holidays.

All features here are pointwise functions of the timestamp only — they never
look at the target or at neighbouring rows, so they cannot leak future
information across the train/test boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Civil calendar / holidays are defined in local time, not UTC.
_LOCAL_TZ = "Europe/Paris"


def _cyclical(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """sin/cos encoding so the model sees e.g. hour 23 as adjacent to hour 0."""
    rad = 2.0 * np.pi * values / period
    return np.sin(rad), np.cos(rad)


def add_calendar(df: pd.DataFrame, country: str = "FR") -> pd.DataFrame:
    """Append cyclical hour/dow/month encodings, a weekend flag and an FR
    public-holiday flag, keyed off the frame's :class:`~pandas.DatetimeIndex`.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_calendar requires a DatetimeIndex")

    out = df.copy()
    idx = out.index
    # Resolve civil-calendar fields in local time (handles UTC-stored data).
    local = idx.tz_convert(_LOCAL_TZ) if idx.tz is not None else idx

    out["hour_sin"], out["hour_cos"] = _cyclical(local.hour.to_numpy(), 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(local.dayofweek.to_numpy(), 7)
    out["month_sin"], out["month_cos"] = _cyclical(local.month.to_numpy(), 12)
    out["is_weekend"] = (local.dayofweek >= 5).astype("float64")

    import holidays as holidays_lib
    years = list(range(int(local.year.min()), int(local.year.max()) + 1))
    fr = holidays_lib.country_holidays(country, years=years)
    out["is_holiday"] = np.array([d in fr for d in local.date], dtype="float64")
    return out
