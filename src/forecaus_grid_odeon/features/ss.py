"""Per-substation (LV-feeder) modelling frames for the federated setup.

Each LV feeder is one federated node: this module turns a feeder's ``load_kw``
series into a tidy, aligned, NA-free modelling frame using the same causal
machinery as the national pipeline (:func:`..features.build`), so the leakage
guarantees carry over verbatim:

  * **calendar** - cyclical hour/dow/month + holidays for the dataset's country
    (GB for the UK Power Networks feeders); pointwise, cannot leak.
  * **lagged load** - strictly past ``shift(k>=1)``; the lag/rolling depth is
    adapted to the series' sampling rate and length (1 step, one day, one week)
    so short samples still build.
  * **weather** - Open-Meteo (keyless), reindexed onto the load index; joined
    only when fully available over that index (else omitted, keeping frames
    NA-free offline).
  * **topology** - optional static per-feeder attributes (e.g. transformer
    rating, supply-point count, position) broadcast as constant ``topo_*``
    columns when present. Static => no leakage.

The train/test split is a single chronological cut at a configurable boundary;
because every feature is causal, the cut introduces no leakage.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import config
from .build_features import build


def periods_per_day(index: pd.DatetimeIndex) -> int:
    """Number of samples per day inferred from the index spacing (>=1)."""
    if len(index) < 2:
        return 24
    step = pd.Series(index).diff().dropna().median()
    secs = step.total_seconds()
    if secs <= 0:
        return 24
    return max(1, int(round(86400.0 / secs)))


def adapt_lags(
    n_obs: int, ppd: int, ppw: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose lag steps and rolling windows that fit ``n_obs`` observations.

    Candidates are 1 step, one day (``ppd``) and one week (``ppw``); each is
    kept only if it still leaves a usable training tail (>= one day of rows),
    so the same code works on the tiny offline sample and on full history.
    """
    keep_floor = max(4, ppd)                       # rows that must survive warm-up
    max_step = n_obs - keep_floor
    lags = tuple(k for k in dict.fromkeys((1, ppd, ppw)) if 1 <= k <= max_step)
    if not lags:
        lags = (1,) if n_obs > 2 else ()
    rolling = tuple(w for w in dict.fromkeys((ppd, ppw)) if 1 < w <= max_step)
    return lags, rolling


def _aligned_weather(
    weather: Optional[pd.DataFrame], index: pd.DatetimeIndex
) -> Optional[pd.DataFrame]:
    """Reindex weather onto the load index; return it only if fully populated
    there (otherwise omit, so the frame stays NA-free without back-fill)."""
    if weather is None or weather.empty:
        return None
    aligned = weather.reindex(index).interpolate(limit_direction="both")
    if aligned.isna().any().any():
        return None
    return aligned


def _topology_columns(
    topology: Optional[Mapping[str, float]], index: pd.DatetimeIndex
) -> Optional[pd.DataFrame]:
    """Broadcast static per-feeder topology attributes to constant columns."""
    if not topology:
        return None
    data = {f"topo_{k}": float(v) for k, v in topology.items()}
    return pd.DataFrame(data, index=index, dtype="float64")


def build_ss_frame(
    series: pd.Series,
    weather: Optional[pd.DataFrame] = None,
    topology: Optional[Mapping[str, float]] = None,
    *,
    country: str = config.SS_COUNTRY,
    lags: Optional[Sequence[int]] = None,
    rolling_windows: Optional[Sequence[int]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build one feeder's modelling frame (target first column).

    ``lags`` / ``rolling_windows`` default to a length-adapted set. ``weather``
    is joined only when fully available over the load index; ``topology`` (if
    given) adds constant ``topo_*`` columns. The result is aligned, sorted and
    NA-free.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by a DatetimeIndex")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]

    ppd = periods_per_day(series.index)
    ppw = ppd * 7
    if lags is None or rolling_windows is None:
        a_lags, a_roll = adapt_lags(len(series), ppd, ppw)
        lags = a_lags if lags is None else lags
        rolling_windows = a_roll if rolling_windows is None else rolling_windows

    wx = _aligned_weather(weather, series.index)
    frame = build(
        series, wx, None,
        lags=lags, rolling_windows=rolling_windows,
        country=country, dropna=dropna,
    )

    topo = _topology_columns(topology, frame.index)
    if topo is not None:
        frame = frame.join(topo, how="left")
    return frame


def ss_train_test_split(
    frame: pd.DataFrame, split=config.SS_SPLIT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test cut at ``split`` (a fraction in (0,1) of the
    rows, or a UTC date string). No feature recompute and a strict boundary, so
    the split introduces no train/test leakage."""
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()

    if isinstance(split, str):
        boundary = pd.Timestamp(split, tz=frame.index.tz)
        train = frame.loc[frame.index <= boundary]
        test = frame.loc[frame.index > boundary]
    else:
        frac = float(split)
        if not 0.0 < frac < 1.0:
            raise ValueError("split fraction must be in (0, 1)")
        cut = int(np.ceil(len(frame) * frac))
        cut = min(max(cut, 1), len(frame) - 1)     # keep both sides non-empty
        train, test = frame.iloc[:cut], frame.iloc[cut:]

    if len(train) == 0 or len(test) == 0:
        raise ValueError("split produced an empty train or test partition")
    # Strict chronological boundary: no timestamp appears on both sides.
    assert train.index.max() < test.index.min()
    return train, test
