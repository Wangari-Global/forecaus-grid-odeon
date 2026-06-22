"""Glue: ingest (cached/offline) -> feature frame ready for forecasting."""
from __future__ import annotations

import pandas as pd

from . import config
from .features import build


def load_frame() -> pd.DataFrame:
    """Load cached/offline data and build the modelling frame (target first column).

    Lag/rolling depth adapts to series length so the pipeline also works on the
    tiny offline sample (which is too short for a 168h lag).
    """
    from .ingest import entsoe, weather

    load = entsoe.fetch_load()["load_mw"]
    wx = weather.fetch_weather(index=load.index)
    price = entsoe.fetch_day_ahead_price()

    if len(load) > 24 * 14:                 # >2 weeks: use weekly features
        lags, rolling = (1, 24, 168), (24, 168)
    else:                                   # short sample: daily only
        lags, rolling = (1, 24), (24,)
    frame = build(load, wx, price, lags=lags, rolling_windows=rolling)
    if frame.empty:
        raise RuntimeError("feature frame is empty — ingest produced too little data")
    return frame


TARGET = "load_mw"
