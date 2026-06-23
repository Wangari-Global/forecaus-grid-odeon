"""Forecast -> flexibility pipeline: a Slice-3 interval forecast on an SS feeder
turned into a risk-adjusted congestion-relief schedule.

Flow: load a feeder's modelling frame -> day-ahead forecast with the transparent
conformal forecaster (point + interval) -> derive thermal/contractual limits ->
:func:`flexibility_need` sizes the per-timestep up/down flex against the interval.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .congestion import flexibility_need, summarize_schedule


def forecast_feeder(
    feeder_id: str, *, horizon_steps: int = 48, alpha: float = 0.1, l2: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Day-ahead Slice-3 interval forecast for one feeder.

    Returns ``(forecast, train, test)`` where ``forecast`` has ``yhat``/``lower``/
    ``upper`` indexed by the held-out final ``horizon_steps`` of the feeder.
    """
    from ..forecast import make_structured
    from ..pipeline import SS_TARGET, load_ss_frame

    frame = load_ss_frame(feeder_id)
    if len(frame) <= horizon_steps:
        raise RuntimeError(f"feeder {feeder_id} too short for a {horizon_steps}-step forecast")
    exog = [c for c in frame.columns if c != SS_TARGET]
    train, test = frame.iloc[:-horizon_steps], frame.iloc[-horizon_steps:]
    model = make_structured(features=exog, alpha=alpha, l2=l2)
    forecast = model(train, test, SS_TARGET)            # yhat / lower / upper
    return forecast, train, test


def run_flex(
    feeder_id: Optional[str] = None,
    *,
    thermal_q: float = 0.85,
    contractual_q: float = 0.92,
    lower_limit: float = 0.0,
    horizon_steps: int = 48,
) -> dict:
    """End-to-end forecast -> risk-adjusted flexibility schedule for one feeder.

    Thermal and contractual limits are derived from the feeder's own training
    history as load quantiles (a transparent, reproducible stand-in for the real
    SS rating / contract). The binding limit is their minimum.

    Returns a dict with the forecast, the interval-sized ``schedule`` (and the
    point-sized one for comparison), the limits, and volume/timing summaries.
    """
    from ..pipeline import SS_TARGET, ss_feeders

    feeder_id = feeder_id or ss_feeders()[0]
    forecast, train, test = forecast_feeder(feeder_id, horizon_steps=horizon_steps)

    hist = train[SS_TARGET]
    thermal_limit = float(hist.quantile(thermal_q))
    contractual_limit = float(hist.quantile(contractual_q))

    schedule = flexibility_need(
        forecast, thermal_limit=thermal_limit, contractual_limit=contractual_limit,
        lower_limit=lower_limit, against="interval",
    )
    point_schedule = flexibility_need(
        forecast, thermal_limit=thermal_limit, contractual_limit=contractual_limit,
        lower_limit=lower_limit, against="point",
    )

    return {
        "feeder_id": feeder_id,
        "forecast": forecast,
        "schedule": schedule,
        "point_schedule": point_schedule,
        "thermal_limit": thermal_limit,
        "contractual_limit": contractual_limit,
        "binding_limit": min(thermal_limit, contractual_limit),
        "summary": summarize_schedule(schedule),
        "summary_point": summarize_schedule(point_schedule),
    }
