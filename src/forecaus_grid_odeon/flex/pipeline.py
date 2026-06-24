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
    feeder_id: str, *, level: str = "feeder", horizon_steps: int = 48,
    alpha: float = 0.1, l2: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Day-ahead Slice-3 interval forecast for one unit.

    ``level`` selects ``"feeder"`` (one LV feeder) or ``"substation"`` (the
    SS-TOTAL series). Returns ``(forecast, train, test)`` with ``yhat``/``lower``/
    ``upper`` over the held-out final ``horizon_steps``.
    """
    from ..forecast import make_structured
    from ..pipeline import SS_TARGET, load_ss_agg_frame, load_ss_frame

    frame = load_ss_agg_frame(feeder_id) if level == "substation" else load_ss_frame(feeder_id)
    if len(frame) <= horizon_steps:
        raise RuntimeError(f"{feeder_id} too short for a {horizon_steps}-step forecast")
    exog = [c for c in frame.columns if c != SS_TARGET]
    train, test = frame.iloc[:-horizon_steps], frame.iloc[-horizon_steps:]
    model = make_structured(features=exog, alpha=alpha, l2=l2)
    forecast = model(train, test, SS_TARGET)            # yhat / lower / upper
    return forecast, train, test


#: How the (illustrative) SS transformer firm-capacity limit is set. Documented so
#: it is never mistaken for the real nameplate rating.
SS_LIMIT_BASIS = (
    "ILLUSTRATIVE limit = the {q:.0%} percentile of the substation's own historical "
    "half-hourly SS-total load — a transparent stand-in for the secondary-transformer "
    "firm rating, deliberately within the operating range so the schedule is "
    "non-trivial. Replace with the real nameplate / seasonal rating once the pilot "
    "provides it (that would typically sit above normal peak, leaving headroom)."
)


def run_flex(
    feeder_id: Optional[str] = None,
    *,
    level: str = "feeder",
    thermal_q: float = 0.85,
    contractual_q: float = 0.92,
    lower_limit: float = 0.0,
    horizon_steps: int = 48,
) -> dict:
    """End-to-end forecast -> risk-adjusted flexibility schedule for one unit.

    ``level="substation"`` runs on the SS-TOTAL series against a substation
    transformer limit; ``"feeder"`` runs on one LV feeder. The thermal /
    contractual limits are percentiles of the unit's own training history — an
    ILLUSTRATIVE stand-in for the real SS rating / contract (see
    :data:`SS_LIMIT_BASIS`); the binding limit is their minimum.
    """
    from ..pipeline import SS_TARGET, ss_agg_substations, ss_feeders

    if feeder_id is None:
        if level == "substation":
            from .. import config
            from ..ingest_ss import aggregate

            def _peak(s):
                if config.OFFLINE:
                    return float(aggregate.load_agg_fixture(s)[SS_TARGET].max())
                series = aggregate.ukpn.normalise(
                    pd.read_parquet(config.RAW / aggregate.AGG_SUBDIR / f"{s}.parquet"))
                return float(series[SS_TARGET].max())

            feeder_id = max(ss_agg_substations(), key=_peak)   # busiest = natural congestion demo
        else:
            feeder_id = ss_feeders()[0]
    forecast, train, test = forecast_feeder(feeder_id, level=level, horizon_steps=horizon_steps)

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
        "level": level,
        "forecast": forecast,
        "schedule": schedule,
        "point_schedule": point_schedule,
        "thermal_limit": thermal_limit,
        "contractual_limit": contractual_limit,
        "binding_limit": min(thermal_limit, contractual_limit),
        "limit_basis": SS_LIMIT_BASIS.format(q=min(thermal_q, contractual_q)),
        "summary": summarize_schedule(schedule),
        "summary_point": summarize_schedule(point_schedule),
    }
