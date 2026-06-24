"""Day-ahead SS-level benchmark: structured GAM vs seasonal-naive vs SARIMAX.

For each LV feeder we walk-forward a one-day-ahead forecast with
:func:`rolling_origin` and report MAE/RMSE/MAPE + conformal interval coverage,
then aggregate (mean) across feeders. The structured GAM is the interpretable,
federatable edge model (:class:`..forecast.StructuredForecaster`); the other two
are the required baselines. Horizon and seasonality adapt to each frame's
sampling rate (48 steps/day for the half-hourly UKPN feeders).
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from ..features.ss import periods_per_day
from ..forecast import StructuredForecaster, make_sarimax, make_seasonal_naive, make_structured
from .backtest import rolling_origin

METRIC_COLS = ["MAE", "RMSE", "MAPE", "coverage", "n_folds"]


def leaksafe_features(exog: list[str], target: str, horizon: int) -> list[str]:
    """Features KNOWN at forecast time for an ``horizon``-step-ahead forecast.

    Drops the target's intra-horizon lags (``lag_k`` with ``k < horizon``) and
    ALL rolling windows (whose windows slide into the forecast horizon) — both
    would leak future load into a day-ahead forecast. Keeps calendar + weather +
    the >= horizon lags (24 h / 168 h), i.e. the *proper* day-ahead lags.
    """
    safe = []
    for c in exog:
        if c.startswith(f"{target}_roll"):
            continue
        if c.startswith(f"{target}_lag_"):
            try:
                if int(c.rsplit("_", 1)[1]) < horizon:
                    continue
            except ValueError:
                continue
        safe.append(c)
    return safe


def _panel(exog: list[str], target: str, ppd: int, alpha: float) -> dict:
    """Day-ahead model panel: structured GAM + the two required baselines.

    All three are **leak-free** for an ``ppd``-step-ahead forecast:
    seasonal-naive repeats the last day; SARIMAX gets only exogenous calendar/
    weather (its AR is on past target); and the structured GAM uses the
    known-at-forecast-time feature set (:func:`leaksafe_features`) — calendar +
    weather + the 24 h / 168 h lags, NOT the intra-horizon lag_1 / rolling stats.
    Light ridge tuning (l2=10) over those ~13 features.
    """
    sarimax_exog = [
        c for c in exog
        if not c.startswith(f"{target}_lag") and not c.startswith(f"{target}_roll")
    ]
    return {
        "seasonal_naive": make_seasonal_naive(season=ppd, alpha=alpha),
        # Non-seasonal (1,1,1) with calendar/weather exog: the calendar carries
        # the daily/weekly shape, and this is ~50x faster than a period-ppd
        # seasonal SARIMA on multi-week half-hourly history (which is intractable).
        "sarimax": make_sarimax(
            exog_cols=sarimax_exog, order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0), alpha=alpha,
        ),
        "structured_gam": make_structured(
            features=leaksafe_features(exog, target, ppd), l2=10.0, alpha=alpha),
    }


def run_ss_benchmark(
    frames: Optional[Iterable[tuple[str, pd.DataFrame]]] = None,
    *,
    target: str = "load_kw",
    alpha: float = 0.1,
    max_feeders: Optional[int] = None,
) -> dict:
    """Benchmark every feeder day-ahead; return per-feeder + aggregate tables.

    Returns ``{"per_feeder": {fid: table}, "aggregate": table, "params":
    (fid, parameter_dict)}``. ``params`` is the structured model's named
    coefficients fit on the first feeder - the audit / federation (Slice 5) hook.
    """
    if frames is None:
        from ..pipeline import iter_ss_frames
        frames = iter_ss_frames()

    per_feeder: dict[str, pd.DataFrame] = {}
    params: Optional[tuple[str, dict]] = None

    for i, (fid, frame) in enumerate(frames):
        if max_feeders is not None and i >= max_feeders:
            break
        ppd = periods_per_day(frame.index)             # day-ahead horizon & season
        exog = [c for c in frame.columns if c != target]
        table = rolling_origin(
            frame, _panel(exog, target, ppd, alpha),
            horizon=ppd, step=ppd, target=target,
        )
        per_feeder[fid] = table
        if params is None:
            fitted = StructuredForecaster(
                leaksafe_features(exog, target, ppd), l2=10.0).fit(frame, target)
            params = (fid, fitted.parameters())

    if not per_feeder:
        raise RuntimeError("no feeders available to benchmark")

    return {
        "per_feeder": per_feeder,
        "aggregate": aggregate_tables(per_feeder),
        "params": params,
    }


def aggregate_tables(per_feeder: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean of each metric across feeders, per model (one row per model)."""
    stacked = pd.concat(per_feeder.values())
    cols = [c for c in METRIC_COLS if c in stacked.columns]
    agg = stacked.groupby(level=0)[cols].mean()
    agg["n_feeders"] = pd.concat(per_feeder.values()).groupby(level=0).size()
    agg.index.name = "model"
    return agg
