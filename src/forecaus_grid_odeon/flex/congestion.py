"""Flexibility-need calculation for secondary-substation congestion relief.

Deterministic by design (auditable): given a day-ahead SS load forecast and the
substation limit(s), return the up/down flexibility volume and timing required
to keep load within bounds. No ML here — every number traces to inputs.

Beyond v0 this is **risk-adjusted**: it consumes the Slice-3 forecast *interval*
(``yhat``/``lower``/``upper``), not just the point. Down-flex is sized against
the UPPER forecast bound so the procured volume keeps even the worst-case high
load under the limit; up-flex (reverse-power relief) is sized against the LOWER
bound. The binding upper limit is the minimum of the thermal and contractual
limits (either or both may be given). Sizing against the point forecast is still
available (``against="point"``) and is what a Series input degenerates to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(value, index: pd.Index) -> pd.Series:
    """Broadcast a scalar / align a series-or-array onto ``index``."""
    if isinstance(value, pd.Series):
        return value.reindex(index).astype(float)
    if hasattr(value, "__len__") and not isinstance(value, str):
        return pd.Series(np.asarray(value, dtype=float), index=index)
    return pd.Series(float(value), index=index)


def _bands(forecast) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (point, lower, upper). A Series has no interval -> all three equal."""
    if isinstance(forecast, pd.DataFrame):
        point = (forecast["yhat"] if "yhat" in forecast else forecast.iloc[:, 0]).astype(float)
        lower = forecast["lower"].astype(float) if "lower" in forecast else point.copy()
        upper = forecast["upper"].astype(float) if "upper" in forecast else point.copy()
        return point, lower, upper
    point = pd.Series(forecast).astype(float)
    return point, point.copy(), point.copy()


def flexibility_need(
    forecast,
    ss_limit=None,
    *,
    lower_limit=0.0,
    thermal_limit=None,
    contractual_limit=None,
    against: str = "interval",
) -> pd.DataFrame:
    """Per-timestep flexibility to keep SS load within ``[lower_limit, upper]``.

    Parameters
    ----------
    forecast : pandas.Series | pandas.DataFrame
        SS load forecast. A Series is treated as a point forecast; a DataFrame
        with ``yhat``/``lower``/``upper`` carries the prediction interval.
    ss_limit, thermal_limit, contractual_limit : float | pandas.Series, optional
        Upper limits (scalar or per-timestep). The binding upper limit is their
        element-wise **minimum**; provide any combination. ``ss_limit`` is the
        v0 single-limit alias.
    lower_limit : float | pandas.Series
        Lower bound (default 0); reverse-power export can set this < 0.
    against : {"interval", "point"}
        Risk basis. ``"interval"`` (default) sizes down-flex against the UPPER
        bound and up-flex against the LOWER bound; ``"point"`` uses ``yhat``.
        (For a Series input the two are identical.)

    Returns
    -------
    pandas.DataFrame
        Columns ``forecast, lower, upper, ss_limit, lower_limit, flex_down,
        flex_up`` indexed by the forecast timestamps (timing). ``flex_down`` >= 0
        is curtailment to respect the upper limit; ``flex_up`` >= 0 is added load
        to respect the lower limit.
    """
    if against not in ("interval", "point"):
        raise ValueError("against must be 'interval' or 'point'")

    point, lower, upper = _bands(forecast)
    idx = point.index

    # Binding upper limit = min over the provided thermal / contractual / alias.
    provided = [lim for lim in (ss_limit, thermal_limit, contractual_limit) if lim is not None]
    if provided:
        cols = pd.concat([_as_series(lim, idx) for lim in provided], axis=1)
        upper_limit = cols.min(axis=1)
    else:
        upper_limit = pd.Series(np.inf, index=idx)
    lower_bound = _as_series(lower_limit, idx)

    # Risk basis: size against the interval bands (worst case) or the point.
    high = upper if against == "interval" else point     # drives down-flex
    low = lower if against == "interval" else point       # drives up-flex

    flex_down = (high - upper_limit).clip(lower=0.0)       # curtail to meet upper limit
    flex_up = (lower_bound - low).clip(lower=0.0)          # add load to meet lower limit

    return pd.DataFrame(
        {
            "forecast": point,
            "lower": lower,
            "upper": upper,
            "ss_limit": upper_limit,
            "lower_limit": lower_bound,
            "flex_down": flex_down,
            "flex_up": flex_up,
        },
        index=idx,
    )


def summarize_schedule(schedule: pd.DataFrame) -> dict:
    """Volume + timing summary of a flexibility schedule (auditable rollup)."""
    idx = schedule.index
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) >= 2 else 1.0
    fd, fu = schedule["flex_down"], schedule["flex_up"]
    down, up = fd > 0, fu > 0

    def _window(mask):
        hit = idx[mask]
        return (hit[0], hit[-1]) if len(hit) else (None, None)

    d0, d1 = _window(down)
    u0, u1 = _window(up)
    return {
        "dt_hours": dt_h,
        "n_breach_down": int(down.sum()),
        "peak_flex_down": float(fd.max()) if len(fd) else 0.0,
        "energy_flex_down_kwh": float(fd.sum() * dt_h),
        "first_breach_down": d0,
        "last_breach_down": d1,
        "n_breach_up": int(up.sum()),
        "peak_flex_up": float(fu.max()) if len(fu) else 0.0,
        "energy_flex_up_kwh": float(fu.sum() * dt_h),
        "first_breach_up": u0,
        "last_breach_up": u1,
    }
