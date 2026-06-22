"""Flexibility-need calculation for secondary-substation congestion relief (v0).

Deterministic by design (auditable): given a day-ahead SS load forecast and the
substation limit, return the up/down flexibility volume and timing required to
keep load within the limit. No ML here — every number traces to inputs.
"""
from __future__ import annotations


def flexibility_need(forecast, ss_limit, *, lower_limit=0.0):
    """Required flexibility per timestep to keep SS load within [lower, ss_limit].

    Parameters
    ----------
    forecast : pandas.Series
        Day-ahead SS load forecast (e.g. kW or MW), timestamp-indexed.
    ss_limit : float | pandas.Series
        Upper limit (thermal/contractual). Scalar or per-timestep series.
    lower_limit : float | pandas.Series
        Lower bound (default 0); reverse-power congestion can set this < 0.

    Returns
    -------
    pandas.DataFrame
        Columns: forecast, ss_limit, flex_down (curtail to meet upper limit,
        >=0), flex_up (raise to meet lower limit, >=0). Timing = the index.

    TODO: distinguish thermal vs contractual limits; aggregate to a single
    procurement envelope; carry forecast confidence interval through to a
    risk-adjusted need. See ODEON_Challenge4_FL_Architecture.docx section 6.
    """
    import pandas as pd

    f = pd.Series(forecast).astype(float)
    upper = ss_limit if hasattr(ss_limit, "__len__") else pd.Series(ss_limit, index=f.index)
    lower = lower_limit if hasattr(lower_limit, "__len__") else pd.Series(lower_limit, index=f.index)
    flex_down = (f - upper).clip(lower=0.0)      # shed/curtail to respect upper limit
    flex_up = (lower - f).clip(lower=0.0)        # add load to respect lower limit
    return pd.DataFrame(
        {"forecast": f, "ss_limit": upper, "flex_down": flex_down, "flex_up": flex_up}
    )
