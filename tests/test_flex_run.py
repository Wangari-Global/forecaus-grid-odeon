"""Risk-adjusted flexibility: interval-based sizing, thermal/contractual limits,
reverse power, and the forecast->flex pipeline schedule."""
import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.flex import flexibility_need, run_flex, summarize_schedule


def _forecast(idx, yhat, half_width):
    """A point + symmetric-interval forecast frame (Slice-3 shape)."""
    yhat = np.asarray(yhat, float)
    return pd.DataFrame(
        {"yhat": yhat, "lower": yhat - half_width, "upper": yhat + half_width}, index=idx
    )


# ----------------------------------------------- interval-based down-flex -----
def test_interval_sizing_uses_upper_bound():
    idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    fc = _forecast(idx, yhat=[100, 590, 620], half_width=30)   # upper = [130, 620, 650]
    out = flexibility_need(fc, ss_limit=600.0, against="interval")
    # Sized against the UPPER bound, not the point.
    assert list(out["flex_down"]) == [0.0, 20.0, 50.0]         # (620-600, 650-600)
    assert (out["flex_up"] == 0.0).all()


def test_interval_sizing_exceeds_point_sizing():
    idx = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
    fc = _forecast(idx, yhat=[580, 595, 610, 500], half_width=40)
    interval = flexibility_need(fc, ss_limit=600.0, against="interval")
    point = flexibility_need(fc, ss_limit=600.0, against="point")
    # Risk-adjusted (upper-bound) sizing never procures less than point sizing,
    # and strictly more whenever the band crosses the limit.
    assert (interval["flex_down"] >= point["flex_down"] - 1e-9).all()
    assert interval["flex_down"].sum() > point["flex_down"].sum()


# -------------------------------------------- thermal AND contractual limits --
def test_binding_limit_is_minimum_of_thermal_and_contractual():
    idx = pd.date_range("2026-01-01", periods=2, freq="30min", tz="UTC")
    fc = _forecast(idx, yhat=[500, 500], half_width=0)
    out = flexibility_need(fc, thermal_limit=520.0, contractual_limit=450.0)
    assert (out["ss_limit"] == 450.0).all()                    # min binds
    assert list(out["flex_down"]) == [50.0, 50.0]              # 500 - 450


def test_per_timestep_limit_series():
    idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    fc = _forecast(idx, yhat=[400, 500, 600], half_width=0)
    limit = pd.Series([450, 450, 450], index=idx)
    out = flexibility_need(fc, thermal_limit=limit)
    assert list(out["flex_down"]) == [0.0, 50.0, 150.0]


# ------------------------------------------ reverse-power (interval up-flex) --
def test_reverse_power_uses_lower_bound():
    idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    # Net export risk: the lower band dips below the export floor of -100.
    fc = _forecast(idx, yhat=[-50, 20, -90], half_width=40)    # lower = [-90, -20, -130]
    out = flexibility_need(fc, ss_limit=600.0, lower_limit=-100.0, against="interval")
    assert list(out["flex_up"]) == [0.0, 0.0, 30.0]            # (-100) - (-130)
    assert (out["flex_down"] == 0.0).all()


# --------------------------------------------------------- timing summary -----
def test_summarize_schedule_volume_and_timing():
    idx = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
    fc = _forecast(idx, yhat=[100, 700, 650, 100], half_width=0)
    sched = flexibility_need(fc, ss_limit=600.0)
    s = summarize_schedule(sched)
    assert s["dt_hours"] == pytest.approx(0.5)
    assert s["n_breach_down"] == 2
    assert s["peak_flex_down"] == pytest.approx(100.0)
    assert s["energy_flex_down_kwh"] == pytest.approx((100 + 50) * 0.5)   # kW * h
    assert s["first_breach_down"] == idx[1] and s["last_breach_down"] == idx[2]


# --------------------------------------------- forecast -> flex pipeline ------
@pytest.fixture(scope="module")
def flex_result(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    _r, _o = config.RAW, config.OFFLINE
    config.RAW, config.OFFLINE = raw, True
    try:
        return run_flex(horizon_steps=48)
    finally:
        config.RAW, config.OFFLINE = _r, _o


def test_run_flex_schedule_shape_and_timing(flex_result):
    sched = flex_result["schedule"]
    assert {"forecast", "lower", "upper", "ss_limit", "flex_down", "flex_up"} <= set(sched.columns)
    assert len(sched) == 48                                    # day-ahead horizon
    assert isinstance(sched.index, pd.DatetimeIndex)           # timing on the SS series
    assert str(sched.index.tz) == "UTC"
    assert (sched["flex_down"] >= 0).all() and (sched["flex_up"] >= 0).all()
    assert not sched.isna().any().any()


def test_run_flex_binding_limit_and_breach(flex_result):
    # Binding limit is the min of thermal/contractual, and the forecast breaches
    # it (a non-trivial relief schedule on the fixture).
    assert flex_result["binding_limit"] == min(
        flex_result["thermal_limit"], flex_result["contractual_limit"])
    assert flex_result["summary"]["n_breach_down"] > 0
    assert flex_result["summary"]["peak_flex_down"] > 0


def test_run_flex_is_risk_adjusted(flex_result):
    # Interval sizing procures at least as much down-flex energy as point sizing.
    assert (flex_result["summary"]["energy_flex_down_kwh"]
            >= flex_result["summary_point"]["energy_flex_down_kwh"] - 1e-9)
