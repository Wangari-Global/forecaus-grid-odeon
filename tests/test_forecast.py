"""Tests for metrics, baselines, intervals, and one rolling-origin fold."""
import math

import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon.eval import rolling_origin, summary
from forecaus_grid_odeon.eval.metrics import coverage, mae, mape, rmse
from forecaus_grid_odeon.forecast import (conformal_width, default_models,
                                    make_conformal, make_seasonal_naive,
                                    seasonal_naive)


# ---------------------------------------------------------------- metrics ----
def test_metric_values():
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
    assert rmse([0, 0, 0], [0, 3, 4]) == pytest.approx(math.sqrt(25 / 3))
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert coverage([1, 2, 3], [0, 0, 0], [2, 2, 2]) == pytest.approx(2 / 3)


def test_mape_handles_zeros():
    assert math.isnan(mape([0, 0], [1, 2]))            # all-zero actuals -> nan
    assert mape([0, 100], [50, 110]) == pytest.approx(10.0)  # zero entry ignored


def test_summary_keys():
    s = summary([1, 2, 3], [1, 2, 3], [0, 0, 0], [4, 4, 4])
    assert set(s) == {"MAE", "RMSE", "MAPE", "coverage"}
    assert s["MAE"] == 0.0 and s["coverage"] == 1.0


# --------------------------------------------------------------- baselines ----
def _series(n=96, season=24):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    vals = 50000 + 5000 * np.sin(np.arange(n) / season * 2 * np.pi)
    return pd.Series(vals, index=idx, name="load_mw")


def test_seasonal_naive_repeats_last_season():
    y = _series()
    fc = seasonal_naive(y, horizon=24, season=24)
    assert len(fc) == 24
    # On a clean period-24 signal, last season repeats exactly.
    np.testing.assert_allclose(fc.to_numpy(), y.to_numpy()[-24:])
    # Index continues immediately after the training series.
    assert fc.index[0] == y.index[-1] + pd.Timedelta(hours=1)


def test_conformal_width_nonneg_and_monotone():
    res = np.array([-3.0, 1.0, -1.0, 2.0, 0.5])
    w90 = conformal_width(res, alpha=0.1)
    w50 = conformal_width(res, alpha=0.5)
    assert w90 >= w50 >= 0.0


def test_conformal_coverage_is_nominal():
    """On exchangeable residuals, conformal intervals cover ~ (1 - alpha)."""
    rng = np.random.default_rng(0)
    calib = rng.standard_normal(5000)              # calibration residuals
    width = conformal_width(calib, alpha=0.1)      # target 90% coverage
    fresh = rng.standard_normal(50000)             # exchangeable test residuals
    cov = float(np.mean(np.abs(fresh) <= width))
    assert 0.88 <= cov <= 0.92                     # ~0.90


def test_make_conformal_wrapper_intervals():
    """Split-conformal wrapper yields ordered intervals and ~nominal coverage
    on a well-specified point model with iid noise."""
    idx = pd.date_range("2025-01-01", periods=1200, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    signal = 100 + 10 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi)
    y = pd.Series(signal + rng.normal(0, 5, len(idx)), index=idx, name="load_mw")
    frame = y.to_frame()

    # Base model = "yesterday, same hour" (a good point predictor here).
    conf = make_conformal(make_seasonal_naive(season=24), alpha=0.1)
    train, test = frame.iloc[:-480], frame.iloc[-480:]
    pred = conf(train, test, "load_mw")
    assert (pred["lower"] <= pred["yhat"]).all() and (pred["yhat"] <= pred["upper"]).all()
    cov = float(((test["load_mw"] >= pred["lower"]) & (test["load_mw"] <= pred["upper"])).mean())
    assert 0.82 <= cov <= 0.97                      # close to nominal 0.90


def test_sarimax_model_runs_with_exog():
    y = _series(120)
    exog = pd.DataFrame({"temp_c": np.cos(np.arange(120) / 24 * 2 * np.pi)}, index=y.index)
    frame = pd.concat([y, exog], axis=1)
    train, test = frame.iloc[:-24], frame.iloc[-24:]
    model = default_models(exog_cols=["temp_c"])["sarimax"]
    pred = model(train, test, "load_mw")
    assert list(pred.columns) == ["yhat", "lower", "upper"]
    assert pred.index.equals(test.index)
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred["lower"] <= pred["upper"]).all()


# ----------------------------------------------------- rolling-origin fold ----
def test_rolling_origin_single_fold():
    y = _series(72)
    frame = y.to_frame()
    # initial=48, step=24, horizon=24 -> origins=[48] -> exactly one fold.
    table = rolling_origin(
        frame, {"seasonal_naive": make_seasonal_naive(season=24)},
        horizon=24, step=24, initial=48, target="load_mw",
    )
    assert list(table.index) == ["seasonal_naive"]
    assert {"MAE", "RMSE", "MAPE", "coverage", "n_folds"} <= set(table.columns)
    assert table.loc["seasonal_naive", "n_folds"] == 1
    mape_val = table.loc["seasonal_naive", "MAPE"]
    assert np.isfinite(mape_val) and 0 <= mape_val < 50  # finite & reasonable


def test_rolling_origin_multi_fold_panel():
    y = _series(144)
    frame = y.to_frame()
    table = rolling_origin(frame, default_models(exog_cols=[]), horizon=24, target="load_mw")
    assert table.loc["seasonal_naive", "n_folds"] >= 1
    assert np.isfinite(table.loc["seasonal_naive", "MAPE"])
