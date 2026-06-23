"""SS-level interpretable forecaster: named parameters, federation hook,
day-ahead benchmark vs baselines, and conformal coverage near nominal."""
import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.eval import rolling_origin, run_ss_benchmark
from forecaus_grid_odeon.features import build_ss_frame
from forecaus_grid_odeon.features.ss import periods_per_day
from forecaus_grid_odeon.fl import fedavg
from forecaus_grid_odeon.forecast import (StructuredForecaster, make_seasonal_naive,
                                          make_structured)

TARGET = "load_kw"


def _frame(days=20, seed=0, noise=1.0):
    """A smooth half-hourly feeder series + iid noise -> a built SS frame."""
    idx = pd.date_range("2025-01-06", periods=48 * days, freq="30min", tz="UTC")
    rng = np.random.default_rng(seed)
    hod = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    signal = (40 + 30 * np.exp(-0.5 * ((hod - 19) / 2.5) ** 2)
              + 8 * np.exp(-0.5 * ((hod - 8) / 1.5) ** 2)
              - 5 * (dow >= 5))
    y = pd.Series(signal + rng.normal(0, noise, len(idx)), index=idx, name=TARGET)
    return build_ss_frame(y, weather=None)


# --------------------------------------------- named params / federation hook ----
def test_structured_exposes_named_parameter_dict():
    frame = _frame()
    exog = [c for c in frame.columns if c != TARGET]
    model = StructuredForecaster(exog).fit(frame, TARGET)

    params = model.parameters()
    assert "intercept" in params
    assert set(params) - {"intercept"} == set(exog)            # one coef per feature
    assert all(np.isfinite(v) for v in params.values())
    assert all(isinstance(v, float) for v in params.values())


def test_set_parameters_round_trips():
    frame = _frame()
    exog = [c for c in frame.columns if c != TARGET]
    model = StructuredForecaster(exog).fit(frame, TARGET)
    params = model.parameters()

    clone = StructuredForecaster().set_parameters(params)
    np.testing.assert_allclose(clone.predict(frame), model.predict(frame))


def test_parameters_federate_with_fedavg():
    """Two feeders expose the same coefficient keys, so a FedAvg round is
    well-defined (the Slice-5 federation contract)."""
    fa = StructuredForecaster().fit(_frame(seed=1), TARGET).parameters()
    fb = StructuredForecaster().fit(_frame(seed=2), TARGET).parameters()
    assert set(fa) == set(fb)
    glob = fedavg([fa, fb], weights=[100, 50])
    assert set(glob) == set(fa)
    for k in glob:
        assert glob[k] == pytest.approx((100 * fa[k] + 50 * fb[k]) / 150)


# ----------------------------------------------------- accuracy + coverage --------
def test_structured_mape_finite_and_low():
    frame = _frame()
    exog = [c for c in frame.columns if c != TARGET]
    ppd = periods_per_day(frame.index)
    table = rolling_origin(
        frame, {"structured_gam": make_structured(features=exog)},
        horizon=ppd, step=ppd, target=TARGET,
    )
    mape = table.loc["structured_gam", "MAPE"]
    assert np.isfinite(mape)
    assert 0.0 <= mape < 15.0          # finite and trending toward the <5% target


def test_conformal_coverage_near_nominal():
    frame = _frame(days=24, noise=2.0)
    exog = [c for c in frame.columns if c != TARGET]
    ppd = periods_per_day(frame.index)
    table = rolling_origin(
        frame, {"structured_gam": make_structured(features=exog, alpha=0.1)},
        horizon=ppd, step=ppd, target=TARGET,
    )
    cov = table.loc["structured_gam", "coverage"]
    assert table.loc["structured_gam", "n_folds"] >= 3
    assert 0.80 <= cov <= 0.99         # ~ nominal 0.90 (alpha=0.1)


def test_structured_beats_seasonal_naive():
    """The structured model should not be worse than the naive baseline."""
    frame = _frame()
    exog = [c for c in frame.columns if c != TARGET]
    ppd = periods_per_day(frame.index)
    table = rolling_origin(
        frame,
        {"seasonal_naive": make_seasonal_naive(season=ppd),
         "structured_gam": make_structured(features=exog)},
        horizon=ppd, step=ppd, target=TARGET,
    )
    assert table.loc["structured_gam", "MAPE"] <= table.loc["seasonal_naive", "MAPE"] + 1e-6


# --------------------------------------------- end-to-end benchmark (offline) ----
def test_run_ss_benchmark_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "OFFLINE", True)

    result = run_ss_benchmark()

    per_feeder = result["per_feeder"]
    assert len(per_feeder) > 1, "federation/benchmark needs >1 feeder"

    agg = result["aggregate"]
    assert {"seasonal_naive", "sarimax", "structured_gam"} <= set(agg.index)
    assert np.isfinite(agg.loc["structured_gam", "MAPE"])
    assert {"MAE", "RMSE", "MAPE", "coverage"} <= set(agg.columns)

    # Slice-5 hook: a named parameter dict is produced.
    fid, params = result["params"]
    assert "intercept" in params and len(params) > 1
