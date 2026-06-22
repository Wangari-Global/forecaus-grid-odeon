"""Feature-engineering tests: alignment, NA-freeness, and no future leakage."""
import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon.features import add_calendar, build, train_test_split
from forecaus_grid_odeon.features.build_features import DEFAULT_LAGS, DEFAULT_ROLLING


@pytest.fixture
def series():
    """3 weeks of hourly UTC data — long enough to exercise the 168h features."""
    idx = pd.date_range("2024-12-15", periods=24 * 21, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    y = pd.Series(
        50000 + 5000 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi) + rng.normal(0, 300, len(idx)),
        index=idx, name="load_mw",
    )
    weather = pd.DataFrame({
        "temp_c": 15 + rng.normal(0, 2, len(idx)),
        "wind_ms": np.abs(rng.normal(4, 1, len(idx))),
        "irradiance_wm2": np.clip(rng.normal(200, 100, len(idx)), 0, None),
    }, index=idx)
    price = pd.DataFrame({"day_ahead_price_eur_mwh": 40 + rng.normal(0, 5, len(idx))}, index=idx)
    return y, weather, price


def test_build_aligned_and_na_free(series):
    y, weather, price = series
    frame = build(y, weather, price)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert not frame.isna().any().any(), "frame must be NA-free"
    # First column is the target; predictors include calendar + exog + lags.
    assert frame.columns[0] == "load_mw"
    assert {"hour_sin", "hour_cos", "is_holiday", "temp_c",
            "day_ahead_price_eur_mwh"} <= set(frame.columns)
    # Warm-up of the longest window/lag is dropped.
    assert frame.index.min() == y.index[max((*DEFAULT_LAGS, *DEFAULT_ROLLING))]


def test_no_future_leakage_in_lags(series):
    """lag_k at time t must equal the raw target at t-k (strictly past)."""
    y, weather, price = series
    frame = build(y, weather, price)
    for k in DEFAULT_LAGS:
        col = frame[f"load_mw_lag_{k}"]
        expected = y.shift(k).reindex(frame.index)
        pd.testing.assert_series_equal(col, expected, check_names=False)
        # A row's own target value never appears in its lag columns.
        assert not np.allclose(col.to_numpy(), frame["load_mw"].to_numpy())


def test_rolling_uses_only_past(series):
    """rollmean_w at t must be the mean of y[t-w : t] (excludes y[t])."""
    y, _, _ = series
    frame = build(y, dropna=True)
    w = DEFAULT_ROLLING[0]
    manual = y.shift(1).rolling(w).mean().reindex(frame.index)
    pd.testing.assert_series_equal(frame[f"load_mw_rollmean_{w}"], manual, check_names=False)
    t = frame.index[100]
    pos = y.index.get_loc(t)
    assert frame.loc[t, f"load_mw_rollmean_{w}"] == pytest.approx(y.iloc[pos - w:pos].mean())


def test_calendar_is_pointwise_and_bounded(series):
    y, _, _ = series
    cal = add_calendar(y.to_frame())
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert cal[c].between(-1.0, 1.0).all()
    assert set(cal["is_weekend"].unique()) <= {0.0, 1.0}
    # 2025-01-01 (New Year) is a FR holiday; a random Tuesday is not.
    assert cal.loc["2025-01-01 12:00+00:00", "is_holiday"] == 1.0
    assert cal.loc["2024-12-17 12:00+00:00", "is_holiday"] == 0.0


def test_train_test_split_no_overlap(series):
    y, weather, price = series
    frame = build(y, weather, price)
    train, test = train_test_split(frame)  # config: 2024-12-31 / 2025-01-01
    assert len(train) and len(test)
    assert train.index.max() <= pd.Timestamp("2024-12-31 23:59", tz="UTC")
    assert test.index.min() >= pd.Timestamp("2025-01-01", tz="UTC")
    assert train.index.intersection(test.index).empty


def test_integration_with_ingest_fixtures():
    """build() works end-to-end on the committed ingest sample (shorter lags)."""
    from forecaus_grid_odeon import config
    config_offline = config.OFFLINE
    config.OFFLINE = True
    try:
        from forecaus_grid_odeon.ingest._io import load_fixture
        load = load_fixture("entsoe_load")["load_mw"]
        weather = load_fixture("weather")
        price = load_fixture("entsoe_price")
        frame = build(load, weather, price, lags=(1, 24), rolling_windows=(24,))
    finally:
        config.OFFLINE = config_offline
    assert not frame.isna().any().any()
    assert len(frame) == len(load) - 24  # warm-up of the 24h window dropped
