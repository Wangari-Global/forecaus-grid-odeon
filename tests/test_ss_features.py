"""Per-substation (LV-feeder) feature frames: alignment, NA-freeness, no
future leakage, and that >1 feeder is available for federation."""
import re

import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon import config, pipeline
from forecaus_grid_odeon.features import (add_calendar, adapt_lags,
                                          build_ss_frame, ss_train_test_split)
from forecaus_grid_odeon.features.ss import periods_per_day


@pytest.fixture
def feeder_series():
    """~10 days of half-hourly UTC demand — long enough for the weekly lag."""
    idx = pd.date_range("2025-03-01", periods=48 * 10, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    hod = idx.hour + idx.minute / 60.0
    daily = 30 + 40 * np.exp(-0.5 * ((hod - 19) / 2.5) ** 2)
    y = pd.Series(daily + rng.normal(0, 1.5, len(idx)), index=idx, name="load_kw")
    weather = pd.DataFrame({
        "temp_c": 8 + rng.normal(0, 2, len(idx)),
        "wind_ms": np.abs(rng.normal(4, 1, len(idx))),
        "irradiance_wm2": np.clip(rng.normal(120, 80, len(idx)), 0, None),
    }, index=idx)
    return y, weather


def _lag_steps(frame):
    return sorted(int(m.group(1)) for c in frame.columns
                  if (m := re.fullmatch(r"load_kw_lag_(\d+)", c)))


# ------------------------------------------------------------- adapt / sampling --
def test_periods_per_day_and_adapt_lags():
    idx = pd.date_range("2025-03-01", periods=100, freq="30min", tz="UTC")
    assert periods_per_day(idx) == 48
    lags, rolling = adapt_lags(n_obs=48 * 4, ppd=48, ppw=336)
    assert 1 in lags and 48 in lags and 336 not in lags     # week lag won't fit 4 days
    assert rolling == (48,)
    # Long series fits the weekly features too.
    lags2, rolling2 = adapt_lags(n_obs=48 * 14, ppd=48, ppw=336)
    assert lags2 == (1, 48, 336) and rolling2 == (48, 336)


# ----------------------------------------------------------- alignment / NA-free --
def test_build_ss_frame_aligned_and_na_free(feeder_series):
    y, weather = feeder_series
    frame = build_ss_frame(y, weather=weather)

    assert frame.columns[0] == "load_kw"
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert not frame.isna().any().any(), "frame must be NA-free"
    # Calendar + weather + lag features all present.
    assert {"hour_sin", "hour_cos", "is_weekend", "is_holiday",
            "temp_c", "wind_ms"} <= set(frame.columns)
    assert _lag_steps(frame), "expected at least one lag column"
    # Warm-up of the longest lag/rolling window is dropped (alignment).
    assert frame.index.min() == y.index[336]


def test_weather_omitted_when_not_aligned(feeder_series):
    """Weather that doesn't overlap the load index is dropped, not back-filled,
    so the frame stays NA-free (the 'join if available' contract)."""
    y, _ = feeder_series
    disjoint = pd.DataFrame(
        {"temp_c": [1.0, 2.0]},
        index=pd.date_range("2030-01-01", periods=2, freq="30min", tz="UTC"),
    )
    frame = build_ss_frame(y, weather=disjoint)
    assert "temp_c" not in frame.columns
    assert not frame.isna().any().any()


def test_topology_features_optional_and_constant(feeder_series):
    y, weather = feeder_series
    frame = build_ss_frame(y, weather=weather, topology={"rating_kva": 500, "n_meters": 42})
    assert {"topo_rating_kva", "topo_n_meters"} <= set(frame.columns)
    assert (frame["topo_rating_kva"] == 500.0).all()
    assert not frame.isna().any().any()


# --------------------------------------------------------------- no leakage ------
def test_no_future_leakage_in_lags(feeder_series):
    """lag_k at time t must equal the raw target at t-k (strictly past)."""
    y, weather = feeder_series
    frame = build_ss_frame(y, weather=weather)
    steps = _lag_steps(frame)
    assert steps  # at least one lag
    for k in steps:
        col = frame[f"load_kw_lag_{k}"]
        expected = y.shift(k).reindex(frame.index)
        pd.testing.assert_series_equal(col, expected, check_names=False)
        # A row's own (present) target never appears in its lag columns.
        assert not np.allclose(col.to_numpy(), frame["load_kw"].to_numpy())


def test_split_is_chronological_and_leak_free(feeder_series):
    y, weather = feeder_series
    frame = build_ss_frame(y, weather=weather)

    train, test = ss_train_test_split(frame, split=0.75)
    assert len(train) and len(test)
    assert train.index.max() < test.index.min()        # strict boundary
    assert train.index.intersection(test.index).empty
    # No test-row lag reaches forward past its own timestamp into "the future".
    for k in _lag_steps(frame):
        ts = test.index[0]
        assert test.loc[ts, f"load_kw_lag_{k}"] == pytest.approx(
            y.shift(k).loc[ts]
        )
    # A date-string boundary works too.
    boundary = frame.index[len(frame) // 2]
    tr2, te2 = ss_train_test_split(frame, split=str(boundary))
    assert tr2.index.max() <= boundary < te2.index.min()


# --------------------------------------------------- GB calendar for the dataset --
def test_calendar_uses_dataset_country_gb():
    idx = pd.date_range("2025-12-24", "2025-12-26 23:00", freq="h", tz="UTC")
    cal = add_calendar(pd.DataFrame(index=idx), country="GB")
    # 25 Dec is a GB bank holiday; 24 Dec is not.
    assert cal.loc["2025-12-25 12:00+00:00", "is_holiday"] == 1.0
    assert cal.loc["2025-12-24 12:00+00:00", "is_holiday"] == 0.0


# ------------------------------------------------- federation: >1 feeder offline --
def test_multiple_feeders_available_for_federation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "OFFLINE", True)

    feeders = pipeline.ss_feeders()
    assert len(feeders) > 1, "federation needs more than one feeder/node"

    frames = list(pipeline.iter_ss_frames())
    assert len(frames) > 1
    ids = [fid for fid, _ in frames]
    assert len(set(ids)) == len(ids)                    # distinct nodes
    for fid, frame in frames:
        assert frame.columns[0] == "load_kw"
        assert not frame.isna().any().any()
        assert len(frame) > 0
        assert _lag_steps(frame)                        # lag features built
