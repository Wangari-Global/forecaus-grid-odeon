"""SS / LV-feeder ingestion tests - run fully offline against committed samples.

Covers the UK Power Networks LV-feeder adapter: fixture schema/freq/feeder
count, the Wh->kW conversion, and offline ingestion writing one parquet per
feeder with no network access.
"""
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.ingest_ss import ukpn

PERIODS = 4 * 48                       # 4 days, half-hourly
EXPECTED_FEEDERS = 5
FIXTURE_KEYS = ukpn.available_fixture_keys()


def test_expected_feeder_count():
    """A handful of feeders across several substations are committed."""
    assert len(FIXTURE_KEYS) == EXPECTED_FEEDERS
    assert len(ukpn.DEFAULT_FEEDERS) == EXPECTED_FEEDERS
    # The default feeder set and the committed samples line up exactly.
    assert {ukpn.feeder_key(s, f) for s, f in ukpn.DEFAULT_FEEDERS} == set(FIXTURE_KEYS)
    # Several distinct substations (not all feeders on one SS).
    substations = {k.split("__")[0] for k in FIXTURE_KEYS}
    assert len(substations) >= 2


@pytest.mark.parametrize("key", FIXTURE_KEYS)
def test_fixture_schema_and_freq(key):
    df = ukpn.load_fixture(key)

    # Schema: exactly load_kw, float, non-negative, no gaps.
    assert list(df.columns) == ukpn.SS_SCHEMA == ["load_kw"]
    assert pd.api.types.is_float_dtype(df["load_kw"])
    assert not df.isna().any().any()
    assert (df["load_kw"] >= 0).all()

    # Index: UTC, half-hourly, strictly increasing, a few full days.
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) == PERIODS
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all()
    assert pd.infer_freq(df.index) in ("30min", "30T")


def test_feeder_key_is_filename_safe_and_stable():
    key = ukpn.feeder_key("EPN/0012 345", 7)
    assert key == "EPN-0012-345__feeder-7"
    assert ukpn._key_to_feeder("EPN0012345__feeder-2") == ("EPN0012345", 2)


def test_wh_to_kw_conversion():
    """500 Wh in a half-hour == 1 kW average power (the documented factor)."""
    assert 500.0 / ukpn.WH_PER_HH_TO_KW == 1.0
    assert 1_000.0 / ukpn.WH_PER_HH_TO_KW == 2.0


def test_offline_ingest_writes_per_feeder_parquet(tmp_path, monkeypatch):
    """FORECAUS_OFFLINE: ingest writes one parquet per feeder, no network."""
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "OFFLINE", True)
    # Guarantee no network is touched even if a path regressed.
    import forecaus_grid_odeon.ingest_ss.ukpn as ukpn_mod
    monkeypatch.setattr(
        ukpn_mod, "_download_feeder",
        lambda *a, **k: pytest.fail("network used in offline mode"),
    )

    out = ukpn.ingest_ss()

    assert set(out) == set(FIXTURE_KEYS)
    assert len(out) == EXPECTED_FEEDERS
    for key, df in out.items():
        path = tmp_path / "ss" / f"{key}.parquet"
        assert path.exists()
        assert list(df.columns) == ["load_kw"]
        assert len(df) == PERIODS

    # Second run hits the per-feeder cache and returns identical data.
    out2 = ukpn.ingest_ss()
    for key in out:
        pd.testing.assert_frame_equal(out[key], out2[key])


def test_fetch_feeder_offline_matches_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "OFFLINE", True)
    sub, fdr = ukpn.DEFAULT_FEEDERS[0]
    df = ukpn.fetch_feeder(sub, fdr)
    pd.testing.assert_frame_equal(df, ukpn.load_fixture(ukpn.feeder_key(sub, fdr)))
