"""Ingestion tests — run fully offline against the committed fixtures."""
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.ingest import SCHEMA
from forecaus_grid_odeon.ingest._io import load_fixture

EXPECTED_HOURS = 24 * 7


@pytest.mark.parametrize("name", sorted(SCHEMA))
def test_fixture_schema_and_freq(name):
    df = load_fixture(name)

    # Schema: exactly the canonical columns, all numeric, no gaps.
    assert list(df.columns) == SCHEMA[name]
    assert all(pd.api.types.is_float_dtype(df[c]) for c in df.columns)
    assert not df.isna().any().any()

    # Index: UTC, hourly, strictly increasing, one full week.
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) == EXPECTED_HOURS
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert pd.infer_freq(df.index) in ("h", "H")


def test_load_is_cross_consistent_with_rte():
    """RTE consumption should track ENTSO-E load within a few percent."""
    load = load_fixture("entsoe_load")["load_mw"]
    cons = load_fixture("rte_eco2mix")["consumption_mw"]
    rel_err = ((cons - load).abs() / load).mean()
    assert rel_err < 0.05


def test_offline_ingest_writes_cache(tmp_path, monkeypatch):
    """With no API token/network, fetch_* must seed the cache from fixtures."""
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "OFFLINE", True)

    from forecaus_grid_odeon.ingest import entsoe
    df = entsoe.fetch_load()
    assert list(df.columns) == SCHEMA["entsoe_load"]
    cached_path = tmp_path / "entsoe_load.parquet"
    assert cached_path.exists()

    # Second call must hit the cache and return identical data.
    df2 = entsoe.fetch_load()
    pd.testing.assert_frame_equal(df, df2)


def test_ingest_run_offline(tmp_path, monkeypatch):
    """End-to-end orchestrator offline: every source produces a cached parquet."""
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "OFFLINE", True)
    from forecaus_grid_odeon import ingest

    out = ingest.run()
    assert set(out) == set(SCHEMA)
    for name in SCHEMA:
        assert (tmp_path / f"{name}.parquet").exists()
        assert len(out[name]) == EXPECTED_HOURS
    # Weather is aligned onto the load index.
    assert out["weather"].index.equals(out["entsoe_load"].index)
