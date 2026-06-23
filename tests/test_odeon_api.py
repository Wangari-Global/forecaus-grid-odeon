"""ODEON Energy Data Space adapters: the mock payload parses into Slice-1 tidy
frames behind a flag, and the live path stays clearly stubbed."""
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.ingest_ss import odeon_api, ukpn

SS_ID = "SS_ODEON_001"


@pytest.fixture(scope="module")
def payload():
    return odeon_api.load_mock_payload()


@pytest.fixture(scope="module")
def slice1_reference():
    """A real Slice-1 frame (UKPN fixture) to assert schema parity against."""
    return ukpn.load_fixture(ukpn.available_fixture_keys()[0])


def _assert_slice1_schema(df, reference):
    assert list(df.columns) == list(reference.columns) == ukpn.SS_SCHEMA == ["load_kw"]
    assert pd.api.types.is_float_dtype(df["load_kw"])
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == reference.index.name == "time"
    assert str(df.index.tz) == str(reference.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert not df.isna().any().any()


# ------------------------------------------------ live path stays stubbed -----
def test_live_path_raises_not_implemented(monkeypatch):
    monkeypatch.setattr(config, "ODEON_MOCK", False)        # default: no flag
    for call in (
        lambda: odeon_api.fetch_ss_scada(SS_ID),
        lambda: odeon_api.fetch_lv_smart_meter(SS_ID),
        lambda: odeon_api.fetch_lv_topology(SS_ID),
    ):
        with pytest.raises(NotImplementedError):
            call()


# ------------------------------------- mock feeds fetch_* -> Slice-1 frames ---
def test_scada_mock_matches_slice1_schema(payload, slice1_reference):
    # Feed the mock JSON straight through the adapter.
    df = odeon_api.fetch_ss_scada(SS_ID, source="mock", payload=payload)
    _assert_slice1_schema(df, slice1_reference)
    assert len(df) == len(payload["scada"]["readings"]) > 0
    assert (df["load_kw"] > 0).all()                        # active power -> load


def test_smart_meter_mock_matches_slice1_schema(payload, slice1_reference):
    df = odeon_api.fetch_lv_smart_meter(SS_ID, source="mock", payload=payload)
    _assert_slice1_schema(df, slice1_reference)
    # 15-min Wh import -> kW: first reading energy / 0.25 h / 1000.
    first = payload["lv_smart_meter"]["readings"][0]
    expected_kw = first["active_import_wh"] / (15 / 60) / 1000
    assert df["load_kw"].iloc[0] == pytest.approx(expected_kw)


def test_flag_routes_fetch_to_mock(monkeypatch, slice1_reference):
    """With the flag set, fetch_* (no explicit source) parses the mock fixture."""
    monkeypatch.setattr(config, "ODEON_MOCK", True)
    df = odeon_api.fetch_ss_scada(SS_ID)                     # reads committed fixture
    _assert_slice1_schema(df, slice1_reference)
    assert len(df) > 0


def test_topology_returns_tidy_feeder_frame(payload):
    topo = odeon_api.fetch_lv_topology(SS_ID, source="mock", payload=payload)
    assert topo.index.name == "lv_feeder_id"
    assert {"secondary_substation_id", "n_supply_points", "phases",
            "transformer_rating_kva"} <= set(topo.columns)
    assert len(topo) == len(payload["lv_topology"]["feeders"]) > 1
    assert (topo["transformer_rating_kva"] == 500).all()


def test_ss_id_filter_and_window(payload):
    # Non-matching SS -> empty (but schema-correct) frame.
    empty = odeon_api.fetch_ss_scada("SS_DOES_NOT_EXIST", source="mock", payload=payload)
    assert list(empty.columns) == ["load_kw"] and len(empty) == 0

    # Time window narrows the series.
    full = odeon_api.fetch_ss_scada(SS_ID, source="mock", payload=payload)
    mid = full.index[len(full) // 2]
    sliced = odeon_api.fetch_ss_scada(SS_ID, start=mid, source="mock", payload=payload)
    assert sliced.index.min() >= mid
    assert len(sliced) < len(full)


def test_mock_load_kw_feeds_feature_pipeline(payload):
    """The ODEON-shaped load_kw frame is build()-compatible like Slice-1 data."""
    from forecaus_grid_odeon.features import build_ss_frame

    df = odeon_api.fetch_lv_smart_meter(SS_ID, source="mock", payload=payload)
    frame = build_ss_frame(df["load_kw"], lags=(1, 4), rolling_windows=(4,))
    assert frame.columns[0] == "load_kw"
    assert not frame.isna().any().any() and len(frame) > 0
