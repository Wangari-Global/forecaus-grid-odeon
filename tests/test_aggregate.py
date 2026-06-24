"""SS aggregation: per-feeder load_kw SUMMED to one SS-total per secondary
substation, on the canonical contract. Runs offline on the committed synthetic
feeder fixtures + the SS-aggregate fixtures."""
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.ingest_ss import aggregate, ukpn


def test_substation_total_is_sum_of_two_feeder_fixtures():
    # Two committed feeder fixtures on the SAME substation.
    f1 = ukpn.load_fixture("EPN0012345__feeder-1")
    f2 = ukpn.load_fixture("EPN0012345__feeder-2")

    totals = aggregate.substation_totals(
        {"EPN0012345__feeder-1": f1, "EPN0012345__feeder-2": f2}, min_feeders=2,
    )

    assert set(totals) == {"EPN0012345"}
    t = totals["EPN0012345"]
    # Schema / freq contract: single load_kw float col, UTC index named time, 30min.
    assert list(t.columns) == ["load_kw"]
    assert t["load_kw"].dtype == "float64"
    assert str(t.index.tz) == "UTC" and t.index.name == "time"
    steps = t.index.to_series().diff().dropna().unique()
    assert len(steps) == 1 and steps[0] == pd.Timedelta("30min")
    # It is exactly the elementwise SUM of the two feeders.
    expected = (f1["load_kw"] + f2["load_kw"]).astype("float64")
    pd.testing.assert_series_equal(t["load_kw"], expected, check_names=False)


def test_min_feeders_gate_drops_undercovered_timesteps():
    # If one feeder is missing the last 5 timesteps, min_feeders=2 drops them.
    f1 = ukpn.load_fixture("EPN0012345__feeder-1")
    f2 = ukpn.load_fixture("EPN0012345__feeder-2").iloc[:-5]

    full = aggregate.substation_totals(
        {"a__feeder-1": f1.rename_axis("time"), "a__feeder-2": f2}, min_feeders=1)["a"]
    gated = aggregate.substation_totals(
        {"a__feeder-1": f1, "a__feeder-2": f2}, min_feeders=2)["a"]

    assert len(full) == len(f1)                 # union kept with min_feeders=1
    assert len(gated) == len(f1) - 5            # last 5 (only 1 feeder) dropped


def test_committed_agg_fixtures_exist_and_match_recomputation(monkeypatch, tmp_path):
    # >1 committed SS-aggregate fixture, each == the recomputed sum of its feeders.
    keys = aggregate.available_agg_fixture_keys()
    assert len(keys) > 1                        # multi-substation aggregate fixture

    monkeypatch.setattr(config, "OFFLINE", True)
    monkeypatch.setattr(config, "RAW", tmp_path)               # no real cache -> use fixtures
    recomputed = aggregate.substation_totals(min_feeders=1)    # from feeder fixtures

    assert set(keys) == set(recomputed)
    for sub in keys:
        committed = aggregate.load_agg_fixture(sub)
        pd.testing.assert_frame_equal(committed, recomputed[sub])
