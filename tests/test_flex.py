import pandas as pd
from forecaus_grid_odeon.flex import flexibility_need


def test_flex_down_only_above_limit():
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    f = pd.Series([100, 700, 650, 200], index=idx, dtype=float)
    out = flexibility_need(f, ss_limit=600.0)
    assert list(out["flex_down"]) == [0.0, 100.0, 50.0, 0.0]
    assert (out["flex_up"] == 0.0).all()


def test_flex_up_for_reverse_power():
    idx = pd.date_range("2026-01-01", periods=2, freq="h")
    f = pd.Series([-50.0, 10.0], index=idx)
    out = flexibility_need(f, ss_limit=600.0, lower_limit=0.0)
    assert out["flex_up"].iloc[0] == 50.0
    assert out["flex_up"].iloc[1] == 0.0
