"""DARTS TFT tests — skipped if the torch extra isn't installed.

Kept tiny (few epochs, small net, one fold) so it runs in seconds on CPU.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="requires darts[torch]")
pytest.importorskip("darts")

from forecaus_grid_odeon.eval import rolling_origin
from forecaus_grid_odeon.forecast import make_tft


@pytest.fixture
def frame():
    idx = pd.date_range("2025-01-01", periods=96, freq="h", tz="UTC")
    n = len(idx)
    y = 50000 + 5000 * np.sin(np.arange(n) / 24 * 2 * np.pi)
    return pd.DataFrame({
        "load_mw": y,
        "temp_c": 15 + np.cos(np.arange(n) / 24 * 2 * np.pi),
        "hour_sin": np.sin(idx.hour.to_numpy() / 24 * 2 * np.pi),
    }, index=idx)


def test_tft_trains_and_predicts(frame):
    """TFT produces a finite quantile forecast with ordered intervals."""
    model = make_tft(cov_cols=["temp_c", "hour_sin"], n_epochs=3, input_chunk_length=48)
    train, test = frame.iloc[:-24], frame.iloc[-24:]
    pred = model(train, test, "load_mw")
    assert list(pred.columns) == ["yhat", "lower", "upper"]
    assert pred.index.equals(test.index)
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred["lower"] <= pred["yhat"]).all() and (pred["yhat"] <= pred["upper"]).all()


def test_tft_appears_in_backtest_table(frame):
    """The neural model is usable in the rolling-origin harness alongside others."""
    table = rolling_origin(
        frame, {"tft": make_tft(cov_cols=["temp_c", "hour_sin"], n_epochs=3, input_chunk_length=48)},
        horizon=24, step=24, initial=48, target="load_mw",
    )
    assert "tft" in table.index
    assert np.isfinite(table.loc["tft", "MAPE"])
    assert {"MAE", "RMSE", "MAPE", "coverage", "n_folds"} <= set(table.columns)
