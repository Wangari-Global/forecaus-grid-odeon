"""Forecast metrics: point accuracy (MAE/RMSE/MAPE) + interval coverage."""
import numpy as np


def mae(y, yhat):  return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))
def rmse(y, yhat): return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mape(y, yhat):
    """Mean absolute percentage error [%], ignoring zero-valued actuals."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); m = y != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100)


def coverage(y, lower, upper):
    """Empirical coverage: fraction of actuals inside [lower, upper]."""
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))))


def summary(y, yhat, lower=None, upper=None) -> dict:
    """One row of metrics for a forecast. Includes coverage when intervals given."""
    out = {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat)}
    if lower is not None and upper is not None:
        out["coverage"] = coverage(y, lower, upper)
    return out
