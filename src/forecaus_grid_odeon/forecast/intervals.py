"""Conformal prediction intervals (model-agnostic).

Given a sample of past forecast residuals, the (1-alpha) absolute-residual
quantile is a distribution-free half-width: applying it as ``yhat +/- width``
yields intervals with approximately ``1-alpha`` marginal coverage, regardless of
the underlying model.
"""
import numpy as np


def conformal_width(residuals, alpha: float = 0.1) -> float:
    """Half-width for (1-alpha) coverage = the (1-alpha) quantile of |residuals|."""
    res = np.abs(np.asarray(residuals, float))
    res = res[np.isfinite(res)]
    if res.size == 0:
        return 0.0
    # Finite-sample conformal correction on the quantile level.
    n = res.size
    level = min(1.0, (1.0 - alpha) * (n + 1) / n)
    return float(np.quantile(res, level))


def conformal_intervals(point, residuals, alpha: float = 0.1):
    """Return (lower, upper) arrays for a point forecast given past residuals."""
    point = np.asarray(point, float)
    w = conformal_width(residuals, alpha)
    return point - w, point + w


def make_conformal(base_model, alpha: float = 0.1, calib_frac: float = 0.3):
    """Wrap any point forecaster with split-conformal intervals (model-agnostic).

    A tail slice of the training data is held out for calibration: the base
    model is fit on the rest, its residuals on the calibration slice give the
    conformal half-width, and the final point forecast (base model re-fit on the
    full train window) is widened by it. Marginal coverage ≈ ``1 - alpha``
    regardless of the base model.
    """
    import pandas as pd

    def model(train, test, target):
        n, h = len(train), len(test)
        k = min(max(h, int(n * calib_frac)), n - h)  # calibration slice size
        fit_part, calib = train.iloc[:-k], train.iloc[-k:]
        cal_pred = base_model(fit_part, calib, target)["yhat"].to_numpy()
        resid = calib[target].to_numpy(dtype=float) - cal_pred
        w = conformal_width(resid, alpha)

        point = base_model(train, test, target)["yhat"].to_numpy()
        return pd.DataFrame(
            {"yhat": point, "lower": point - w, "upper": point + w}, index=test.index
        )

    return model
