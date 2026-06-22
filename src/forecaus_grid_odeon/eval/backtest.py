"""Rolling-origin (walk-forward) backtest comparing forecasters.

At each origin the model trains on all data up to the origin and predicts the
next ``horizon`` steps; the origin then advances by ``step``. Per-fold metrics
are averaged into one row per model. This is the harness later used to compare
causal-augmented vs purely-ML vs naive models, including on a structural-break
window.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Optional

import pandas as pd

from .metrics import summary


def rolling_origin(
    df: pd.DataFrame,
    models: Mapping[str, "callable"],
    horizon: int = 24,
    step: Optional[int] = None,
    initial: Optional[int] = None,
    target: Optional[str] = None,
) -> pd.DataFrame:
    """Walk-forward evaluation; return a per-model metric table.

    Columns: MAE, RMSE, MAPE, coverage (when models emit intervals), n_folds.
    A model that errors on a fold is skipped for that fold (with a warning),
    not fatal to the run.
    """
    target = target or df.columns[0]
    n = len(df)
    step = step or horizon
    if initial is None:
        initial = max(2 * horizon, n // 2)
    if initial > n - horizon:
        raise ValueError(
            f"not enough data: need initial+horizon <= {n}, got {initial}+{horizon}"
        )

    origins = range(initial, n - horizon + 1, step)
    records: dict[str, list[dict]] = {name: [] for name in models}
    n_folds = 0
    for o in origins:
        train, test = df.iloc[:o], df.iloc[o:o + horizon]
        n_folds += 1
        y = test[target]
        for name, model in models.items():
            try:
                pred = model(train, test, target)
            except Exception as exc:  # noqa: BLE001 - keep other models/folds alive
                warnings.warn(f"{name} failed at origin {o}: {exc!r}")
                continue
            records[name].append(
                summary(y, pred["yhat"], pred.get("lower"), pred.get("upper"))
            )

    rows = {}
    for name, recs in records.items():
        if recs:
            row = pd.DataFrame(recs).mean()
            row["n_folds"] = len(recs)
            rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    return table


def format_table(table: pd.DataFrame) -> str:
    """Pretty-print a metric table with sensible rounding."""
    disp = table.copy()
    for c in disp.columns:
        disp[c] = disp[c].round(4 if c == "coverage" else 2)
    return disp.to_string()
