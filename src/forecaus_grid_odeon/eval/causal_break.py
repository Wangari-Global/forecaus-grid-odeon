"""Headline experiment: causal-augmented vs purely-correlational vs naive,
across a structural break / regime change.

The scientific point (invariant causal prediction): a purely-correlational model
will exploit any feature that predicts the target in-distribution — including a
**downstream effect** of the target whose relationship is *not* invariant. A
causal-augmented model selects only the outcome's **direct causes** (DAG
parents), whose effect on the outcome is invariant under interventions
elsewhere. When a regime change breaks the spurious mechanism, the correlational
model degrades sharply while the causal model (and a naive seasonal baseline)
do not.

Controlled, seeded experiment (the build plan permits injecting a known shift):
  * load_mw is driven by an INVARIANT process: calendar + temperature + noise.
  * gas_proxy is a DOWNSTREAM EFFECT of demand (merit-order: more demand -> more
    gas). In the NORMAL regime gas_proxy ≈ load; in the BREAK regime the
    mechanism decouples (e.g. fuel switch / plants offline), so gas_proxy no
    longer tracks load.
  * The correlational model uses {calendar, temp, gas_proxy}; it leans on the
    near-perfect gas_proxy. The causal model uses only the DAG parents of load
    {calendar, temp}. The load process itself is unchanged across regimes, so
    the causal model and naive baseline are unaffected by the break.

On real ENTSO-E data one would instead locate a real regime change (e.g. the
2021–22 energy-price crisis); here we inject a known shift for a clean,
reproducible demonstration of the mechanism.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from ..features.calendar import add_calendar
from ..forecast import make_conformal, make_linear, make_seasonal_naive
from .metrics import summary

TARGET = "load_mw"
CAUSAL_FEATURES = ["hour_sin", "hour_cos", "is_weekend", "temp_c"]
SPURIOUS_FEATURE = "gas_proxy"


def break_dag() -> nx.DiGraph:
    """DAG for the experiment: calendar & temp CAUSE load; load CAUSES gas_proxy.

    ``causal_parents(g, 'load_mw')`` therefore yields the causal feature set and
    excludes ``gas_proxy`` (a child/effect, not a cause)."""
    g = nx.DiGraph()
    for c in ["hour_sin", "hour_cos", "is_weekend"]:
        g.add_edge(c, TARGET)
    g.add_edge("temp_c", TARGET)
    g.add_edge(TARGET, SPURIOUS_FEATURE)   # demand -> gas (effect, not cause)
    return g


def generate_break_data(normal_days: int = 45, break_days: int = 7,
                        seed: int = 6) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (frame, break_start). The load process is invariant; only the
    gas_proxy<->load mechanism changes at ``break_start``."""
    rng = np.random.default_rng(seed)
    n = (normal_days + break_days) * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    h = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    weekend = (dow >= 5).astype(float)

    # Temperature: diurnal + slow seasonal drift + noise (varies day to day).
    temp = (12 + 6 * np.sin((h - 15) / 24 * 2 * np.pi)
            + 4 * np.sin(np.arange(n) / (24 * 20) * 2 * np.pi) + rng.normal(0, 1.2, n))

    # INVARIANT demand process: calendar rhythm + cooling/heating + noise.
    daily = 7000 * np.sin((h - 8) / 24 * 2 * np.pi) + 4000 * np.sin((h - 18) / 12 * np.pi)
    load = 48000 + daily - 5000 * weekend + 500 * (temp - temp.mean()) + rng.normal(0, 600, n)

    # gas_proxy: downstream effect of demand. Normal: tracks load tightly.
    gas = load + rng.normal(0, 150, n)
    # BREAK: decouple gas_proxy from load (regime change in the spurious mechanism).
    break_start_pos = normal_days * 24
    decoupled = load.mean() + 4000 * np.sin(np.arange(n) / 11) + rng.normal(0, 150, n)
    gas[break_start_pos:] = decoupled[break_start_pos:]

    frame = pd.DataFrame({TARGET: load, "temp_c": temp, SPURIOUS_FEATURE: gas}, index=idx)
    frame = add_calendar(frame)  # adds hour_sin/cos, is_weekend, is_holiday, ...
    return frame, idx[break_start_pos]


def build_models(alpha: float = 0.1) -> dict:
    """The three contenders, all emitting prediction intervals."""
    all_features = CAUSAL_FEATURES + [SPURIOUS_FEATURE]
    return {
        "naive": make_conformal(make_seasonal_naive(season=24), alpha=alpha),
        "correlational_ml": make_linear(all_features, alpha=alpha),     # uses gas_proxy
        "causal_augmented": make_linear(CAUSAL_FEATURES, alpha=alpha),  # causes only
    }


def evaluate(frame: pd.DataFrame, models: dict, break_start: pd.Timestamp,
             horizon: int = 24, initial: Optional[int] = None,
             target: str = TARGET) -> dict:
    """Rolling-origin walk-forward across the whole series, tagging each fold
    'normal' or 'break' by its test-window start. Returns per-fold metrics, the
    headline table, and stitched break-window predictions for plotting."""
    n = len(frame)
    initial = initial or (21 * 24)
    origins = range(initial, n - horizon + 1, horizon)

    fold_rows, stitched = [], {name: [] for name in models}
    actual_break = []
    for o in origins:
        train, test = frame.iloc[:o], frame.iloc[o:o + horizon]
        regime = "break" if test.index[0] >= break_start else "normal"
        y = test[target]
        if regime == "break":
            actual_break.append(y)
        for name, model in models.items():
            pred = model(train, test, target)
            row = summary(y, pred["yhat"], pred.get("lower"), pred.get("upper"))
            row.update(model=name, regime=regime, origin=test.index[0])
            fold_rows.append(row)
            if regime == "break":
                stitched[name].append(pred)

    fold_df = pd.DataFrame(fold_rows)
    table = _headline_table(fold_df)
    preds = {name: pd.concat(parts) for name, parts in stitched.items() if parts}
    actual = pd.concat(actual_break) if actual_break else pd.Series(dtype=float)
    return {"fold_df": fold_df, "table": table, "break_preds": preds, "break_actual": actual}


def _headline_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model MAE/MAPE/coverage for overall, normal and break windows, plus
    the break/normal MAPE degradation ratio (the headline number)."""
    metrics = ["MAE", "MAPE", "coverage"]
    overall = fold_df.groupby("model")[metrics].mean()
    by_regime = fold_df.groupby(["model", "regime"])[metrics].mean().unstack("regime")

    out = pd.DataFrame(index=overall.index)
    for m in metrics:
        out[f"{m}_overall"] = overall[m]
        for reg in ("normal", "break"):
            if (m, reg) in by_regime.columns:
                out[f"{m}_{reg}"] = by_regime[(m, reg)]
    out["MAPE_degradation"] = out["MAPE_break"] / out["MAPE_normal"]
    return out.sort_values("MAPE_break")


def make_break_figure(result: dict, break_start: pd.Timestamp, path: str,
                      *, ylabel: str = "load [MW]") -> str:
    """Headline whitepaper figure: (left) MAPE normal vs break per model,
    (right) the break window — actual load vs each model's forecast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, preds, actual = result["table"], result["break_preds"], result["break_actual"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = list(table.index)
    x = np.arange(len(models))
    w = 0.38
    ax1.bar(x - w / 2, table["MAPE_normal"], w, label="normal", color="#4c9f70")
    ax1.bar(x + w / 2, table["MAPE_break"], w, label="break window", color="#c0504d")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=15)
    ax1.set_ylabel("MAPE [%]"); ax1.set_title("Accuracy: normal vs structural-break window")
    ax1.legend()
    for i, m in enumerate(models):
        ax1.text(i + w / 2, table["MAPE_break"].iloc[i], f"{table['MAPE_break'].iloc[i]:.1f}",
                 ha="center", va="bottom", fontsize=8)

    colors = {"naive": "#888", "correlational_ml": "#c0504d", "causal_augmented": "#1f6fb4"}
    ax2.plot(actual.index, actual.to_numpy(), color="k", lw=2, label="actual load")
    for name, df in preds.items():
        ax2.plot(df.index, df["yhat"].to_numpy(), lw=1.4, alpha=0.9,
                 color=colors.get(name), label=name)
    ax2.set_title("Break window: forecasts vs actual")
    ax2.set_ylabel(ylabel); ax2.tick_params(axis="x", rotation=30)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Causal-augmented vs correlational forecasting across a regime change", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_break_experiment(seed: int = 6, horizon: int = 24) -> dict:
    """End-to-end: generate -> evaluate -> table. (Figure saved separately.)"""
    frame, break_start = generate_break_data(seed=seed)
    models = build_models()
    result = evaluate(frame, models, break_start, horizon=horizon)
    result["frame"] = frame
    result["break_start"] = break_start
    return result
