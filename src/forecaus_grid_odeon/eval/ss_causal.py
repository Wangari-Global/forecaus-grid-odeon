"""SS-level causal layer: temperature -> load effect + the causal-vs-correlational
structural-break experiment, reusing :mod:`causal` and :mod:`eval.causal_break`.

Two deliverables, mirroring the national headline but at LV/secondary-substation
scale (``load_kw``, half-hourly):

1. **Effect estimate + refuters.** The SS DAG (:func:`causal.build_ss_dag`) says
   ``temperature -> load`` is confounded by the diurnal calendar; DoWhy adjusts
   for the calendar backdoor and estimates the effect, then the placebo /
   random-common-cause / data-subset refuters stress it. We report the effect,
   its CI and the pass/fail table **honestly, including a negative sign**
   (heating: colder => more load is a negative coefficient).

2. **Causal-augmented vs correlational across a regime change.** A purely
   correlational model also uses a *downstream* PV/EV-interaction proxy that
   tracks load in normal operation; the causal-augmented model uses only the
   DAG parents of load (calendar + temperature). When PV/EV penetration shifts
   (the regime change), the proxy decouples and the correlational model
   degrades, while the causal model — keyed on invariant causes — does not.

No documented sub-feeder break exists in the tiny offline UKPN sample, so the
shift is **clearly labelled as injected** (the PV/EV-penetration mechanism it
mimics is, however, a real and documented LV-network trend). The construction is
seeded and reproducible; the load process itself is invariant across regimes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..causal import SS_DOWNSTREAM, SS_OUTCOME, SS_TREATMENT, build_ss_dag, run_causal
from ..features.calendar import add_calendar
from ..forecast import make_conformal, make_linear, make_seasonal_naive
from .causal_break import evaluate, make_break_figure

TARGET = SS_OUTCOME                                    # "load_kw"
SS_CAUSAL_FEATURES = ["hour_sin", "hour_cos", "is_weekend", SS_TREATMENT]
PERIODS_PER_DAY = 48                                   # half-hourly
TRUE_TEMP_COEF = -1.5                                  # kW per degC (heating: cold -> more load)


def generate_ss_break_data(
    normal_days: int = 30, break_days: int = 7, seed: int = 6,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (frame, break_start) for one feeder. INJECTED shift (clearly
    labelled): only the load<->pv_ev_proxy mechanism changes at ``break_start``;
    the load process (calendar + temperature) is invariant.

    Confounding is built linearly in the calendar encodings the DAG adjusts for
    (``hour_sin``/``hour_cos``), so the temperature -> load effect is cleanly
    identifiable (the recovered coefficient ~ :data:`TRUE_TEMP_COEF`).
    """
    rng = np.random.default_rng(seed)
    n = (normal_days + break_days) * PERIODS_PER_DAY
    idx = pd.date_range("2025-01-06", periods=n, freq="30min", tz="UTC")

    # Calendar encodings (GB local time) drive both temperature and load -> the
    # diurnal confounder. Read them back from add_calendar so adjustment is exact.
    cal = add_calendar(pd.DataFrame(index=idx), country="GB")
    hs, hc, we = cal["hour_sin"].to_numpy(), cal["hour_cos"].to_numpy(), cal["is_weekend"].to_numpy()

    # Temperature: diurnal (linear in hour encodings) + slow seasonal drift + noise.
    drift = 4.0 * np.sin(np.arange(n) / (PERIODS_PER_DAY * 20) * 2 * np.pi)
    temp = 10.0 - 5.0 * hc + 1.5 * hs + drift + rng.normal(0, 1.2, n)

    # INVARIANT demand process: calendar rhythm + heating response + noise.
    daily = 18.0 * hs - 22.0 * hc                      # smooth daily shape (kW)
    load = (45.0 + daily - 6.0 * we
            + TRUE_TEMP_COEF * (temp - temp.mean()) + rng.normal(0, 1.5, n))
    load = np.clip(load, 1.0, None)

    # pv_ev_proxy: downstream effect of load. Normal: tracks load tightly.
    pv_ev = load + rng.normal(0, 1.0, n)
    # BREAK: PV/EV penetration shift decouples the proxy from load.
    bpos = normal_days * PERIODS_PER_DAY
    decoupled = load.mean() + 25.0 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 1.0, n)
    pv_ev[bpos:] = decoupled[bpos:]

    frame = pd.DataFrame({TARGET: load, SS_TREATMENT: temp, SS_DOWNSTREAM: pv_ev}, index=idx)
    frame = pd.concat([frame, cal], axis=1)
    return frame, idx[bpos]


def build_ss_models(alpha: float = 0.1) -> dict:
    """Three contenders, all emitting intervals (mirrors the national panel)."""
    all_features = SS_CAUSAL_FEATURES + [SS_DOWNSTREAM]
    return {
        "naive": make_conformal(make_seasonal_naive(season=PERIODS_PER_DAY), alpha=alpha),
        "correlational_ml": make_linear(all_features, alpha=alpha),       # uses pv_ev_proxy
        "causal_augmented": make_linear(SS_CAUSAL_FEATURES, alpha=alpha),  # causes only
    }


def run_ss_causal_effect(
    frame: Optional[pd.DataFrame] = None, *, seed: int = 6, num_simulations: int = 50,
) -> tuple[dict, pd.DataFrame]:
    """Estimate temp -> load with the SS DAG + run the three refuters.

    Returns ``(result, refute_table)`` exactly like :func:`causal.run_causal`.
    """
    if frame is None:
        frame, _ = generate_ss_break_data(seed=seed)
    graph = build_ss_dag()
    return run_causal(
        frame, treatment=SS_TREATMENT, outcome=SS_OUTCOME,
        random_seed=seed, num_simulations=num_simulations, graph=graph,
    )


def run_ss_break_experiment(seed: int = 6, horizon: int = PERIODS_PER_DAY) -> dict:
    """Generate -> evaluate the causal-vs-correlational break experiment."""
    frame, break_start = generate_ss_break_data(seed=seed)
    models = build_ss_models()
    result = evaluate(frame, models, break_start, horizon=horizon,
                      initial=14 * PERIODS_PER_DAY, target=TARGET)
    result["frame"] = frame
    result["break_start"] = break_start
    return result
