"""Generate the committed offline SS sample under tests/fixtures/ss/.

The live UKPN "Smart Meter Consumption - LV Feeder" records are (currently)
gated behind portal login, so this script writes a small, deterministic,
schema-accurate stand-in: a handful of LV feeders across a few substations, a
few days of half-hourly ``load_kw`` with realistic LV demand shapes. It matches
the exact contract the real :func:`forecaus_grid_odeon.ingest_ss.ukpn.fetch_feeder`
produces (UTC half-hourly index named ``time``, single ``load_kw`` column), so
tests exercise the same code path offline.

Run from the repo root:  python scripts/make_ss_fixtures.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecaus_grid_odeon.ingest_ss import ukpn

# A few days, half-hourly, UTC - small enough to commit.
START = "2025-03-03"        # a Monday
DAYS = 4
PERIODS = DAYS * 48


def _feeder_series(seed: int, base_kw: float, peak_kw: float) -> pd.Series:
    """A plausible LV-feeder demand shape: morning + evening peaks + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=PERIODS, freq=ukpn.FREQ, tz="UTC", name=ukpn.INDEX_NAME)
    hod = idx.hour + idx.minute / 60.0
    morning = np.exp(-0.5 * ((hod - 7.5) / 1.6) ** 2)
    evening = np.exp(-0.5 * ((hod - 19.0) / 2.0) ** 2)
    shape = base_kw + peak_kw * (0.6 * morning + 1.0 * evening)
    weekend = (idx.dayofweek >= 5).astype(float)
    shape = shape * (1.0 - 0.15 * weekend)              # lighter weekend demand
    noise = rng.normal(0, 0.03 * peak_kw, PERIODS)
    return pd.Series(np.clip(shape + noise, 0.0, None), index=idx, name="load_kw")


# (substation, feeder, base_kw, peak_kw) - mirrors ukpn.DEFAULT_FEEDERS.
SPECS = [
    ("EPN0012345", 1, 18.0, 55.0),
    ("EPN0012345", 2, 12.0, 40.0),
    ("LPN0004567", 1, 30.0, 90.0),
    ("LPN0004567", 3, 22.0, 70.0),
    ("SPN0008901", 1, 15.0, 48.0),
]


def main() -> None:
    out_dir = ukpn._fixture_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    feeders = {}
    for i, (sub, fdr, base, peak) in enumerate(SPECS):
        s = _feeder_series(seed=100 + i, base_kw=base, peak_kw=peak)
        key = ukpn.feeder_key(sub, fdr)
        path = out_dir / f"{key}.parquet"
        s.to_frame().to_parquet(path)
        feeders[key] = s.to_frame()
        print(f"wrote {path}  ({len(s)} rows, {s.min():.1f}-{s.max():.1f} kW)")

    # SS-aggregate fixtures (SYNTHETIC): SUM the synthetic feeder fixtures per
    # substation -> one SS-total per substation, for offline aggregation tests/CI.
    from forecaus_grid_odeon.ingest_ss import aggregate

    agg_dir = aggregate._agg_fixture_dir()
    agg_dir.mkdir(parents=True, exist_ok=True)
    for sub, df in aggregate.substation_totals(feeders, min_feeders=1).items():
        n = sum(1 for k in feeders if aggregate.substation_of(k) == sub)
        df.to_parquet(agg_dir / f"{sub}.parquet")
        print(f"wrote {agg_dir / f'{sub}.parquet'}  (SS-total of {n} feeder(s), "
              f"{df['load_kw'].min():.1f}-{df['load_kw'].max():.1f} kW)")


if __name__ == "__main__":
    main()
