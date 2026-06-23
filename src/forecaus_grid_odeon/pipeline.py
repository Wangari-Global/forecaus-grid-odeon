"""Glue: ingest (cached/offline) -> feature frame ready for forecasting.

Two entry points:
  * :func:`load_frame` - the national (ENTSO-E proxy) modelling frame.
  * :func:`load_ss_frame` / :func:`iter_ss_frames` - one modelling frame per
    LV feeder (UKPN), the per-node view the federated setup trains on.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping, Optional

import pandas as pd

from . import config
from .features import build, build_ss_frame


def load_frame() -> pd.DataFrame:
    """Load cached/offline data and build the modelling frame (target first column).

    Lag/rolling depth adapts to series length so the pipeline also works on the
    tiny offline sample (which is too short for a 168h lag).
    """
    from .ingest import entsoe, weather

    load = entsoe.fetch_load()["load_mw"]
    wx = weather.fetch_weather(index=load.index)
    price = entsoe.fetch_day_ahead_price()

    if len(load) > 24 * 14:                 # >2 weeks: use weekly features
        lags, rolling = (1, 24, 168), (24, 168)
    else:                                   # short sample: daily only
        lags, rolling = (1, 24), (24,)
    frame = build(load, wx, price, lags=lags, rolling_windows=rolling)
    if frame.empty:
        raise RuntimeError("feature frame is empty — ingest produced too little data")
    return frame


TARGET = "load_mw"
SS_TARGET = config.SS_TARGET


# --------------------------------------------------------- SS / federated layer --
def ss_feeders() -> list[str]:
    """Feeder keys available for ingestion (>1 => federation is possible).

    Offline: the committed sample feeders. Online: live-discovered feeders,
    falling back to the default set.
    """
    from .ingest_ss import ukpn

    if config.OFFLINE:
        keys = ukpn.available_fixture_keys()
    else:
        keys = [ukpn.feeder_key(s, f) for s, f in ukpn.discover_feeders()]
    return keys or [ukpn.feeder_key(s, f) for s, f in ukpn.DEFAULT_FEEDERS]


def load_ss_frame(
    feeder_id: str,
    *,
    with_weather: bool = True,
    topology: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Build one feeder's modelling frame (target ``load_kw`` first column).

    Loads the feeder's demand (cached/offline/online), joins weather when it is
    fully available over the load index, adds optional ``topology`` features,
    and engineers length-adapted calendar + lag features. Aligned and NA-free.
    """
    from .ingest import weather as weather_mod
    from .ingest_ss import ukpn

    substation_id, lv_feeder_id = ukpn._key_to_feeder(feeder_id)
    series = ukpn.fetch_feeder(substation_id, lv_feeder_id)[SS_TARGET]

    wx = None
    if with_weather:
        try:
            wx = weather_mod.fetch_weather(region=config.SS_REGION, index=series.index)
        except Exception:  # noqa: BLE001 - weather is optional; never block the frame
            wx = None

    frame = build_ss_frame(
        series, weather=wx, topology=topology, country=config.SS_COUNTRY,
    )
    if frame.empty:
        raise RuntimeError(f"SS frame for {feeder_id} is empty — series too short")
    return frame


def iter_ss_frames(
    feeders: Optional[Iterable[str]] = None,
    *,
    with_weather: bool = True,
    topology: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(feeder_id, frame)`` per feeder - the federated node iterator.

    ``topology`` may map ``feeder_id -> {attr: value}`` to attach per-feeder
    static features. Feeders whose frame can't be built are skipped (logged),
    so one bad node never breaks the federation sweep.
    """
    feeders = list(feeders) if feeders is not None else ss_feeders()
    topo_map = topology or {}
    for fid in feeders:
        try:
            yield fid, load_ss_frame(
                fid, with_weather=with_weather, topology=topo_map.get(fid),
            )
        except Exception as exc:  # noqa: BLE001 - skip unbuildable node, keep going
            print(f"[pipeline] skipping feeder {fid}: {exc!r}")
