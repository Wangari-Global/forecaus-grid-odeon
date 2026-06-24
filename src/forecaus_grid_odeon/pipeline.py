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
    """Feeder keys to model (>1 => federation is possible).

    Precedence: offline -> committed fixtures; else REAL per-feeder caches under
    ``data/raw/ss/`` (from ``make ingest-ss-real``) if present; else live
    discovery, falling back to the default set.
    """
    from .ingest_ss import ukpn

    if config.OFFLINE:
        keys = ukpn.available_fixture_keys()
    else:
        keys = ukpn.cached_feeder_keys()                       # real data takes priority
        if not keys:
            keys = [ukpn.feeder_key(s, f) for s, f in ukpn.discover_feeders()]
    return keys or [ukpn.feeder_key(s, f) for s, f in ukpn.DEFAULT_FEEDERS]


def using_real_ss_data() -> bool:
    """True iff real LV-feeder caches exist and we are not forced offline."""
    from .ingest_ss import ukpn
    return (not config.OFFLINE) and bool(ukpn.cached_feeder_keys())


def _ss_weather(index) -> Optional[pd.DataFrame]:
    """Weather over the feeder's ACTUAL span (not the national 2024-25 window),
    fetched directly from Open-Meteo so the real April-2026 SS data gets a valid
    join. Returns None offline / on any failure (the frame then builds without it).
    """
    if config.OFFLINE or index is None or len(index) == 0:
        return None
    try:
        import requests

        from .ingest.weather import _ARCHIVE_URL, _COORDS, _HOURLY_VARS
        lat, lon = _COORDS.get(config.SS_REGION, _COORDS["FR"])
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": str(pd.Timestamp(index.min()).date()),
            "end_date": str(pd.Timestamp(index.max()).date()),
            "hourly": ",".join(_HOURLY_VARS), "timezone": "UTC",
        }
        hourly = requests.get(_ARCHIVE_URL, params=params, timeout=90).json().get("hourly", {})
        if not hourly.get("time"):
            return None
        wx = pd.DataFrame(hourly).set_index("time")
        wx.index = pd.to_datetime(wx.index, utc=True)
        return wx.rename(columns={"temperature_2m": "temp_c", "wind_speed_10m": "wind_ms",
                                  "shortwave_radiation": "irradiance_wm2"})
    except Exception:  # noqa: BLE001 - weather is optional; never block the frame
        return None


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
    from .ingest_ss import ukpn

    substation_id, lv_feeder_id = ukpn._key_to_feeder(feeder_id)
    series = ukpn.fetch_feeder(substation_id, lv_feeder_id)[SS_TARGET]

    wx = _ss_weather(series.index) if with_weather else None

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
