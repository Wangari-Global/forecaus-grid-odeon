"""Weather ingestion (temperature, wind, irradiance) — demand & renewable drivers.

Uses ERA5 reanalysis served by the Open-Meteo archive API: free, keyless, and
hourly — a practical stand-in for a direct ECMWF/CDS (``cdsapi``) or
Meteo-France pull, which need account credentials. Coordinates default to Paris
(representative for the FR zone). Falls back to the committed offline sample
when there is no network.

Alternative (documented, not default): ECMWF ERA5 via ``cdsapi`` with a free CDS
account, or Meteo-France's open API — both require credentials.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .. import config
from ._io import cached, log

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
# Representative point for the FR bidding zone.
_COORDS = {"FR": (48.8566, 2.3522)}
_HOURLY_VARS = [
    "temperature_2m",         # -> temp_c
    "wind_speed_10m",         # -> wind_ms
    "shortwave_radiation",    # -> irradiance_wm2
]


def fetch_weather(region: str = config.BIDDING_ZONE,
                  start: Optional[str] = None, end: Optional[str] = None,
                  index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Hourly weather features aligned to the load index.

    If ``index`` (typically the ENTSO-E load index) is given, the result is
    reindexed onto it. Cached to data/raw/weather.parquet.
    """
    def download():
        try:
            import requests
        except ImportError:
            return None
        lat, lon = _COORDS.get(region, _COORDS["FR"])
        s = pd.Timestamp(start or config.INGEST_START).date()
        e = pd.Timestamp(end or config.INGEST_END).date()
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": str(s), "end_date": str(e),
            "hourly": ",".join(_HOURLY_VARS),
            "timezone": "UTC",
        }
        log(f"weather: GET {_ARCHIVE_URL} ({lat},{lon})")
        resp = requests.get(_ARCHIVE_URL, params=params, timeout=60)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        if not hourly.get("time"):
            return None
        df = pd.DataFrame(hourly).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.rename(columns={
            "temperature_2m": "temp_c",
            "wind_speed_10m": "wind_ms",
            "shortwave_radiation": "irradiance_wm2",
        })
        out = df.resample("h").mean()
        if index is not None:
            out = out.reindex(index).interpolate(limit_direction="both")
        return out

    df = cached("weather", download)
    if index is not None:
        # Align cached/fixture data onto the requested load index too.
        df = df.reindex(index).interpolate(limit_direction="both")
    return df
