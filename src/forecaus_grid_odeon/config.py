"""Central config: data paths, target series, public-data source settings."""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
# Tiny committed sample (1 week, hourly) used as the offline seed for `make ingest`
# and for tests that must run with no network / no API token.
FIXTURES = _ROOT / "tests" / "fixtures"

# Forecast target for the TRL-4 demo (electricity demand; extendable to heat).
TARGET = "electricity_load_mw"
BIDDING_ZONE = "FR"          # ENTSO-E zone / RTE region
FREQ = "h"                   # hourly (pandas offset alias)
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"

# Default ingestion window (UTC). Real downloads use this; the fixture seed
# ignores it and always returns its fixed sample week.
INGEST_START = "2024-01-01"
INGEST_END = "2025-06-01"

# Force the offline sample seed (skip all network/API). Set FORECAUS_OFFLINE=1
# for fully reproducible, network-free ingestion (used by the test-suite).
OFFLINE = os.environ.get("FORECAUS_OFFLINE", "").lower() in ("1", "true", "yes")

# ENTSO-E free API token. Get one for free: register at
# https://transparency.entsoe.eu/ then email transparency@entsoe.eu asking for
# "Restful API access" (see README). Exported as an env var, never committed.
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY")

# Public data sources (all free).
SOURCES = {
    "entsoe": "load, generation, day-ahead price (entsoe-py client, needs free API token)",
    "rte": "RTE eCO2mix open data via ODRE opendatasoft (load/generation/CO2)",
    "weather": "ERA5 reanalysis via Open-Meteo archive API, keyless (temperature/wind/irradiance)",
}
