"""Secondary-substation (SS) / LV-feeder data adapters.

Two layers live here:

* :mod:`.ukpn` - a **working** ingester for a real public LV/secondary-substation
  dataset (UK Power Networks "Smart Meter Consumption - LV Feeder", keyless
  opendatasoft API). It pulls per-feeder half-hourly demand, normalises to a
  tidy UTC ``load_kw`` frame, and caches one parquet per feeder. This is the
  development data source for many real LV feeders.
* :mod:`.odeon_api` - the ODEON Energy Data Space adapters. The live spec is not
  released pre-submission (help-desk Q1/Q4) so the live path raises
  NotImplementedError, but a committed MOCK payload + a documented interface
  contract let the SS-level inputs (SCADA, LV smart meter, LV topology) parse
  into the SAME tidy ``load_kw`` frames as the UKPN slice (set
  ``FORECAUS_ODEON_MOCK=1`` / ``source="mock"``).
"""
from . import odeon_api, ukpn
from .ukpn import SS_SCHEMA, fetch_feeder, ingest_ss

__all__ = ["ukpn", "odeon_api", "ingest_ss", "fetch_feeder", "SS_SCHEMA"]
