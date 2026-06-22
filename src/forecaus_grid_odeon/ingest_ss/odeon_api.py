"""ODEON Energy Data Space adapters — secondary-substation level (STUB).

Challenge 4 inputs (per Technical Guidelines / Annex 1):
  * SCADA at the MV/LV transformer winding  (power, voltage, current; 1-5 min)
  * LV smart-meter import/export at supply points            (15 min)
  * LV grid topology                                         (all pilots)

All functions raise NotImplementedError until the ODEON API spec is available.
Design intent: return tidy pandas frames indexed by UTC timestamp so they drop
straight into forecaus_grid_odeon.features.build / .pipeline.
"""
from __future__ import annotations

_PENDING = (
    "ODEON Energy Data Space API spec not released pre-submission "
    "(help-desk Q1/Q4). Use forecaus_grid_odeon.ingest (public proxy) for dev."
)


def fetch_ss_scada(ss_id: str, start, end):
    """SCADA at the MV/LV winding for one secondary substation (1-5 min)."""
    raise NotImplementedError(_PENDING)


def fetch_lv_smart_meter(ss_id: str, start, end):
    """LV supply-point smart-meter import/export for one SS (15 min)."""
    raise NotImplementedError(_PENDING)


def fetch_lv_topology(ss_id: str):
    """LV grid topology for one SS (feeder/supply-point graph)."""
    raise NotImplementedError(_PENDING)
