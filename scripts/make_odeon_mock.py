"""Generate the committed MOCK ODEON Energy Data Space payload.

Writes ``tests/fixtures/odeon/odeon_ss_mock.json`` — a small, deterministic JSON
fixture mimicking the three Challenge-4 SS-level inputs (SCADA at the MV/LV
winding, LV smart-meter import/export, LV grid topology). It is the concrete
**interface contract** the :mod:`forecaus_grid_odeon.ingest_ss.odeon_api`
adapters parse, so the pipeline can run against "ODEON-shaped" data with no real
access. Field names/units here are this project's documented assumption, to be
reconciled with the real spec when it lands (help-desk Q1/Q4).

Run from the repo root:  python scripts/make_odeon_mock.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

SS_ID = "SS_ODEON_001"
START = datetime(2025, 3, 1, 0, 0, tzinfo=timezone.utc)
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "odeon" / "odeon_ss_mock.json"


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _daily_kw(t: datetime, base: float, peak: float) -> float:
    """A plausible LV evening-peak demand shape in kW."""
    h = t.hour + t.minute / 60.0
    morning = math.exp(-0.5 * ((h - 8) / 1.5) ** 2)
    evening = math.exp(-0.5 * ((h - 19) / 2.5) ** 2)
    return round(base + peak * (0.5 * morning + evening), 3)


def build() -> dict:
    # SCADA: 5-minute winding telemetry for 3 hours around the evening peak.
    scada = []
    for i in range(36):                                   # 36 x 5 min = 3 h
        t = START + timedelta(hours=18) + timedelta(minutes=5 * i)
        kw = _daily_kw(t, base=40.0, peak=45.0)
        scada.append({
            "timestamp": _iso(t),
            "active_power_kw": kw,
            "voltage_v": round(415.0 - 0.05 * kw, 2),
            "current_a": round(kw * 1000 / (math.sqrt(3) * 415.0), 2),
        })

    # LV smart meter: 15-minute aggregated import/export energy (Wh) over a day.
    meter = []
    for i in range(96):                                   # 96 x 15 min = 24 h
        t = START + timedelta(minutes=15 * i)
        kw = _daily_kw(t, base=38.0, peak=42.0)
        import_wh = round(kw * 0.25 * 1000)               # kW over 0.25 h -> Wh
        export_wh = 0                                     # no PV export in this sample
        meter.append({
            "timestamp": _iso(t),
            "active_import_wh": import_wh,
            "active_export_wh": export_wh,
        })

    topology = {
        "secondary_substation_id": SS_ID,
        "transformer_rating_kva": 500,
        "feeders": [
            {"lv_feeder_id": 1, "n_supply_points": 28, "phases": 3},
            {"lv_feeder_id": 2, "n_supply_points": 14, "phases": 3},
            {"lv_feeder_id": 3, "n_supply_points": 9, "phases": 1},
        ],
    }

    return {
        "dataspace": "ODEON Energy Data Space (MOCK)",
        "schema_version": "mock-0.1",
        "note": ("Synthetic stand-in. Field names/units are this project's "
                 "documented CONTRACT, to be reconciled with the real ODEON API "
                 "spec (help-desk Q1/Q4)."),
        "scada": {
            "secondary_substation_id": SS_ID,
            "interval_minutes": 5,
            "units": {"active_power": "kW", "voltage": "V", "current": "A"},
            "readings": scada,
        },
        "lv_smart_meter": {
            "secondary_substation_id": SS_ID,
            "interval_minutes": 15,
            "units": {"active_import": "Wh", "active_export": "Wh"},
            "n_supply_points": 51,
            "readings": meter,
        },
        "lv_topology": topology,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
