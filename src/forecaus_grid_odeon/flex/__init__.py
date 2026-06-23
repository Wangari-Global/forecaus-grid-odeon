"""Flexibility / congestion application: turn an SS load forecast into the
flexibility volume + timing needed to keep the substation within its limits.

This is the Challenge-4 "energy application" layer (net-new; not reused from the
microgrid optimiser). The calculation is transparent and deterministic and is
**risk-adjusted**: it sizes flexibility against the Slice-3 forecast *interval*
(thermal and/or contractual limits), and a forecast->flex pipeline produces the
per-timestep congestion-relief schedule for a feeder.
"""
from .congestion import flexibility_need, summarize_schedule
from .pipeline import forecast_feeder, run_flex

__all__ = ["flexibility_need", "summarize_schedule", "run_flex", "forecast_feeder"]
