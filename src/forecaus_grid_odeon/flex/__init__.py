"""Flexibility / congestion application: turn an SS load forecast into the
flexibility volume needed to keep the substation within its limit.

This is the Challenge-4 "energy application" layer (net-new; not reused from
the microgrid optimiser). v0 is a transparent deterministic calculation; the
contractual-limit handling and per-direction aggregation are flagged TODO.
"""
from .congestion import flexibility_need

__all__ = ["flexibility_need"]
