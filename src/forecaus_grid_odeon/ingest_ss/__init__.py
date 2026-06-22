"""Secondary-substation (SS) data adapters for the ODEON Energy Data Space.

ODEON delivers pilot data via its API (not public ENTSO-E). The exact API
surface is not released pre-submission (help-desk Q1/Q4), so these adapters are
stubs with the intended signatures; the generic :mod:`forecaus_grid_odeon.ingest`
framework (public ENTSO-E/RTE/weather) remains available as a development proxy.
"""
from . import odeon_api

__all__ = ["odeon_api"]
