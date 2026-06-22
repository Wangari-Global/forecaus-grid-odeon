"""Deterministic-first validation: trace every narrated number to a computed value."""
from .deterministic_guard import extract_numbers, validate_claims

__all__ = ["validate_claims", "extract_numbers"]
