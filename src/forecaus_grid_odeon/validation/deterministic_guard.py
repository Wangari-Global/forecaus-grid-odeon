"""Deterministic-first guard: every numeric claim in narrative output must be
traceable to a computed value; otherwise reject.

This substantiates the *no-hallucination / auditable* architecture claim: an LLM
(or human) may narrate the deterministic pipeline's results, but
:func:`validate_claims` re-parses every number in the text and checks it against
the actual computed outputs. Any number that cannot be traced to a computed
value within tolerance is flagged as ``unsupported`` — a potential hallucination.

Matching tolerates how numbers are normally written: thousands separators
(``160,500``), trailing units (``207 MW``), percentages (``10.8%`` matches a
stored fraction ``0.108`` *or* a stored ``10.8``), and rounding to the precision
shown in the text (``EUR 207`` matches a computed ``207.02``).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# A number: optional sign, thousands-grouped or plain digits, optional decimals,
# optional exponent. Not preceded by a letter/digit/dot (so "CO2", "R2", parts of
# other numbers are not split out).
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def _decimals(token: str) -> Optional[int]:
    """Decimal places shown in a numeric token (None if it uses an exponent)."""
    if "e" in token or "E" in token:
        return None
    return len(token.split(".")[1]) if "." in token else 0


def extract_numbers(text: str) -> list[dict]:
    """Find numeric tokens in ``text``. Each item: {text, value, decimals, is_pct}."""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group()
        token = raw.replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        is_pct = re.match(r"\s*%", text[m.end():]) is not None
        out.append({"text": raw, "value": value, "decimals": _decimals(token), "is_pct": is_pct})
    return out


def _candidates(num: dict) -> list[tuple[float, Optional[int]]]:
    """Value interpretations of a token: the literal, plus the fraction form for
    percentages (so ``10.8%`` can match either ``10.8`` or ``0.108``)."""
    cands = [(num["value"], num["decimals"])]
    if num["is_pct"]:
        dec = None if num["decimals"] is None else num["decimals"] + 2
        cands.append((num["value"] / 100.0, dec))
    return cands


def _flatten(computed: dict) -> list[float]:
    vals: list[float] = []
    for v in computed.values():
        seq = v if isinstance(v, (list, tuple, set)) else [v]
        try:  # numpy arrays / scalars
            import numpy as np
            if isinstance(v, np.ndarray):
                seq = v.ravel().tolist()
        except ImportError:
            pass
        for x in seq:
            if isinstance(x, bool):
                continue
            if isinstance(x, (int, float)):
                vals.append(float(x))
    return vals


def _is_supported(num: dict, values: list[float], rtol: float, atol: float) -> Optional[float]:
    """Return the matched computed value, or None."""
    for cval, cdec in _candidates(num):
        for v in values:
            if abs(cval - v) <= atol + rtol * abs(v):              # tolerance match
                return v
            if cdec is not None and round(v, cdec) == round(cval, cdec):  # rounding match
                return v
    return None


def validate_claims(narrative: str, computed: dict, *, rtol: float = 0.0,
                    atol: float = 1e-6, ignore: Optional[Iterable[float]] = None) -> dict:
    """Check that every number in ``narrative`` is traceable to ``computed``.

    Parameters
    ----------
    narrative : free text containing numeric claims.
    computed  : mapping name -> computed value(s) (scalars / lists / arrays).
    rtol, atol: relative / absolute tolerance for a match.
    ignore    : numeric values to skip (e.g. years, section numbers).

    Returns ``{ok, unsupported, supported, n_numbers}`` where ``unsupported`` is a
    list of ``{text, value}`` for numbers with no computed support. ``ok`` is True
    iff every parsed number is supported.
    """
    values = _flatten(computed)
    ignore_set = {float(x) for x in (ignore or [])}

    supported, unsupported = [], []
    for num in extract_numbers(narrative):
        if num["value"] in ignore_set:
            continue
        matched = _is_supported(num, values, rtol, atol)
        if matched is None:
            unsupported.append({"text": num["text"], "value": num["value"]})
        else:
            supported.append({"text": num["text"], "value": num["value"], "matched": matched})

    return {
        "ok": len(unsupported) == 0,
        "unsupported": unsupported,
        "supported": supported,
        "n_numbers": len(supported) + len(unsupported),
    }
