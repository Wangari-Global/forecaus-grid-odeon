"""Deterministic-first guard tests: supported claims pass, hallucinations flagged."""
from forecaus_grid_odeon.validation import extract_numbers, validate_claims

# A realistic set of deterministic pipeline outputs.
COMPUTED = {
    "sarimax_mape": 5.0,
    "temp_effect_mw": 699.24,
    "capex_eur": 160500,
    "mpc_saving_frac": 0.108,
    "n_refuters": 3,
}


def test_passing_narrative_is_supported():
    """Every number traces to a computed value (with rounding / unit tolerance)."""
    narrative = (
        "The SARIMAX baseline reached 5.0% MAPE. Temperature raised demand by "
        "about 699 MW. Optimised CAPEX was EUR 160,500 and predictive control cut "
        "cost by 10.8%. All 3 refuters passed."
    )
    res = validate_claims(narrative, COMPUTED)
    assert res["ok"] is True
    assert res["unsupported"] == []
    assert res["n_numbers"] >= 5


def test_hallucinated_number_is_flagged():
    """A fabricated number with no computed support is reported (the acceptance)."""
    narrative = (
        "Temperature raised demand by 699 MW, and the system saved EUR 1,234,567 "
        "per year."  # the savings figure is invented
    )
    res = validate_claims(narrative, COMPUTED)
    assert res["ok"] is False
    flagged = [u["value"] for u in res["unsupported"]]
    assert 1234567.0 in flagged
    assert 699.0 not in flagged  # the supported claim is not flagged


def test_percentage_matches_stored_fraction():
    res = validate_claims("control cut cost by 10.8%", {"saving": 0.108})
    assert res["ok"] and res["supported"][0]["matched"] == 0.108


def test_percentage_matches_stored_percent():
    res = validate_claims("MAPE was 5.0%", {"mape": 5.0})
    assert res["ok"]


def test_rounding_within_displayed_precision():
    # 207 supports 207.02; 207.9 does not.
    assert validate_claims("EUR 207", {"cost": 207.02})["ok"]
    assert not validate_claims("EUR 207.9", {"cost": 207.02})["ok"]


def test_thousands_separator_and_units():
    res = validate_claims("CAPEX EUR 160,500 over 179 kW of PV", {"capex": 160500, "pv": 179.0})
    assert res["ok"]


def test_ignore_values_skipped():
    """Non-claim numbers (e.g. a year) can be whitelisted."""
    res = validate_claims("In 2024 the effect was 699 MW.", {"effect": 699.24}, ignore=[2024])
    assert res["ok"]
    # Without ignoring, the year is (correctly) flagged as unsupported.
    assert validate_claims("In 2024 the effect was 699 MW.", {"effect": 699.24})["ok"] is False


def test_no_numbers_is_ok():
    res = validate_claims("The causal model was more robust under regime change.", COMPUTED)
    assert res["ok"] and res["n_numbers"] == 0


def test_does_not_split_alphanumeric_tokens():
    """'CO2' must not yield a spurious '2' claim."""
    nums = [n["value"] for n in extract_numbers("CO2 intensity and PV output")]
    assert nums == []
