import pytest

from rvo.mortgage import amortization_schedule, cmhc_premium_rate, minimum_down_payment, payment, setup_mortgage


def test_canadian_payment_uses_semi_annual_compounding():
    # Standard Canadian table value: 5% over 25 years ≈ $581.60 per $100k.
    assert payment(100_000, 0.05, 300) == pytest.approx(581.60, abs=0.01)


def test_minimum_down_payment_tiers():
    assert minimum_down_payment(400_000) == pytest.approx(20_000)
    assert minimum_down_payment(800_000) == pytest.approx(25_000 + 30_000)
    assert minimum_down_payment(1_600_000) == pytest.approx(320_000)


def test_cmhc_premium_and_30_year_rules():
    assert cmhc_premium_rate(0.80, 25) == 0.0
    assert cmhc_premium_rate(0.90, 25) == pytest.approx(0.031)
    assert cmhc_premium_rate(0.95, 30) == pytest.approx(0.042)
    s = setup_mortgage(500_000, 0.05, 30, first_time_buyer=False, is_new_build=False)
    assert s.amortization_years == 25 and s.warnings
    s = setup_mortgage(500_000, 0.05, 30, first_time_buyer=True, is_new_build=False)
    assert s.amortization_years == 30
    assert s.insurance_premium == pytest.approx(475_000 * 0.042)


def test_schedule_pays_off_and_renewal_changes_payment():
    sch = amortization_schedule(300_000, 0.04, 0.06, 5, 25, 30 * 12)
    assert sch.principal.sum() == pytest.approx(300_000, rel=1e-9)
    assert sch.balance[300] == pytest.approx(0, abs=1e-6)
    assert sch.payment[300:].sum() == 0
    assert sch.payment[60] > sch.payment[59]
