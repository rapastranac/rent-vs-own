import pytest

from rvo.assumptions import scenario_for_type
from rvo.costs import gst_on_new_home, land_titles_fees, maintenance_schedule, selling_costs
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


def test_gst_rebates():
    assert gst_on_new_home(400_000, True)["net"] == pytest.approx(0)
    assert gst_on_new_home(400_000, False)["rebate"] == pytest.approx(6_300 * 0.5)
    assert gst_on_new_home(1_250_000, True)["rebate"] == pytest.approx(50_000 * 0.5)
    assert gst_on_new_home(600_000, False)["net"] == pytest.approx(30_000)


def test_alberta_fees_and_selling_costs():
    assert land_titles_fees(402_000, 380_000) == 50 + 5 * 81 + 50 + 5 * 76
    s = scenario_for_type("detached")
    assert float(selling_costs(500_000, s)) == pytest.approx((7_000 + 12_000) * 1.05 + s.sell_legal_fees)


def test_new_build_has_lower_early_maintenance_than_old_home():
    new = scenario_for_type("detached", is_new_build=True, horizon_years=5)
    old = scenario_for_type("detached", is_new_build=False, home_age_years=30, horizon_years=5)
    assert maintenance_schedule(new)[0].sum() < maintenance_schedule(old)[0].sum()


def test_big_ticket_items_fire_on_schedule():
    s = scenario_for_type("detached", home_age_years=18, horizon_years=5)
    _, big = maintenance_schedule(s)
    # Roof, furnace and fence (20-yr lifespans) are due when the home turns 20, i.e. year 2.
    assert big[24] > 0 and big[:24].sum() == 0
