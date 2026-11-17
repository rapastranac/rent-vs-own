"""Canadian mortgage mathematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INSURED_PRICE_CAP = 1_500_000
# CMHC homeowner mortgage loan insurance premiums by loan-to-value (upper bound, rate).
CMHC_PREMIUMS: tuple[tuple[float, float], ...] = (
    (0.65, 0.0060),
    (0.75, 0.0170),
    (0.80, 0.0240),
    (0.85, 0.0280),
    (0.90, 0.0310),
    (0.95, 0.0400),
)
CMHC_30_YEAR_SURCHARGE = 0.0020


def monthly_rate(annual_rate: float) -> float:
    """Fixed-rate Canadian mortgages compound semi-annually, not in advance."""
    return (1 + annual_rate / 2) ** (1 / 6) - 1


def payment(principal: float, annual_rate: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    r = monthly_rate(annual_rate)
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def minimum_down_payment(price: float) -> float:
    if price >= INSURED_PRICE_CAP:
        return 0.20 * price
    if price <= 500_000:
        return 0.05 * price
    return 0.05 * 500_000 + 0.10 * (price - 500_000)


def cmhc_premium_rate(ltv: float, amortization_years: int) -> float:
    if ltv <= 0.80:
        # Conventional mortgages (20%+ down) are normally uninsured.
        return 0.0
    for upper, rate in CMHC_PREMIUMS:
        if ltv <= upper + 1e-9:
            return rate + (CMHC_30_YEAR_SURCHARGE if amortization_years > 25 else 0.0)
    raise ValueError(f"Loan-to-value {ltv:.1%} exceeds the 95% insured maximum")


@dataclass
class MortgageSetup:
    price: float
    down_payment: float
    base_loan: float
    insurance_premium: float
    loan: float
    insured: bool
    amortization_years: int
    warnings: list[str]


def setup_mortgage(price: float, down_payment_pct: float, amortization_years: int,
                   first_time_buyer: bool, is_new_build: bool) -> MortgageSetup:
    warnings: list[str] = []
    min_down = minimum_down_payment(price)
    down = max(down_payment_pct * price, min_down)
    if down > down_payment_pct * price + 0.5:
        warnings.append(f"Down payment raised to the legal minimum of ${min_down:,.0f}.")
    base_loan = price - down
    ltv = base_loan / price if price else 0.0
    insured = ltv > 0.80
    amort = amortization_years
    if insured and amort > 25 and not (first_time_buyer or is_new_build):
        warnings.append("Insured 30-year amortization is only for first-time buyers or new builds; using 25 years.")
        amort = 25
    if insured and amort > 30:
        warnings.append("Insured mortgages are limited to 30-year amortization.")
        amort = 30
    premium = base_loan * cmhc_premium_rate(ltv, amort) if insured else 0.0
    return MortgageSetup(price, down, base_loan, premium, base_loan + premium, insured, amort, warnings)


@dataclass
class Schedule:
    payment: np.ndarray  # (months,)
    interest: np.ndarray  # (months,)
    principal: np.ndarray  # (months,)
    balance: np.ndarray  # (months + 1,) balance at each month boundary
    rate: np.ndarray  # (months,) annual contract rate in effect


def amortization_schedule(loan: float, initial_rate: float, renewal_rate: float, term_years: int,
                          amortization_years: int, months: int) -> Schedule:
    """Monthly schedule; the payment is recomputed at each renewal over the remaining amortization."""
    pay = np.zeros(months)
    interest = np.zeros(months)
    princ = np.zeros(months)
    rates = np.zeros(months)
    balance = np.zeros(months + 1)
    balance[0] = loan
    amort_months = amortization_years * 12
    term_months = max(1, term_years * 12)
    current_payment = 0.0
    rate = initial_rate
    for m in range(months):
        bal = balance[m]
        if m >= amort_months or bal <= 1e-6:
            balance[m + 1] = 0.0
            continue
        if m % term_months == 0:
            rate = initial_rate if m == 0 else renewal_rate
            current_payment = payment(bal, rate, amort_months - m)
        r = monthly_rate(rate)
        i = bal * r
        p = min(current_payment - i, bal)
        pay[m] = i + p
        interest[m] = i
        princ[m] = p
        rates[m] = rate
        balance[m + 1] = bal - p
    return Schedule(pay, interest, princ, balance, rates)
