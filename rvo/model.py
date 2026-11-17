"""Monthly owner-vs-renter simulation with cash-flow parity.

Both households start with the same cash and spend the same amount every month:
* At purchase the owner spends the down payment and closing costs; the renter invests that cash.
* Each month whichever side has the lower housing cost invests the difference.

Net worth:
* Owner (gross)   = home value - mortgage balance + owner's portfolio
* Owner (if sold) = gross - selling costs (realtor commission + GST + legal)
* Renter          = renter's portfolio

The simulation is vectorized over ``n`` paths so the same code serves the deterministic
projection (n=1) and the Monte Carlo (n=many).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assumptions import Scenario
from .costs import buying_costs, maintenance_schedule, selling_costs
from .mortgage import MortgageSetup, Schedule, amortization_schedule, setup_mortgage


def annual_to_monthly(rate) -> np.ndarray:
    return (1 + np.asarray(rate, dtype=float)) ** (1 / 12) - 1


@dataclass
class Result:
    scenario: Scenario
    setup: MortgageSetup
    schedule: Schedule
    buying_costs: dict[str, float]
    upfront_cash: float
    home_value: np.ndarray  # (n, M+1)
    owner_portfolio: np.ndarray  # (n, M+1)
    renter_portfolio: np.ndarray  # (n, M+1)
    selling_costs: np.ndarray  # (n, M+1)
    owner_costs: dict[str, np.ndarray]  # each (n, M)
    renter_costs: dict[str, np.ndarray]  # each (M,)
    deflator: np.ndarray  # (M+1,)
    warnings: list[str] = field(default_factory=list)

    @property
    def months(self) -> int:
        return self.scenario.months

    @property
    def owner_gross(self) -> np.ndarray:
        return self.home_value - self.schedule.balance[None, :] + self.owner_portfolio

    @property
    def owner_net(self) -> np.ndarray:
        return self.owner_gross - self.selling_costs

    @property
    def renter(self) -> np.ndarray:
        return self.renter_portfolio

    @property
    def gap_gross(self) -> np.ndarray:
        return self.owner_gross - self.renter

    @property
    def gap_net(self) -> np.ndarray:
        return self.owner_net - self.renter

    @property
    def owner_monthly_total(self) -> np.ndarray:
        return sum(v for k, v in self.owner_costs.items())

    @property
    def renter_monthly_total(self) -> np.ndarray:
        return sum(self.renter_costs.values())


def simulate(s: Scenario, home_monthly: np.ndarray | None = None,
             portfolio_monthly: np.ndarray | None = None) -> Result:
    M = s.months
    if M <= 0:
        raise ValueError("Horizon must be at least one month")
    if home_monthly is None:
        home_monthly = np.full((1, M), annual_to_monthly(s.home_appreciation))
    if portfolio_monthly is None:
        portfolio_monthly = np.full((1, M), annual_to_monthly(s.portfolio_return))
    home_monthly = np.atleast_2d(home_monthly)
    portfolio_monthly = np.atleast_2d(portfolio_monthly)
    n = max(home_monthly.shape[0], portfolio_monthly.shape[0])
    home_monthly = np.broadcast_to(home_monthly, (n, M))
    portfolio_monthly = np.broadcast_to(portfolio_monthly, (n, M))

    setup = setup_mortgage(s.price, s.down_payment_pct, s.amortization_years, s.first_time_buyer, s.is_new_build)
    sched = amortization_schedule(setup.loan, s.mortgage_rate, s.renewal_rate, s.term_years,
                                  setup.amortization_years, M)
    closing = buying_costs(s, setup.loan)
    upfront = setup.down_payment + sum(closing.values())

    years = np.arange(M) // 12
    tax_rate = s.property_tax_rate * (1 + s.property_tax_rate_drift) ** years
    condo = s.condo_fee_monthly * (1 + s.condo_fee_growth) ** years
    insurance = s.home_insurance_annual / 12 * (1 + s.insurance_growth) ** years
    routine, big = maintenance_schedule(s)
    rent = s.monthly_rent * (1 + s.rent_growth) ** years
    tenant_ins = s.tenant_insurance_annual / 12 * (1 + s.tenant_insurance_growth) ** years
    renter_total = rent + tenant_ins

    value = np.empty((n, M + 1))
    value[:, 0] = s.price
    value[:, 1:] = s.price * np.cumprod(1 + home_monthly, axis=1)
    prop_tax = value[:, :-1] * tax_rate[None, :] / 12

    fixed_owner = sched.payment + condo + insurance + routine + big  # (M,)
    owner_total = fixed_owner[None, :] + prop_tax  # (n, M)
    diff = owner_total - renter_total[None, :]

    owner_pf = np.zeros((n, M + 1))
    renter_pf = np.zeros((n, M + 1))
    renter_pf[:, 0] = upfront
    growth = 1 + portfolio_monthly
    for m in range(M):
        renter_pf[:, m + 1] = renter_pf[:, m] * growth[:, m] + np.maximum(diff[:, m], 0)
        owner_pf[:, m + 1] = owner_pf[:, m] * growth[:, m] + np.maximum(-diff[:, m], 0)

    bcast = lambda a: np.broadcast_to(a, (n, M))  # noqa: E731
    owner_costs = {
        "Mortgage interest": bcast(sched.interest),
        "Mortgage principal": bcast(sched.principal),
        "Property tax": prop_tax,
        "Condo fees": bcast(condo),
        "Home insurance": bcast(insurance),
        "Routine maintenance": bcast(routine),
        "Big-ticket repairs": bcast(big),
    }
    renter_costs = {"Rent": rent, "Tenant insurance": tenant_ins}
    deflator = (1 + s.inflation) ** (np.arange(M + 1) / 12)

    warnings = list(setup.warnings)
    return Result(
        scenario=s,
        setup=setup,
        schedule=sched,
        buying_costs=closing,
        upfront_cash=upfront,
        home_value=value,
        owner_portfolio=owner_pf,
        renter_portfolio=renter_pf,
        selling_costs=selling_costs(value, s),
        owner_costs=owner_costs,
        renter_costs=renter_costs,
        deflator=deflator,
        warnings=warnings,
    )
