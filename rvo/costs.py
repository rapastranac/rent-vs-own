"""Transaction and running costs for owning in Calgary / Alberta."""

from __future__ import annotations

import math

import numpy as np

from .assumptions import BigTicketItem, Scenario

GST_RATE = 0.05


def gst_on_new_home(price: float, first_time_buyer: bool) -> dict[str, float]:
    """GST payable on a new home net of federal rebates.

    * GST New Housing Rebate: 36% of GST up to $6,300, phased out between $350k and $450k.
    * First-Time Home Buyers' GST Rebate (agreements from May 27, 2025): full GST relief up to
      $50,000 on new homes up to $1M, phased out linearly to zero at $1.5M.
    """
    gst = GST_RATE * price
    regular = min(0.36 * gst, 6_300.0)
    if price > 450_000:
        regular = 0.0
    elif price > 350_000:
        regular *= (450_000 - price) / 100_000
    fthb = 0.0
    if first_time_buyer:
        fthb = min(gst, 50_000.0)
        if price > 1_500_000:
            fthb = 0.0
        elif price > 1_000_000:
            fthb *= (1_500_000 - price) / 500_000
    rebate = max(regular, fthb)
    return {"gst": gst, "rebate": rebate, "net": gst - rebate}


def land_titles_fees(price: float, mortgage: float) -> float:
    """Alberta Land Titles: $50 + $5 per $5,000 (or part) for both transfer and mortgage registration."""
    transfer = 50 + 5 * math.ceil(price / 5_000)
    mortgage_reg = 50 + 5 * math.ceil(mortgage / 5_000) if mortgage > 0 else 0
    return float(transfer + mortgage_reg)


def buying_costs(s: Scenario, mortgage_amount: float) -> dict[str, float]:
    costs = {
        "Legal fees & title insurance": s.buy_legal_fees,
        "Alberta Land Titles fees": land_titles_fees(s.price, mortgage_amount),
    }
    if not s.is_new_build:
        costs["Inspection & appraisal"] = s.inspection_appraisal
    if s.is_new_build:
        if s.include_gst_on_new:
            costs["GST (net of rebates)"] = gst_on_new_home(s.price, s.first_time_buyer)["net"]
        costs["New-build extras (landscaping, blinds, ...)"] = s.new_build_extras
    return costs


def selling_costs(value, s: Scenario):
    """Realtor commission (Calgary convention: x% on first $100k, y% on balance) + GST + legal."""
    v = np.asarray(value, dtype=float)
    commission = s.sell_commission_first_100k * np.minimum(v, 100_000) + s.sell_commission_rest * np.maximum(
        v - 100_000, 0
    )
    return commission * (1 + s.gst_on_commission) + s.sell_legal_fees


def routine_maintenance_rate(age: int, table: tuple[tuple[int, float], ...]) -> float:
    for upper, rate in table:
        if age <= upper:
            return rate
    return table[-1][1]


def maintenance_schedule(s: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Monthly routine and big-ticket maintenance (nominal dollars) over the horizon.

    Routine upkeep is a share of the structure value (price excluding land) that depends on
    the home's age. Big-ticket components are replaced whenever the home's age reaches a
    multiple of the component lifespan; replacements due at purchase are assumed to be
    already priced in (they're charged from year 1 onward).
    """
    months = s.months
    routine = np.zeros(months)
    big = np.zeros(months)
    structure_value = s.price * (1 - s.land_share)
    age0 = s.effective_home_age
    for y in range(s.horizon_years):
        infl = (1 + s.maintenance_inflation) ** y
        age = age0 + y
        rate = routine_maintenance_rate(age, s.routine_maintenance_by_age)
        start, end = y * 12, min((y + 1) * 12, months)
        routine[start:end] = rate * structure_value * s.maintenance_factor * infl / 12
        if y >= 1:
            big[start] = sum(item.cost for item in s.big_ticket if _due(item, age)) * infl
    return routine, big


def _due(item: BigTicketItem, age: int) -> bool:
    return item.lifespan_years > 0 and age > 0 and age % item.lifespan_years == 0


def big_ticket_events(s: Scenario) -> list[dict]:
    events = []
    age0 = s.effective_home_age
    for y in range(1, s.horizon_years):
        age = age0 + y
        infl = (1 + s.maintenance_inflation) ** y
        for item in s.big_ticket:
            if _due(item, age):
                events.append({"year": y, "home_age": age, "item": item.name, "cost": item.cost * infl})
    return events
