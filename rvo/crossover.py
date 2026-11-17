"""Crossover / break-even analysis of owner-minus-renter net worth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assumptions import Scenario
from .model import Result, simulate


@dataclass
class Crossing:
    month: float  # interpolated month at which the gap changes sign
    direction: str  # "owner" (owning takes the lead) or "renter"

    @property
    def year(self) -> float:
        return self.month / 12


def find_crossings(gap: np.ndarray) -> list[Crossing]:
    """All sign changes in a 1-D gap series (owner - renter), with linear interpolation."""
    gap = np.asarray(gap, dtype=float)
    crossings: list[Crossing] = []
    sign = np.sign(gap)
    # Treat exact zeros as continuing the previous sign so a touch isn't double-counted.
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    for i in range(1, len(gap)):
        if sign[i - 1] != 0 and sign[i] != sign[i - 1]:
            g0, g1 = gap[i - 1], gap[i]
            frac = g0 / (g0 - g1) if g0 != g1 else 0.0
            crossings.append(Crossing(i - 1 + frac, "owner" if g1 > 0 else "renter"))
    return crossings


def min_holding_months(gap_net: np.ndarray) -> np.ndarray:
    """Earliest month after which owning (net of selling costs) stays ahead to the horizon.

    Works on (M+1,) or (n, M+1) arrays. Returns NaN where owning is behind at the horizon.
    """
    g = np.atleast_2d(gap_net)
    ahead = g >= 0
    # suffix_all[:, m] is True when owning is ahead from month m through the horizon.
    suffix_all = np.flip(np.logical_and.accumulate(np.flip(ahead, axis=1), axis=1), axis=1)
    has = suffix_all[:, -1]
    first = np.argmax(suffix_all, axis=1).astype(float)
    first[~has] = np.nan
    return first if np.ndim(gap_net) > 1 else first[:1]


def classify(gap: np.ndarray, horizon_years: int) -> str:
    gap = np.asarray(gap, dtype=float)
    crossings = find_crossings(gap)
    if not crossings:
        lead = "Owning" if gap[-1] >= 0 else "Renting"
        return f"{lead} is ahead for the entire {horizon_years}-year horizon; the lines never cross."
    if len(crossings) == 1:
        c = crossings[0]
        if c.direction == "owner":
            return f"Renting is ahead until year {c.year:.1f}, then owning takes the lead and keeps it."
        return f"Owning is ahead until year {c.year:.1f}, then renting takes the lead and keeps it."
    parts = ", ".join(f"year {c.year:.1f} → {'owning' if c.direction == 'owner' else 'renting'}"
                      for c in crossings)
    final = "owning" if gap[-1] >= 0 else "renting"
    return f"The lead changes {len(crossings)} times ({parts}); {final} finishes ahead."


def sell_year_table(r: Result, path: int = 0) -> pd.DataFrame:
    """Outcome if the owner sells at the end of each year."""
    years = np.arange(0, r.scenario.horizon_years + 1)
    idx = years * 12
    closing = sum(r.buying_costs.values())
    df = pd.DataFrame({
        "year": years,
        "home_value": r.home_value[path, idx],
        "mortgage_balance": r.schedule.balance[idx],
        "owner_gross": r.owner_gross[path, idx],
        "owner_net": r.owner_net[path, idx],
        "renter": r.renter[path, idx],
        "selling_costs": r.selling_costs[path, idx],
        "buying_costs": closing + r.setup.insurance_premium,
        "deflator": r.deflator[idx],
    })
    df["gap_gross"] = df["owner_gross"] - df["renter"]
    df["gap_net"] = df["owner_net"] - df["renter"]
    df["sell_wins"] = df["gap_net"] >= 0
    return df


def unrecoverable_costs(r: Result, path: int = 0) -> pd.DataFrame:
    """Cumulative money spent that does not build wealth (equity or portfolio)."""
    oc = r.owner_costs
    owner = (oc["Mortgage interest"][path] + oc["Property tax"][path] + oc["Condo fees"][path]
             + oc["Home insurance"][path] + oc["Routine maintenance"][path] + oc["Big-ticket repairs"][path])
    renter = r.renter_monthly_total
    upfront = sum(r.buying_costs.values()) + r.setup.insurance_premium
    return pd.DataFrame({
        "month": np.arange(1, r.months + 1),
        "owner": upfront + np.cumsum(owner),
        "renter": np.cumsum(renter),
    })


def breakeven_year(s: Scenario) -> float:
    """Minimum holding period in years (NaN if owning never pulls ahead by the horizon)."""
    r = simulate(s)
    return float(min_holding_months(r.gap_net[0])[0]) / 12


def sensitivity_grid(s: Scenario, appreciations: list[float], x_values: list[float],
                     x_field: str = "portfolio_return") -> pd.DataFrame:
    """Break-even holding period for a grid of home appreciation x another assumption."""
    rows = []
    for a in appreciations:
        for x in x_values:
            rows.append({"home_appreciation": a, x_field: x,
                         "breakeven_years": breakeven_year(s.with_(home_appreciation=a, **{x_field: x}))})
    return pd.DataFrame(rows)
