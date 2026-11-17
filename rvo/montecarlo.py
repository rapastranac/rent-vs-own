"""Monte Carlo over home appreciation and portfolio returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assumptions import Scenario
from .crossover import min_holding_months
from .model import Result, simulate


@dataclass
class MonteCarloResult:
    result: Result
    breakeven_months: np.ndarray  # (n,), NaN where owning never stays ahead

    @property
    def n(self) -> int:
        return self.result.home_value.shape[0]

    def percentiles(self, series: np.ndarray, qs=(10, 50, 90)) -> pd.DataFrame:
        p = np.percentile(series, qs, axis=0)
        return pd.DataFrame({f"p{q}": p[i] for i, q in enumerate(qs)})

    def prob_owner_ahead_by_year(self) -> pd.DataFrame:
        years = np.arange(0, self.result.scenario.horizon_years + 1)
        idx = years * 12
        return pd.DataFrame({
            "year": years,
            "p_owner_ahead_if_sold": (self.result.gap_net[:, idx] >= 0).mean(axis=0),
            "p_owner_ahead_gross": (self.result.gap_gross[:, idx] >= 0).mean(axis=0),
        })


def annual_paths(years: int, mean: float, vol: float, z: np.ndarray) -> np.ndarray:
    """Lognormal annual returns whose median equals the deterministic assumption."""
    log_r = np.log1p(mean) + vol * z[:, :years]
    return np.expm1(log_r)


def run_monte_carlo(s: Scenario, n: int = 1000, home_vol: float = 0.08, correlation: float = 0.0,
                    seed: int | None = 42) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    years = s.horizon_years
    z1 = rng.standard_normal((n, years))
    z2 = correlation * z1 + np.sqrt(max(0.0, 1 - correlation**2)) * rng.standard_normal((n, years))
    home_annual = annual_paths(years, s.home_appreciation, home_vol, z1)
    port_annual = annual_paths(years, s.portfolio_return, s.portfolio_volatility, z2)
    to_monthly = lambda a: np.repeat((1 + a) ** (1 / 12) - 1, 12, axis=1)[:, : s.months]  # noqa: E731
    r = simulate(s, home_monthly=to_monthly(home_annual), portfolio_monthly=to_monthly(port_annual))
    return MonteCarloResult(result=r, breakeven_months=min_holding_months(r.gap_net))
