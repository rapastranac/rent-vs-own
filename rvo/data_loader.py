"""Load bundled official data and derive default assumptions from it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

AGE_BANDS = (("new_0_3", 3), ("age_4_15", 15), ("age_16_30", 30), ("age_31_plus", 10_000))
# Borrowers typically negotiate below the CMHC/posted conventional rate.
DEFAULT_MORTGAGE_DISCOUNT = 0.0075


@dataclass
class HousingData:
    cpi: pd.DataFrame
    nhpi: pd.DataFrame
    rents: pd.DataFrame
    vacancy: pd.DataFrame
    mortgage: pd.DataFrame
    assessments_current: pd.DataFrame
    assessments_history: pd.DataFrame
    tax_rates: pd.DataFrame
    creb: pd.DataFrame | None
    metadata: dict = field(default_factory=dict)


def _read(name: str, data_dir: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, **kw)


def load_data(data_dir: Path = DATA_DIR) -> HousingData:
    hist = _read("assessments_history_by_type.csv", data_dir)
    # Years where property-use codes were only sparsely populated aren't representative.
    max_n = hist.groupby("property_type")["n"].transform("max")
    hist = hist[hist["n"] >= 0.5 * max_n]
    creb_path = data_dir / "manual" / "creb_benchmark.csv"
    meta_path = data_dir / "metadata.json"
    return HousingData(
        cpi=_read("cpi_calgary_annual.csv", data_dir).set_index("year"),
        nhpi=_read("nhpi_calgary_monthly.csv", data_dir, parse_dates=["date"]).set_index("date"),
        rents=_read("cmhc_rents_calgary.csv", data_dir),
        vacancy=_read("cmhc_vacancy_calgary.csv", data_dir),
        mortgage=_read("mortgage_rate_5yr.csv", data_dir, parse_dates=["date"]),
        assessments_current=_read("assessments_current_by_type.csv", data_dir),
        assessments_history=hist,
        tax_rates=_read("manual/property_tax_rates.csv", data_dir),
        creb=pd.read_csv(creb_path, parse_dates=["date"]) if creb_path.exists() else None,
        metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


def cagr(start: float, end: float, years: float) -> float:
    if years <= 0 or start <= 0:
        return float("nan")
    return (end / start) ** (1 / years) - 1


def series_cagr(s: pd.Series, last_n_years: int | None = None) -> tuple[float, int, int]:
    """CAGR of an annual series (index = year). Returns (rate, start_year, end_year)."""
    s = s.dropna()
    if last_n_years:
        s = s[s.index >= s.index.max() - last_n_years]
    y0, y1 = int(s.index.min()), int(s.index.max())
    return cagr(float(s.loc[y0]), float(s.loc[y1]), y1 - y0), y0, y1


def nhpi_annual(d: HousingData, col: str = "total") -> pd.Series:
    s = d.nhpi[col].groupby(d.nhpi.index.year).mean()
    counts = d.nhpi[col].groupby(d.nhpi.index.year).count()
    return s[counts == 12]  # complete years only


@dataclass
class Estimate:
    value: float
    label: str  # human-readable provenance


@dataclass
class Defaults:
    inflation: Estimate
    home_appreciation: dict[str, Estimate]
    home_volatility: Estimate
    rent_growth: dict[str, Estimate]
    maintenance_inflation: Estimate
    insurance_growth: Estimate
    property_tax_rate: Estimate
    mortgage_rate: Estimate
    posted_mortgage_rate: Estimate | None


def derive_defaults(d: HousingData) -> Defaults:
    infl, y0, y1 = series_cagr(d.cpi["all_items"], 40)
    inflation = Estimate(infl, f"Calgary CPI all-items CAGR {y0}–{y1} (StatCan 18-10-0005-01)")

    nh = nhpi_annual(d)
    a_full, f0, f1 = series_cagr(nh)
    a_20, t0, t1 = series_cagr(nh, 20)
    allres = d.assessments_history[d.assessments_history.property_type == "all_residential"].set_index("roll_year")
    a_ass, s0, s1 = series_cagr(allres["median"])
    home = {
        "nhpi_full": Estimate(a_full, f"Calgary New Housing Price Index CAGR {f0}–{f1}"),
        "nhpi_20": Estimate(a_20, f"Calgary New Housing Price Index CAGR {t0}–{t1}"),
        "assessments": Estimate(a_ass, f"City of Calgary median residential assessment CAGR {s0}–{s1}"),
    }
    vol = float(np.log(nh).diff().dropna().std())
    home_vol = Estimate(vol, f"Std. dev. of annual log change, Calgary NHPI {f0}–{f1}")

    r2 = d.rents[(d.rents.structure == "row_and_apartment") & (d.rents.unit == "2br")].set_index("year")["rent"]
    g_cmhc, c0, c1 = series_cagr(r2)
    g_cpi, p0, p1 = series_cagr(d.cpi["rented_accommodation"], 40)
    rent_growth = {
        "cmhc_2br": Estimate(g_cmhc, f"CMHC Calgary 2-bedroom average rent CAGR {c0}–{c1}"),
        "cpi_rent": Estimate(g_cpi, f"Calgary CPI rented accommodation CAGR {p0}–{p1}"),
    }

    m_infl, m0, m1 = series_cagr(d.cpi["ab_maintenance_repairs"], 40)
    i_infl, i0, i1 = series_cagr(d.cpi["ab_home_insurance"], 40)

    tax = d.tax_rates.sort_values("year").iloc[-1]
    cm = d.mortgage[d.mortgage.source == "statcan_cmhc"].sort_values("date").iloc[-1]
    boc = d.mortgage[d.mortgage.source == "boc_posted"].sort_values("date")
    return Defaults(
        inflation=inflation,
        home_appreciation=home,
        home_volatility=home_vol,
        rent_growth=rent_growth,
        maintenance_inflation=Estimate(m_infl, f"Alberta CPI homeowners' maintenance & repairs CAGR {m0}–{m1}"),
        insurance_growth=Estimate(i_infl, f"Alberta CPI homeowners' insurance CAGR {i0}–{i1}"),
        property_tax_rate=Estimate(float(tax.total_rate), f"City of Calgary {int(tax.year)} residential rate "
                                   "(municipal + provincial)"),
        mortgage_rate=Estimate(
            float(cm.rate) / 100 - DEFAULT_MORTGAGE_DISCOUNT,
            f"CMHC 5-yr conventional lending rate {cm.date:%b %Y} ({cm.rate:.2f}%) less a typical "
            f"{DEFAULT_MORTGAGE_DISCOUNT:.2%} negotiated discount",
        ),
        posted_mortgage_rate=(Estimate(float(boc.iloc[-1].rate) / 100,
                                       f"Bank of Canada posted 5-yr rate {boc.iloc[-1].date:%b %Y}")
                              if len(boc) else None),
    )


def age_band(age: int, is_new: bool) -> str:
    if is_new:
        return "new_0_3"
    for band, upper in AGE_BANDS:
        if age <= upper:
            return band
    return AGE_BANDS[-1][0]


def typical_price(d: HousingData, property_type: str, age: int, is_new: bool) -> tuple[float, str]:
    cur = d.assessments_current
    band = age_band(age, is_new)
    row = cur[(cur.property_type == property_type) & (cur.age_band == band)]
    if row.empty:
        row = cur[(cur.property_type == property_type) & (cur.age_band == "all")]
        band = "all"
    r = row.iloc[0]
    band_label = {"new_0_3": "built in the last 3 years", "age_4_15": "4–15 years old",
                  "age_16_30": "16–30 years old", "age_31_plus": "31+ years old", "all": "all ages"}[band]
    return float(r["median"]), (f"City of Calgary {int(r.roll_year)} median assessment, {band_label} "
                                f"({int(r.n):,} properties)")


def latest_rent(d: HousingData, structure: str, unit: str) -> tuple[float, int]:
    s = d.rents[(d.rents.structure == structure) & (d.rents.unit == unit)].sort_values("year")
    if s.empty:
        s = d.rents[(d.rents.structure == "row_and_apartment") & (d.rents.unit == unit)].sort_values("year")
    r = s.iloc[-1]
    return float(r.rent), int(r.year)
