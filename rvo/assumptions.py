"""Scenario inputs and per-property-type defaults.

Numbers that are not available from an official data series (condo fees, insurance
premiums, component replacement costs, ...) are documented estimates for Calgary and
are always editable in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

PROPERTY_TYPES: dict[str, str] = {
    "detached": "Detached house",
    "semi_detached": "Semi-detached / duplex",
    "townhouse": "Townhouse / row",
    "apartment_condo": "Apartment condo",
}


@dataclass(frozen=True)
class BigTicketItem:
    name: str
    cost: float  # today's dollars
    lifespan_years: int


@dataclass(frozen=True)
class TypeDefaults:
    condo_fee_monthly: float
    home_insurance_annual: float
    land_share: float  # share of price that is land (not subject to wear)
    maintenance_factor: float  # owner's share of routine upkeep (condo corp covers the rest)
    new_build_extras: float  # landscaping, fence, blinds, appliances not in builder price
    rent_structure: str  # CMHC structure used as the rent anchor
    rent_unit: str  # CMHC unit size used as the rent anchor
    rent_premium: float  # multiplier on the CMHC anchor (house > row, new lease > avg tenant)
    big_ticket: tuple[BigTicketItem, ...]


_APPLIANCES = BigTicketItem("Appliances", 6_000, 15)
_FLOORING_PAINT = BigTicketItem("Flooring & interior paint", 9_000, 18)

TYPE_DEFAULTS: dict[str, TypeDefaults] = {
    "detached": TypeDefaults(
        condo_fee_monthly=0,
        home_insurance_annual=2_600,  # Calgary hail claims push premiums above national average
        land_share=0.45,
        maintenance_factor=1.0,
        new_build_extras=20_000,
        rent_structure="row",
        rent_unit="3br",
        rent_premium=1.45,
        big_ticket=(
            BigTicketItem("Roof (asphalt, hail-prone)", 16_000, 20),
            BigTicketItem("Furnace", 7_000, 20),
            BigTicketItem("Hot water tank", 2_800, 12),
            BigTicketItem("Windows", 25_000, 30),
            BigTicketItem("Siding / exterior", 20_000, 35),
            BigTicketItem("Fence & deck", 9_000, 20),
            _APPLIANCES,
            _FLOORING_PAINT,
        ),
    ),
    "semi_detached": TypeDefaults(
        condo_fee_monthly=0,
        home_insurance_annual=2_200,
        land_share=0.40,
        maintenance_factor=0.85,
        new_build_extras=15_000,
        rent_structure="row",
        rent_unit="3br",
        rent_premium=1.25,
        big_ticket=(
            BigTicketItem("Roof (asphalt, hail-prone)", 10_000, 20),
            BigTicketItem("Furnace", 6_500, 20),
            BigTicketItem("Hot water tank", 2_800, 12),
            BigTicketItem("Windows", 16_000, 30),
            BigTicketItem("Siding / exterior", 12_000, 35),
            BigTicketItem("Fence & deck", 6_000, 20),
            _APPLIANCES,
            _FLOORING_PAINT,
        ),
    ),
    "townhouse": TypeDefaults(
        condo_fee_monthly=400,  # condo townhouse complexes; exterior covered by the corporation
        home_insurance_annual=800,
        land_share=0.25,
        maintenance_factor=0.5,
        new_build_extras=6_000,
        rent_structure="row",
        rent_unit="3br",
        rent_premium=1.10,
        big_ticket=(
            BigTicketItem("Furnace", 6_000, 20),
            BigTicketItem("Hot water tank", 2_600, 12),
            _APPLIANCES,
            BigTicketItem("Flooring & interior paint", 7_000, 18),
        ),
    ),
    "apartment_condo": TypeDefaults(
        condo_fee_monthly=550,
        home_insurance_annual=550,
        land_share=0.10,
        maintenance_factor=0.3,
        new_build_extras=3_000,
        rent_structure="apartment",
        rent_unit="2br",
        rent_premium=1.10,
        big_ticket=(
            BigTicketItem("Appliances", 5_000, 15),
            BigTicketItem("Flooring & interior paint", 5_500, 18),
        ),
    ),
}

# Routine upkeep as a share of the structure value per year, by home age (upper bound, rate).
# New homes are cheap to run, especially under the Alberta new home warranty
# (1 yr materials/labour, 2 yr systems, 5 yr envelope, 10 yr structure).
ROUTINE_MAINTENANCE_BY_AGE: tuple[tuple[int, float], ...] = (
    (1, 0.001),
    (4, 0.004),
    (9, 0.007),
    (24, 0.010),
    (999, 0.013),
)

# Approximate long-run nominal annual returns (geometric) and volatility.
PORTFOLIO_PRESETS: dict[str, tuple[float, float]] = {
    "S&P 500 (total return, ~1926-2025)": (0.103, 0.18),
    "S&P/TSX Composite (total return, ~1970-2025)": (0.088, 0.16),
    "Global balanced 60/40": (0.072, 0.10),
    "GICs / high-interest savings": (0.035, 0.01),
}


@dataclass(frozen=True)
class Scenario:
    # Projection
    horizon_years: int = 25
    inflation: float = 0.028

    # Property and purchase
    property_type: str = "apartment_condo"
    price: float = 330_000
    is_new_build: bool = False
    home_age_years: int = 15
    down_payment_pct: float = 0.10
    first_time_buyer: bool = True
    include_gst_on_new: bool = True
    home_appreciation: float = 0.035

    # Mortgage (Canadian fixed-rate, semi-annual compounding)
    mortgage_rate: float = 0.0445
    renewal_rate: float = 0.0445
    term_years: int = 5
    amortization_years: int = 25

    # Ownership running costs
    property_tax_rate: float = 0.0066499
    property_tax_rate_drift: float = 0.0  # annual relative change of the mill rate
    condo_fee_monthly: float = 550
    condo_fee_growth: float = 0.04
    home_insurance_annual: float = 550
    insurance_growth: float = 0.05
    maintenance_inflation: float = 0.03
    maintenance_factor: float = 0.3
    land_share: float = 0.10
    routine_maintenance_by_age: tuple[tuple[int, float], ...] = ROUTINE_MAINTENANCE_BY_AGE
    big_ticket: tuple[BigTicketItem, ...] = field(default_factory=tuple)
    new_build_extras: float = 3_000

    # Transaction costs (Alberta)
    buy_legal_fees: float = 1_800
    inspection_appraisal: float = 600  # only charged on resale purchases
    sell_commission_first_100k: float = 0.07
    sell_commission_rest: float = 0.03
    gst_on_commission: float = 0.05
    sell_legal_fees: float = 1_200

    # Renting
    monthly_rent: float = 2_000
    rent_growth: float = 0.03
    tenant_insurance_annual: float = 300
    tenant_insurance_growth: float = 0.03

    # Investing
    portfolio_return: float = 0.103
    portfolio_volatility: float = 0.18

    @property
    def months(self) -> int:
        return int(self.horizon_years * 12)

    @property
    def effective_home_age(self) -> int:
        return 0 if self.is_new_build else int(self.home_age_years)

    def with_(self, **changes) -> "Scenario":
        return replace(self, **changes)


def scenario_for_type(property_type: str, **overrides) -> Scenario:
    d = TYPE_DEFAULTS[property_type]
    base = dict(
        property_type=property_type,
        condo_fee_monthly=d.condo_fee_monthly,
        home_insurance_annual=d.home_insurance_annual,
        land_share=d.land_share,
        maintenance_factor=d.maintenance_factor,
        new_build_extras=d.new_build_extras,
        big_ticket=d.big_ticket,
    )
    base.update(overrides)
    return Scenario(**base)
