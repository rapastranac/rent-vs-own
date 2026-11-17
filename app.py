"""Calgary Rent vs Own — Streamlit app.

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from rvo import charts
from rvo.assumptions import (PORTFOLIO_PRESETS, PROPERTY_TYPES, ROUTINE_MAINTENANCE_BY_AGE, TYPE_DEFAULTS,
                             BigTicketItem, scenario_for_type)
from rvo.costs import big_ticket_events
from rvo.crossover import (classify, find_crossings, min_holding_months, sell_year_table, sensitivity_grid,
                           unrecoverable_costs)
from rvo.data_loader import derive_defaults, latest_rent, load_data, nhpi_annual, typical_price
from rvo.model import simulate
from rvo.montecarlo import run_monte_carlo

st.set_page_config(page_title="Calgary Rent vs Own", page_icon="🏠", layout="wide")


@st.cache_data
def _data():
    d = load_data()
    return d, derive_defaults(d)


data, defaults = _data()


def pct_input(label: str, value: float, key: str, step: float = 0.1, min_value: float = -50.0,
              max_value: float = 100.0, help: str | None = None, container=st.sidebar) -> float:
    return container.number_input(label, min_value=min_value, max_value=max_value, value=round(value * 100, 2),
                                  step=step, format="%.2f", key=key, help=help) / 100


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def money_short(x: float) -> str:
    sign, a = ("-" if x < 0 else ""), abs(x)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:,.0f}"


def yrs(m: float) -> str:
    return "never" if np.isnan(m) else f"{m / 12:.1f} yrs"


# ---------------------------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------------------------
sb = st.sidebar
sb.title("🏠 Calgary Rent vs Own")
sb.caption("Defaults come from official Calgary data; every value is editable.")

sb.header("Property")
ptype = sb.selectbox("Property type", list(PROPERTY_TYPES), format_func=PROPERTY_TYPES.get)
td = TYPE_DEFAULTS[ptype]
is_new = sb.radio("Condition", ["Resale", "New build"], horizontal=True) == "New build"
age = 0 if is_new else int(sb.number_input("Home age at purchase (years)", 0, 120, 20, 1))
default_price, price_src = typical_price(data, ptype, age, is_new)
price = sb.number_input("Purchase price ($)", 50_000, 5_000_000, int(round(default_price, -3)), 5_000,
                        key=f"price_{ptype}_{is_new}_{price_src}", help=f"Default: {price_src}")
sb.caption(f"Default: {price_src}")

sb.header("Mortgage")
down_pct = pct_input("Down payment (%)", 0.10, "down", step=1.0, min_value=5.0, max_value=100.0,
                     help="Raised automatically to the legal minimum (5% of the first $500k, 10% above).")
first_time = sb.checkbox("First-time home buyer", True,
                         help="Allows insured 30-year amortization and the federal FTHB GST rebate on new homes.")
amort = sb.selectbox("Amortization (years)", [25, 30, 20, 15], index=0)
mrate = pct_input("Mortgage rate, first 5-yr term (%)", defaults.mortgage_rate.value, "mrate", step=0.05,
                  min_value=0.0, max_value=25.0, help=defaults.mortgage_rate.label)
rrate = pct_input("Rate at each renewal (%)", mrate, f"rrate_{mrate}", step=0.05, min_value=0.0, max_value=25.0,
                  help="Applied every 5 years after the first term; the payment is recalculated.")

sb.header("Home value growth")
basis_labels = {k: f"{v.value:.2%} – {v.label}" for k, v in defaults.home_appreciation.items()}
basis_labels["custom"] = "Custom"
basis = sb.selectbox("Default basis", list(basis_labels), format_func=basis_labels.get)
base_appr = defaults.home_appreciation[basis].value if basis != "custom" else 0.035
appr = pct_input("Home appreciation / yr (%)", base_appr, f"appr_{basis}", step=0.1, min_value=-10.0,
                 max_value=20.0)

sb.header("Ownership costs")
tax_rate = pct_input("Property tax rate (% of value)", defaults.property_tax_rate.value, "tax", step=0.01,
                     min_value=0.0, max_value=5.0, help=defaults.property_tax_rate.label)
tax_drift = pct_input("Annual change in tax rate (%)", 0.0, "taxdrift", step=0.5,
                      help="0 = taxes grow with home value only. Positive = the mill rate rises too.")
condo = sb.number_input("Condo / HOA fees ($/month)", 0, 5_000, int(td.condo_fee_monthly), 25, key=f"condo_{ptype}",
                        help="No official series; typical Calgary levels. Townhouse default assumes a condo complex.")
condo_g = pct_input("Condo fee growth / yr (%)", 0.04, "condog", step=0.5)
ins = sb.number_input("Home insurance ($/year)", 0, 20_000, int(td.home_insurance_annual), 50, key=f"ins_{ptype}",
                      help="Calgary premiums are elevated by hail risk.")
ins_g = pct_input("Insurance growth / yr (%)", defaults.insurance_growth.value, "insg", step=0.5,
                  help=defaults.insurance_growth.label)

with sb.expander("Maintenance & repairs"):
    maint_infl = pct_input("Repair cost inflation / yr (%)", max(defaults.maintenance_inflation.value,
                                                                 defaults.inflation.value),
                           "maintinfl", step=0.1, container=st,
                           help=f"{defaults.maintenance_inflation.label} = "
                                f"{defaults.maintenance_inflation.value:.2%}; default uses the higher of this "
                                f"and general inflation.")
    maint_factor = st.slider("Owner's share of upkeep", 0.0, 1.0, float(td.maintenance_factor), 0.05,
                             key=f"mf_{ptype}", help="Condo corporations cover common-element upkeep via fees.")
    land_share = st.slider("Land share of price (land doesn't wear out)", 0.0, 0.8, float(td.land_share), 0.05,
                           key=f"land_{ptype}")
    st.caption("Routine upkeep as % of structure value per year, by home age. Low in early years "
               "(Alberta new home warranty: 1/2/5/10 years).")
    routine_df = st.data_editor(
        pd.DataFrame(ROUTINE_MAINTENANCE_BY_AGE, columns=["up_to_age", "rate"]).assign(rate=lambda x: x.rate * 100)
        .rename(columns={"up_to_age": "Up to age", "rate": "% of structure / yr"}),
        key="routine", hide_index=True, width="stretch")
    st.caption("Big-ticket replacements (today's $), triggered when the home's age hits a multiple of the lifespan.")
    bt_df = st.data_editor(
        pd.DataFrame([(b.name, b.cost, b.lifespan_years) for b in td.big_ticket],
                     columns=["Item", "Cost ($)", "Lifespan (yrs)"]),
        key=f"bt_{ptype}", num_rows="dynamic", hide_index=True, width="stretch")
    extras = st.number_input("New-build extras ($, one-time)", 0, 200_000, int(td.new_build_extras), 500,
                             key=f"extras_{ptype}", disabled=not is_new)
    include_gst = st.checkbox("Pay GST on new build (net of rebates)", True, disabled=not is_new)

sb.header("Renting the equivalent")
units = ["bachelor", "1br", "2br", "3br"]
unit = sb.selectbox("CMHC rent anchor (unit size)", units, index=units.index(td.rent_unit), key=f"unit_{ptype}")
anchor, anchor_year = latest_rent(data, td.rent_structure, unit)
premium = sb.number_input("Rent premium vs CMHC average (×)", 0.5, 3.0, float(td.rent_premium), 0.05,
                          key=f"prem_{ptype}",
                          help="CMHC averages purpose-built apartments/rows incl. long-time tenants. Renting a "
                               "house or a new lease costs more.")
rent = sb.number_input("Monthly rent ($)", 300, 20_000, int(round(anchor * premium, -1)), 25,
                       key=f"rent_{ptype}_{unit}_{premium}")
sb.caption(f"Anchor: CMHC {anchor_year} Calgary average, {td.rent_structure} {unit} = ${anchor:,.0f} × {premium:.2f}")
rent_basis = sb.selectbox("Rent growth basis", list(defaults.rent_growth) + ["custom"],
                          format_func=lambda k: "Custom" if k == "custom" else
                          f"{defaults.rent_growth[k].value:.2%} – {defaults.rent_growth[k].label}")
rent_g = pct_input("Rent growth / yr (%)", defaults.rent_growth.get(rent_basis, defaults.inflation).value,
                   f"rentg_{rent_basis}", step=0.1, help="Alberta has no rent control.")
tenant_ins = sb.number_input("Tenant insurance ($/year)", 0, 5_000, 300, 25)

sb.header("Investing the difference")
preset = sb.selectbox("Portfolio", list(PORTFOLIO_PRESETS) + ["Custom"])
p_ret, p_vol = PORTFOLIO_PRESETS.get(preset, (0.08, 0.15))
port_ret = pct_input("Annual return (%)", p_ret, f"ret_{preset}", step=0.25, min_value=-10.0, max_value=30.0,
                     help="Long-run nominal average; no account types or taxes are modelled.")
port_vol = pct_input("Annual volatility (%)", p_vol, f"vol_{preset}", step=0.5, min_value=0.0, max_value=60.0)

sb.header("Projection")
horizon = sb.slider("Horizon (years)", 1, 40, 25)
inflation = pct_input("Inflation / yr (%)", defaults.inflation.value, "infl", step=0.1, help=defaults.inflation.label)
real = sb.toggle("Show in today's dollars (inflation-adjusted)", False)

sb.header("Monte Carlo")
mc_on = sb.toggle("Run Monte Carlo", False)
mc_n = sb.select_slider("Simulations", [250, 500, 1000, 2000, 5000], 1000, disabled=not mc_on)
home_vol = pct_input("Home price volatility (%)", defaults.home_volatility.value, "hvol", step=0.5,
                     min_value=0.0, max_value=50.0, help=defaults.home_volatility.label +
                     ". New-home index is smoother than resale prices.")
corr = sb.slider("Correlation: home prices vs portfolio", -1.0, 1.0, 0.2, 0.05, disabled=not mc_on)

# ---------------------------------------------------------------------------------------------
# Build scenario
# ---------------------------------------------------------------------------------------------
routine_table = tuple(
    (int(r["Up to age"]), float(r["% of structure / yr"]) / 100)
    for _, r in routine_df.dropna().sort_values("Up to age").iterrows()
) or ROUTINE_MAINTENANCE_BY_AGE
big_ticket = tuple(
    BigTicketItem(str(r["Item"]), float(r["Cost ($)"]), int(r["Lifespan (yrs)"]))
    for _, r in bt_df.dropna().iterrows()
)
scenario = scenario_for_type(
    ptype,
    horizon_years=horizon, inflation=inflation, price=float(price), is_new_build=is_new, home_age_years=age,
    down_payment_pct=down_pct, first_time_buyer=first_time, include_gst_on_new=include_gst,
    home_appreciation=appr, mortgage_rate=mrate, renewal_rate=rrate, amortization_years=amort,
    property_tax_rate=tax_rate, property_tax_rate_drift=tax_drift, condo_fee_monthly=float(condo),
    condo_fee_growth=condo_g, home_insurance_annual=float(ins), insurance_growth=ins_g,
    maintenance_inflation=maint_infl, maintenance_factor=maint_factor, land_share=land_share,
    routine_maintenance_by_age=routine_table, big_ticket=big_ticket, new_build_extras=float(extras),
    monthly_rent=float(rent), rent_growth=rent_g, tenant_insurance_annual=float(tenant_ins),
    tenant_insurance_growth=inflation, portfolio_return=port_ret, portfolio_volatility=port_vol,
)
res = simulate(scenario)
tbl = sell_year_table(res)
k_end = 1 / res.deflator[-1] if real else 1.0
min_hold = float(min_holding_months(res.gap_net[0])[0])

st.title("Calgary: rent vs own")
st.caption(f"{PROPERTY_TYPES[ptype]}, {'new build' if is_new else f'{age}-year-old resale'}, "
           f"{money(price)} · {horizon}-year projection · "
           f"{'today’s dollars' if real else 'nominal dollars'}")
for w in res.warnings:
    st.warning(w)

tabs = st.tabs(["📈 Results", "⚖️ Break-even", "💸 Monthly costs", "🎲 Monte Carlo", "🏙️ Calgary history",
                "📋 Assumptions & sources"])

# ---------------------------------------------------------------------------------------------
with tabs[0]:
    c = st.columns(3)
    c[0].metric("Owner (keep home)", money_short(res.owner_gross[0, -1] * k_end),
                help="Home value − mortgage balance + owner's portfolio at the horizon.")
    c[1].metric("Owner (if sold)", money_short(res.owner_net[0, -1] * k_end),
                help="As above, minus realtor commission (+GST) and legal fees.")
    c[2].metric("Renter", money_short(res.renter[0, -1] * k_end), help="Renter's portfolio at the horizon.")
    gap = res.gap_net[0, -1] * k_end
    c = st.columns(3)
    c[0].metric("Owner − renter", money_short(gap), delta=f"{'owning' if gap >= 0 else 'renting'} wins",
                delta_color="off", delta_arrow="off", help="Owner (if sold) minus renter at the horizon.")
    c[1].metric("Min. holding period", yrs(min_hold),
                help=f"Earliest year after which selling always beats renting ('never' = not within {horizon} years).")
    c[2].metric("Upfront cash", money_short(res.upfront_cash),
                help="Down payment + closing costs. The renter invests this same amount on day one.")
    c = st.columns(3)
    c[0].metric("Owner cost / mo", money(res.owner_monthly_total[0, 0]), help="Month 1, all-in.")
    c[1].metric("Renter cost / mo", money(res.renter_monthly_total[0]), help="Month 1, rent + tenant insurance.")
    c[2].metric("Difference invested / mo", money(abs(res.owner_monthly_total[0, 0] - res.renter_monthly_total[0])),
                help="Month 1: the cheaper side invests this amount.")

    st.markdown(f"**If sold:** {classify(res.gap_net[0], horizon)}  \n"
                f"**If kept:** {classify(res.gap_gross[0], horizon)}")
    st.plotly_chart(charts.net_worth_chart(res, real), width="stretch")
    st.caption("✕ marks crossover points. Both households start with the same cash and spend the same amount "
               "each month; whoever has the cheaper housing invests the difference in the portfolio.")

# ---------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("When does owning pull ahead?")
    crossings = find_crossings(res.gap_net[0])
    c = st.columns(3)
    c[0].metric("Minimum holding period (if sold)", yrs(min_hold),
                help="Earliest year after which selling always beats having rented, through the horizon.")
    first_ahead = next((x.year for x in crossings if x.direction == "owner"), None)
    c[1].metric("First time owning is ahead (if sold)", f"{first_ahead:.1f} years" if first_ahead else "—")
    c[2].metric("Lead changes", len(crossings))
    if crossings:
        st.table(pd.DataFrame([{"Year": f"{x.year:.1f}", "Who takes the lead": "Owning" if x.direction ==
                                "owner" else "Renting"} for x in crossings]).set_index("Year"))
    st.plotly_chart(charts.sell_year_chart(tbl, real), width="stretch")
    st.caption("Red shading = years where selling would leave you behind the renter, largely due to realtor "
               "commission (+GST), closing costs, CMHC premium and an interest-heavy early amortization.")

    st.subheader("Sensitivity")
    axis = st.radio("Compare home appreciation against", ["Portfolio return", "Rent growth"], horizontal=True)
    appr_grid = sorted({round(appr + d, 4) for d in np.arange(-0.03, 0.0301, 0.01)})
    if axis == "Portfolio return":
        xs = sorted({round(port_ret + d, 4) for d in np.arange(-0.04, 0.0401, 0.02)})
        field, label, cur = "portfolio_return", "Portfolio return / yr", (appr, port_ret)
    else:
        xs = sorted({round(rent_g + d, 4) for d in np.arange(-0.02, 0.0201, 0.01)})
        field, label, cur = "rent_growth", "Rent growth / yr", (appr, rent_g)
    grid = sensitivity_grid(scenario, appr_grid, xs, field)
    st.plotly_chart(charts.sensitivity_heatmap(grid, field, label, horizon, cur), width="stretch")
    st.caption(f"Circled cell = current assumptions. \"never\" = owning doesn't stay ahead within {horizon} years.")

    with st.expander("Year-by-year table"):
        show = tbl.drop(columns=["deflator", "sell_wins"]).copy()
        if real:
            for col in show.columns.drop("year"):
                show[col] = show[col] / tbl["deflator"]
        st.dataframe(show.style.format({c: "${:,.0f}" for c in show.columns if c != "year"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
with tabs[2]:
    st.plotly_chart(charts.yearly_costs_chart(res, real), width="stretch")
    st.caption("Mortgage principal is a cost to cash flow but builds equity; everything else is unrecoverable.")
    st.plotly_chart(charts.unrecoverable_chart(unrecoverable_costs(res), res, real), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Purchase**")
        rows = [("Purchase price", price), ("Down payment", res.setup.down_payment),
                ("Mortgage (incl. CMHC premium)", res.setup.loan), ("CMHC insurance premium (added to loan)",
                                                                   res.setup.insurance_premium)]
        rows += list(res.buying_costs.items())
        rows += [("Total upfront cash", res.upfront_cash), ("Initial monthly payment", res.schedule.payment[0])]
        st.table(pd.DataFrame([(k, money(v)) for k, v in rows], columns=["Item", "Amount"]).set_index("Item"))
    with c2:
        st.markdown("**Scheduled big-ticket replacements**")
        ev = pd.DataFrame(big_ticket_events(scenario))
        if ev.empty:
            st.info("None due within the horizon.")
        else:
            ev["cost"] = ev["cost"].map(money)
            st.table(ev.rename(columns={"year": "Year", "home_age": "Home age", "item": "Item", "cost": "Cost"})
                     .set_index("Year"))

# ---------------------------------------------------------------------------------------------
with tabs[3]:
    if not mc_on:
        st.info("Turn on **Run Monte Carlo** in the sidebar to see a range of outcomes.")
    else:
        mc = run_monte_carlo(scenario, n=mc_n, home_vol=home_vol, correlation=corr)
        never = np.isnan(mc.breakeven_months).mean()
        p_end = (mc.result.gap_net[:, -1] >= 0).mean()
        c = st.columns(3)
        c[0].metric("Probability owning wins at horizon (if sold)", f"{p_end:.0%}")
        c[1].metric("Median minimum holding period", yrs(float(np.nanmedian(mc.breakeven_months)))
                    if never < 1 else "never")
        c[2].metric("Simulations where owning never stays ahead", f"{never:.0%}")
        defl = res.deflator if real else None
        st.plotly_chart(charts.fan_chart(mc.percentiles(mc.result.gap_net), scenario.months,
                                         "Owner (if sold) minus renter: P10 / median / P90", defl),
                        width="stretch")
        st.plotly_chart(charts.probability_chart(mc.prob_owner_ahead_by_year()), width="stretch")
        st.plotly_chart(charts.breakeven_histogram(mc.breakeven_months, horizon), width="stretch")
        st.caption(f"{mc_n:,} simulations. Annual home and portfolio returns are lognormal with medians equal to "
                   "your assumptions; rent, costs and mortgage rates are held at your assumptions.")

# ---------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("How Calgary's housing market has changed")
    cpi = data.cpi["all_items"]
    real_hist = st.toggle("Inflation-adjust history (Calgary CPI, latest year dollars)", False, key="realhist")

    def deflate(s: pd.Series) -> pd.Series:
        if not real_hist:
            return s
        f = cpi.reindex(s.index).ffill().bfill()
        return s * cpi.iloc[-1] / f

    nh = pd.DataFrame({c: data.nhpi[c].groupby(data.nhpi.index.year).mean() for c in ["total", "house", "land"]})
    nh.columns = ["House + land", "House only", "Land only"]
    st.plotly_chart(charts.line_chart(nh.apply(deflate), "New Housing Price Index, Calgary (Dec 2016 = 100)",
                                      "Index"), width="stretch")

    h = data.assessments_history.pivot(index="roll_year", columns="property_type", values="median")
    h = h.rename(columns={**PROPERTY_TYPES, "all_residential": "All residential (excl. parking/storage)"})
    st.plotly_chart(charts.line_chart(h.apply(deflate), "City of Calgary median assessed value", "$", "$,.0f"),
                    width="stretch")
    st.caption("Assessment roll year Y reflects market value on July 1 of year Y−1. Per-type values are only "
               "published from 2023.")

    r = data.rents[data.rents.structure == "row_and_apartment"].pivot(index="year", columns="unit", values="rent")
    r = r[["bachelor", "1br", "2br", "3br"]]
    st.plotly_chart(charts.line_chart(r.apply(deflate), "CMHC average rent, Calgary (row + apartment)", "$ / month",
                                      "$,.0f"), width="stretch")

    nh_a = nhpi_annual(data)
    r2 = data.rents[(data.rents.structure == "row_and_apartment") & (data.rents.unit == "2br")] \
        .set_index("year")["rent"]
    common = nh_a.index.intersection(r2.index)
    ratio = pd.DataFrame({"Price index ÷ rent index": (nh_a[common] / nh_a[common].iloc[0])
                          / (r2[common] / r2[common].iloc[0]) * 100})
    st.plotly_chart(charts.line_chart(ratio, f"Price-to-rent (NHPI vs CMHC 2br, {common[0]} = 100)", "Index"),
                    width="stretch")
    st.caption("Above 100 = home prices have outpaced rents since the base year (owning got relatively pricier).")

    mr = data.mortgage.pivot_table(index="date", columns="source", values="rate")
    mr = mr.rename(columns={"statcan_cmhc": "CMHC conventional 5-yr", "boc_posted": "Bank of Canada posted 5-yr"})
    st.plotly_chart(charts.line_chart(mr[mr.index.year >= 1980], "5-year mortgage rate (%)", "%"), width="stretch")

    cp = data.cpi[["all_items", "shelter", "rented_accommodation"]].rename(
        columns={"all_items": "All items", "shelter": "Shelter", "rented_accommodation": "Rented accommodation"})
    st.plotly_chart(charts.line_chart(cp[cp.index >= 1980], "Consumer Price Index, Calgary (2002 = 100)", "Index"),
                    width="stretch")
    st.plotly_chart(charts.line_chart(data.vacancy.set_index("year").rename(
        columns={"vacancy_rate": "Vacancy rate"}), "CMHC apartment vacancy rate, Calgary (%)", "%"), width="stretch")
    st.markdown("**City of Calgary residential property tax rates**")
    tr = data.tax_rates.drop(columns="source").copy()
    tr["median_detached_assessment"] = tr["median_detached_assessment"].map(money)
    st.table(tr.rename(columns={"year": "Year", "city_rate": "City", "provincial_rate": "Provincial",
                                "total_rate": "Total", "median_detached_assessment": "Median detached assessment"})
             .set_index("Year"))
    if data.creb is not None:
        creb = data.creb.pivot_table(index="date", columns="property_type", values="benchmark_price")
        st.plotly_chart(charts.line_chart(creb, "CREB benchmark price", "$", "$,.0f"), width="stretch")

# ---------------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Data-derived defaults")
    rows = [
        ("Inflation", defaults.inflation),
        *[(f"Home appreciation ({k})", v) for k, v in defaults.home_appreciation.items()],
        ("Home price volatility", defaults.home_volatility),
        *[(f"Rent growth ({k})", v) for k, v in defaults.rent_growth.items()],
        ("Repair cost inflation", defaults.maintenance_inflation),
        ("Home insurance growth", defaults.insurance_growth),
        ("Property tax rate", defaults.property_tax_rate),
        ("Mortgage rate", defaults.mortgage_rate),
    ]
    if defaults.posted_mortgage_rate:
        rows.append(("Posted 5-yr rate (reference)", defaults.posted_mortgage_rate))
    st.table(pd.DataFrame([(n, f"{e.value:.4%}" if "tax" in n.lower() else f"{e.value:.2%}", e.label)
                           for n, e in rows], columns=["Assumption", "Value", "Source"]).set_index("Assumption"))
    refreshed = data.metadata.get("refreshed_at", "unknown")
    st.caption(f"Data last refreshed: {refreshed}. Run `python scripts/refresh_data.py` to update.")
    st.subheader("How the model works")
    st.markdown("""
* **Same money, two paths.** Both households start with the owner's upfront cash (down payment + closing
  costs). The renter invests it on day one. Each month, whichever side has lower housing costs invests the
  difference in the same portfolio. After the mortgage is paid off, the owner is usually the one investing.
* **Owner costs:** mortgage payment (Canadian semi-annual compounding, renewed every 5 years), property tax
  (assessed value × Calgary residential rate), condo fees, home insurance, routine upkeep that rises with the
  home's age, and scheduled big-ticket replacements (roof, furnace, windows…).
* **Purchase:** minimum down payment rules, CMHC premium when < 20% down (added to the loan), Alberta Land
  Titles fees (no land transfer tax in Alberta), legal fees, inspection (resale), and for new builds GST net of
  the New Housing Rebate / First-Time Home Buyers' GST Rebate, plus extras like landscaping and blinds.
* **Selling:** Calgary-style commission (7% on the first $100k + 3% on the rest by default) + GST + legal.
* **Not modelled:** income/investment taxes (the principal residence is tax-free; investment account types
  are ignored by design), mortgage prepayment penalties, moving costs (both sides move), rent deposits.
""")
    st.subheader("Sources")
    st.markdown("""
* Statistics Canada: CPI 18-10-0005-01 (Calgary; Alberta for sub-indexes not published for Calgary),
  New Housing Price Index 18-10-0205-01, CMHC average rents 34-10-0133-01, CMHC vacancy rates 34-10-0127-01,
  CMHC 5-year conventional mortgage rate 34-10-0145-01.
* Bank of Canada Valet API: posted 5-year conventional mortgage rate (V80691335).
* City of Calgary Open Data: Current Year Property Assessments (4bsw-nn7w), Historical Property Assessments
  (4ur7-wsgc); City of Calgary property tax rates (calgary.ca).
* Optional: CREB benchmark prices (`data/manual/creb_benchmark.csv`, manual download).
* Estimates (no official series): condo fees, insurance premiums, component costs/lifespans, rent premiums for
  houses vs. purpose-built rentals, portfolio return presets.
""")
