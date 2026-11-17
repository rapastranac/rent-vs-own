# Calgary Rent vs Own

A Streamlit tool that projects a single person's net worth forward for **owning vs renting in Calgary**. The cheaper option's monthly savings, plus the renter's unspent down payment, go into an investment portfolio. Default assumptions are derived from official Calgary data, and every value is editable.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The data is bundled in `data/`, so the app works offline. To refresh it from the official APIs:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_data.py
```

Run the tests with `.\.venv\Scripts\python.exe -m pytest`.

## What it models

- **Cash-flow parity.** Both households start with the same cash: the owner's down payment plus closing costs. Each month, whichever has the cheaper housing invests the difference in the same portfolio.
- **Property types:** detached, semi-detached/duplex, townhouse/row and apartment condo. Each has its own default price (the City of Calgary median assessment for that type and age band), condo fees, insurance, upkeep and rent anchor. The renter rents the equivalent property type.
- **New vs old:**
  - Routine upkeep rises with the home's age.
  - Big-ticket replacements are scheduled by component lifespan: roof, furnace, hot water tank, windows, siding, appliances and so on.
  - New builds have low early costs because of the Alberta new-home warranty. They also pay GST net of the New Housing Rebate or the First-Time Home Buyers' GST Rebate, plus extras like landscaping and blinds.
- **Mortgage:**
  - Canadian semi-annual compounding, with 5-year terms and a renewal rate.
  - Minimum down payment rules.
  - CMHC insurance premiums, with a surcharge for a 30-year amortization.
- **Alberta transaction costs:**
  - Land Titles fees (there is no land transfer tax), legal fees and an inspection.
  - Selling costs: realtor commission (7% on the first $100k plus 3% on the rest), GST on the commission, and legal fees.
- **Property tax:** home value times the City of Calgary residential rate (municipal plus provincial), with an optional annual change in the rate.
- **Inflation:** results in nominal dollars, or in today's dollars via a toggle.

## Crossover / break-even analysis

- **Lead changes:** every month where owner and renter net worth swap places, for both the "keep the home" and "sell" views. The outcome is described in plain language, e.g. "renting ahead until year 6.3, then owning".
- **Minimum holding period:** the earliest year after which selling always beats having rented.
- **"If you sell in year k" chart:** shows the gain or loss for each sell year. Losing years are shaded, and the drag from selling costs is shown separately.
- **Sensitivity heatmaps:** the break-even year across home appreciation × portfolio return, or home appreciation × rent growth.
- **Monte Carlo (optional):** lognormal home and portfolio returns, with correlation. Shows a P10–P90 range, the probability owning is ahead in each year, and the distribution of break-even years.

## Project layout

```
app.py                  Streamlit UI
rvo/assumptions.py      Scenario dataclass, per-type defaults, portfolio presets
rvo/data_loader.py      Load bundled CSVs, derive default assumptions
rvo/mortgage.py         Canadian mortgage math, CMHC insurance
rvo/costs.py            GST/rebates, Land Titles, selling costs, maintenance schedule
rvo/model.py            Vectorized monthly simulation
rvo/crossover.py        Crossings, minimum holding period, sell-year table, sensitivity
rvo/montecarlo.py       Monte Carlo wrapper
rvo/charts.py           Plotly figures
scripts/refresh_data.py Download official data → data/*.csv
data/SOURCES.md         Data provenance
```

## Limitations

- Income and investment taxes are not modelled. Home gains are tax-free (principal residence), and investment account types are ignored by design.
- The City of Calgary only publishes per-type assessment values from 2023 onward. Long-run price history comes from the StatCan New Housing Price Index (1981 on, new homes only) and the City's median assessment for all residential properties (2005 on).
- CMHC rents are averages for purpose-built rentals and include long-time tenants. House rents use an editable premium on top of that.
- Condo fees, insurance premiums and component costs have no official series. They are documented estimates.
