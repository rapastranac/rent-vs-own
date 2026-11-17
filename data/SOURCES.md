# Data sources

All files except `manual/` are produced by `scripts/refresh_data.py`.

| File | Source | Coverage |
|---|---|---|
| `cpi_calgary_annual.csv` | Statistics Canada 18-10-0005-01, CPI annual average. Calgary for all-items, shelter and rented accommodation. Alberta (`ab_` columns) for sub-indexes not published for Calgary: property taxes, homeowners' insurance, maintenance & repairs. | 1971+ |
| `nhpi_calgary_monthly.csv` | Statistics Canada 18-10-0205-01, New Housing Price Index, Calgary (total / house / land, Dec 2016 = 100) | 1981+ |
| `cmhc_rents_calgary.csv` | Statistics Canada 34-10-0133-01, CMHC average rents, Calgary, by structure type and bedrooms | 1992+ |
| `cmhc_vacancy_calgary.csv` | Statistics Canada 34-10-0127-01, CMHC vacancy rate, Calgary (apartments, 6+ units) | 1971+ |
| `mortgage_rate_5yr.csv` | `statcan_cmhc`: Statistics Canada 34-10-0145-01, CMHC conventional 5-year lending rate (monthly). `boc_posted`: Bank of Canada Valet V80691335, chartered banks' posted 5-year rate (monthly average of weekly values). | 1951+ / last 5 yrs |
| `assessments_current_by_type.csv` | City of Calgary Open Data `4bsw-nn7w`, current-year property assessments. Median and mean by property type and age band. | current roll |
| `assessments_history_by_type.csv` | City of Calgary Open Data `4ur7-wsgc`, historical assessments. Median for all residential properties, excluding values under $50k (parking and storage units). Per-type medians where property-use codes exist. | 2005+ / 2023+ |
| `manual/property_tax_rates.csv` | City of Calgary, [Property tax rates and bill calculation](https://www.calgary.ca/property-owners/taxes/bill-rate-calculation.html) | 2023+ |
| `manual/creb_benchmark.csv` *(optional)* | CREB monthly statistics (manual download). Columns: `date,property_type,benchmark_price` | — |

Property-type mapping (City of Calgary sub-property-use codes):

| App type | Current dataset | Historical dataset |
|---|---|---|
| Detached | R110 | RE0110 |
| Semi-detached / duplex | R120 | RE0120 |
| Townhouse / row | R401, R402 | RE0401, RE0410 |
| Apartment condo | R201, R301 | RE0201, RE0301 |

An assessment roll for year Y reflects market value on July 1 of year Y−1.
