"""Download official Calgary housing data and write bundled CSVs to ./data.

Sources:
  * Statistics Canada Web Data Service (CPI, New Housing Price Index, CMHC rents,
    CMHC vacancy rates, CMHC 5-year conventional mortgage rate)
  * Bank of Canada Valet API (latest weekly 5-year conventional mortgage rate)
  * City of Calgary Open Data (property assessments)

Run:  python scripts/refresh_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANUAL = DATA / "manual"

STATCAN_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromCubePidCoordAndLatestNPeriods"
BOC_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json"
CALGARY_URL = "https://data.calgary.ca/resource/{ds}.json"

CPI_PID = 18100005
CPI_CALGARY = 25
CPI_ALBERTA = 23
CPI_MEMBERS = {
    "all_items": 2,
    "shelter": 79,
    "rented_accommodation": 80,
    "tenant_insurance": 82,
    "property_taxes": 87,
    "home_insurance": 88,
    "maintenance_repairs": 89,
}
NHPI_PID = 18100205
NHPI_CALGARY = 35
NHPI_MEMBERS = {"total": 1, "house": 2, "land": 3}
RENT_PID = 34100133
RENT_CALGARY = 146
RENT_STRUCTURES = {1: "apartment", 2: "row", 3: "row_and_apartment"}
RENT_UNITS = {1: "bachelor", 2: "1br", 3: "2br", 4: "3br"}
VACANCY_PID = 34100127
VACANCY_CALGARY = 2
MORTGAGE_PID = 34100145
BOC_MORTGAGE_WEEKLY = "V80691335"

CURRENT_ASSESSMENTS = "4bsw-nn7w"
HISTORICAL_ASSESSMENTS = "4ur7-wsgc"
# City of Calgary sub-property-use codes: current-year dataset vs historical dataset (2023+).
TYPE_CODES_CURRENT = {
    "detached": ["R110"],
    "semi_detached": ["R120"],
    "townhouse": ["R401", "R402"],
    "apartment_condo": ["R201", "R301"],
}
TYPE_CODES_HISTORICAL = {
    "detached": ["RE0110"],
    "semi_detached": ["RE0120"],
    "townhouse": ["RE0401", "RE0410"],
    "apartment_condo": ["RE0201", "RE0301"],
}
# Excludes titled parking/storage units, which carry the residential class but tiny values.
MIN_RESIDENTIAL_VALUE = 50_000
AGE_BANDS = {
    "new_0_3": (0, 3),
    "age_4_15": (4, 15),
    "age_16_30": (16, 30),
    "age_31_plus": (31, 500),
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "rent-vs-own-calgary/1.0"
_retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
               allowed_methods=None)
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))


def _coord(*members: int) -> str:
    parts = list(members) + [0] * (10 - len(members))
    return ".".join(str(p) for p in parts)


def statcan_series(pid: int, *members: int, latest_n: int = 2000) -> pd.Series:
    payload = [{"productId": pid, "coordinate": _coord(*members), "latestN": latest_n}]
    r = SESSION.post(STATCAN_URL, json=payload, timeout=120)
    r.raise_for_status()
    body = r.json()[0]
    if body.get("status") != "SUCCESS":
        raise RuntimeError(f"StatCan {pid} {members}: {body}")
    pts = body["object"]["vectorDataPoint"]
    s = pd.Series(
        {pd.Timestamp(p["refPer"]): p["value"] for p in pts if p.get("value") is not None},
        dtype="float64",
    )
    return s.sort_index()


def calgary_query(ds: str, **params) -> list[dict]:
    r = SESSION.get(CALGARY_URL.format(ds=ds), params=params, timeout=300)
    r.raise_for_status()
    return r.json()


def _in(codes: list[str]) -> str:
    return "(" + ",".join(f"'{c}'" for c in codes) + ")"


def fetch_cpi() -> pd.DataFrame:
    cols = {}
    for name, member in CPI_MEMBERS.items():
        try:
            cols[name] = statcan_series(CPI_PID, CPI_CALGARY, member)
            continue
        except Exception as exc:
            if name == "all_items":
                raise
            print(f"  CPI {name}: not published for Calgary, using Alberta ({str(exc)[:60]})", file=sys.stderr)
        # Fine sub-indexes are only published at the provincial level.
        try:
            cols[f"ab_{name}"] = statcan_series(CPI_PID, CPI_ALBERTA, member)
        except Exception as exc:
            print(f"  CPI {name} (Alberta): {exc}", file=sys.stderr)
    df = pd.DataFrame(cols)
    df.index = df.index.year
    df.index.name = "year"
    return df.dropna(how="all")


def fetch_nhpi() -> pd.DataFrame:
    df = pd.DataFrame({n: statcan_series(NHPI_PID, NHPI_CALGARY, m) for n, m in NHPI_MEMBERS.items()})
    df.index.name = "date"
    return df


def fetch_rents() -> pd.DataFrame:
    rows = []
    for s_id, s_name in RENT_STRUCTURES.items():
        for u_id, u_name in RENT_UNITS.items():
            try:
                s = statcan_series(RENT_PID, RENT_CALGARY, s_id, u_id)
            except Exception as exc:  # some combinations are suppressed by CMHC
                print(f"  rents {s_name}/{u_name}: {exc}", file=sys.stderr)
                continue
            rows += [{"year": ts.year, "structure": s_name, "unit": u_name, "rent": v} for ts, v in s.items()]
    return pd.DataFrame(rows).sort_values(["structure", "unit", "year"])


def fetch_vacancy() -> pd.DataFrame:
    s = statcan_series(VACANCY_PID, VACANCY_CALGARY)
    return pd.DataFrame({"year": s.index.year, "vacancy_rate": s.values})


def fetch_mortgage_rates() -> pd.DataFrame:
    s = statcan_series(MORTGAGE_PID, 1)
    df = pd.DataFrame({"date": s.index, "rate": s.values, "source": "statcan_cmhc"})
    # Bank of Canada's chartered-bank posted rate is a different (higher) series; kept separately.
    try:
        r = SESSION.get(BOC_URL.format(series=BOC_MORTGAGE_WEEKLY), params={"recent": 260}, timeout=60)
        r.raise_for_status()
        obs = r.json()["observations"]
        weekly = pd.Series(
            {pd.Timestamp(o["d"]): float(o[BOC_MORTGAGE_WEEKLY]["v"]) for o in obs if o.get(BOC_MORTGAGE_WEEKLY)}
        ).sort_index()
        monthly = weekly.groupby(weekly.index.to_period("M")).mean()
        df = pd.concat(
            [df, pd.DataFrame({"date": monthly.index.to_timestamp(), "rate": monthly.values.round(3),
                               "source": "boc_posted"})],
            ignore_index=True,
        )
    except Exception as exc:
        print(f"  Bank of Canada fetch failed (StatCan series kept): {exc}", file=sys.stderr)
    return df


def fetch_current_assessments() -> pd.DataFrame:
    roll_year = int(float(calgary_query(CURRENT_ASSESSMENTS, **{"$select": "max(roll_year) as y"})[0]["y"]))
    rows = []
    for ptype, codes in TYPE_CODES_CURRENT.items():
        for band, rng in {"all": None, **AGE_BANDS}.items():
            where = f"assessment_class='RE' and sub_property_use in {_in(codes)}"
            if rng:
                where += f" and year_of_construction between {roll_year - rng[1]} and {roll_year - rng[0]}"
            res = calgary_query(
                CURRENT_ASSESSMENTS,
                **{"$select": "count(*) as n, median(assessed_value) as median, avg(assessed_value) as mean",
                   "$where": where},
            )[0]
            if int(res.get("n", 0)) == 0:
                continue
            rows.append({"roll_year": roll_year, "property_type": ptype, "age_band": band,
                         "n": int(res["n"]), "median": float(res["median"]), "mean": float(res["mean"])})
    return pd.DataFrame(rows)


def fetch_assessment_history(current: pd.DataFrame) -> pd.DataFrame:
    value = "replace(assessed_value, ',', '')::number"
    rows = []
    for r in calgary_query(
        HISTORICAL_ASSESSMENTS,
        **{"$select": f"roll_year, count(*) as n, median({value}) as median",
           "$where": f"assessment_class='RE' and {value} >= {MIN_RESIDENTIAL_VALUE}",
           "$group": "roll_year", "$order": "roll_year", "$limit": 100},
    ):
        rows.append({"roll_year": int(r["roll_year"]), "property_type": "all_residential",
                     "n": int(r["n"]), "median": float(r["median"])})
    for ptype, codes in TYPE_CODES_HISTORICAL.items():
        for r in calgary_query(
            HISTORICAL_ASSESSMENTS,
            **{"$select": f"roll_year, count(*) as n, median({value}) as median",
               "$where": f"assessment_class='RE' and sub_property_use in {_in(codes)}",
               "$group": "roll_year", "$order": "roll_year", "$limit": 100},
        ):
            rows.append({"roll_year": int(r["roll_year"]), "property_type": ptype,
                         "n": int(r["n"]), "median": float(r["median"])})

    # Append the current roll year so the history is up to date.
    roll_year = int(current["roll_year"].iloc[0])
    for _, r in current[current["age_band"] == "all"].iterrows():
        rows.append({"roll_year": roll_year, "property_type": r["property_type"],
                     "n": int(r["n"]), "median": float(r["median"])})
    res = calgary_query(
        CURRENT_ASSESSMENTS,
        **{"$select": "count(*) as n, median(assessed_value) as median",
           "$where": f"assessment_class='RE' and assessed_value >= {MIN_RESIDENTIAL_VALUE}"},
    )[0]
    rows.append({"roll_year": roll_year, "property_type": "all_residential",
                 "n": int(res["n"]), "median": float(res["median"])})
    df = pd.DataFrame(rows).drop_duplicates(["roll_year", "property_type"], keep="last")
    return df.sort_values(["property_type", "roll_year"])


STEPS = [
    ("cpi_calgary_annual.csv", fetch_cpi, True),
    ("nhpi_calgary_monthly.csv", fetch_nhpi, True),
    ("cmhc_rents_calgary.csv", fetch_rents, False),
    ("cmhc_vacancy_calgary.csv", fetch_vacancy, False),
    ("mortgage_rate_5yr.csv", fetch_mortgage_rates, False),
]


def main() -> int:
    DATA.mkdir(exist_ok=True)
    MANUAL.mkdir(exist_ok=True)
    meta = {"refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    failures = 0
    for fname, fn, keep_index in STEPS:
        print(f"Fetching {fname} ...")
        try:
            df = fn()
            df.to_csv(DATA / fname, index=keep_index)
            meta["files"][fname] = {"rows": len(df)}
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    print("Fetching City of Calgary assessments ...")
    try:
        current = fetch_current_assessments()
        current.to_csv(DATA / "assessments_current_by_type.csv", index=False)
        meta["files"]["assessments_current_by_type.csv"] = {"rows": len(current)}
        hist = fetch_assessment_history(current)
        hist.to_csv(DATA / "assessments_history_by_type.csv", index=False)
        meta["files"]["assessments_history_by_type.csv"] = {"rows": len(hist)}
    except Exception as exc:
        failures += 1
        print(f"  FAILED: {exc}", file=sys.stderr)

    creb = MANUAL / "creb_benchmark.csv"
    if creb.exists():
        df = pd.read_csv(creb)
        missing = {"date", "property_type", "benchmark_price"} - set(df.columns)
        if missing:
            print(f"WARNING: {creb} is missing columns {sorted(missing)}", file=sys.stderr)
        else:
            print(f"CREB manual file OK ({len(df)} rows).")

    (DATA / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Done." if not failures else f"Done with {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
