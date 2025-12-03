# scripts/lookup_hq_counties.py

"""
Look up the *true* county for each HQ lat/lon using the FCC Census API.

Output:
    data/intermediate/hq_with_county.csv

Columns:
    ticker, firm, state, hq_lat, hq_lon,
    state_name, county_name, county_fips
"""

from pathlib import Path
import time
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

SP500_PATH = RAW_DIR / "sp500_clean.csv"
OUT_PATH = INTERMEDIATE_DIR / "hq_with_county.csv"


def _load_sp500_clean() -> pd.DataFrame:
    df = pd.read_csv(SP500_PATH)

    # Try to be flexible about column names
    col_map = {}

    # ticker
    for c in ["ticker", "Ticker", "TICKER"]:
        if c in df.columns:
            col_map[c] = "ticker"
            break

    # firm name
    for c in ["name", "company", "Company", "security", "Security"]:
        if c in df.columns:
            col_map[c] = "firm"
            break

    # latitude / longitude
    for c in ["lat", "Lat", "latitude", "Latitude"]:
        if c in df.columns:
            col_map[c] = "hq_lat"
            break
    for c in ["lon", "Lon", "lng", "Lng", "longitude", "Longitude"]:
        if c in df.columns:
            col_map[c] = "hq_lon"
            break

    # state postal code
    for c in ["state", "State", "state_abbrev", "state_code"]:
        if c in df.columns:
            col_map[c] = "state"
            break

    required = {"ticker", "firm", "hq_lat", "hq_lon", "state"}
    found = set(col_map.values())
    missing = required - found
    if missing:
        raise ValueError(
            f"sp500_clean.csv is missing required columns. "
            f"Found columns: {df.columns.tolist()}\n"
            f"Needed logical fields: {required}. "
            f"Please rename columns in sp500_clean or update this script."
        )

    df = df[list(col_map.keys())].rename(columns=col_map)
    df["hq_lat"] = df["hq_lat"].astype(float)
    df["hq_lon"] = df["hq_lon"].astype(float)

    return df


def fcc_lookup(lat: float, lon: float) -> dict:
    """
    Query FCC API to get county + state for a single lat/lon.
    """
    url = (
        "https://geo.fcc.gov/api/census/block/find"
        f"?latitude={lat}&longitude={lon}&format=json"
    )
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    j = r.json()

    county = j.get("County", {})
    state = j.get("State", {})

    return {
        "state_name": state.get("name"),
        "state_code": state.get("code"),
        "county_name": county.get("name"),
        "county_fips": county.get("FIPS"),
    }


def main() -> None:
    print(f"📥 Loading {SP500_PATH} ...")
    sp500 = _load_sp500_clean()

    # Unique HQs (most firms just once)
    hq_unique = (
        sp500[["ticker", "firm", "state", "hq_lat", "hq_lon"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"🔍 Unique HQ coordinates: {len(hq_unique):d}")

    rows = []
    for i, row in hq_unique.iterrows():
        lat = row["hq_lat"]
        lon = row["hq_lon"]
        tkr = row["ticker"]

        try:
            info = fcc_lookup(lat, lon)
        except Exception as e:
            print(f"⚠️  FCC lookup failed for {tkr} ({lat}, {lon}): {e}")
            info = {
                "state_name": None,
                "state_code": None,
                "county_name": None,
                "county_fips": None,
            }

        rows.append(
            {
                "ticker": tkr,
                "firm": row["firm"],
                "state": row["state"],  # postal
                "hq_lat": lat,
                "hq_lon": lon,
                **info,
            }
        )

        if (i + 1) % 25 == 0:
            print(f"   → {i+1} / {len(hq_unique)} HQs done...")
            # tiny sleep to be nice to the API
            time.sleep(0.5)

    out = pd.DataFrame(rows)

    print(f"\n✅ Saving HQ + county table → {OUT_PATH}")
    out.to_csv(OUT_PATH, index=False)

    # Quick sanity check
    print("\nExample rows:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
