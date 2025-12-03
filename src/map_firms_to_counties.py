# src/map_firms_to_counties.py

import pandas as pd
from pathlib import Path

FIRMS = Path("data/raw/sp500_clean.csv")
COUNTIES = Path("data/intermediate/lights_monthly_by_coord.csv")
OUT = Path("data/intermediate/firm_hq_county.csv")


def _find_column(df, candidates, what):
    """
    Find the first matching column name in `candidates` (case-insensitive).
    Raise a clear error if none are found.
    """
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(
        f"sp500_clean.csv must have a column for {what}. "
        f"Tried: {candidates}. Found columns: {list(df.columns)}"
    )


def build_firm_hq_to_county():
    # ---------- Load inputs ----------
    firms = pd.read_csv(FIRMS)
    lights = pd.read_csv(COUNTIES)

    # Normalize lights column names
    lights.columns = [c.strip().lower() for c in lights.columns]

    # We expect county-level centroids in lights_monthly_by_coord
    required_lights = {"id_2", "name_2", "name_1", "lat_round", "lon_round"}
    if not required_lights.issubset(set(lights.columns)):
        raise ValueError(
            "lights_monthly_by_coord.csv is missing county centroid columns. "
            f"Expected at least: {required_lights}. Found: {set(lights.columns)}"
        )

    # ---------- Work out firms columns flexibly ----------
    ticker_col = _find_column(firms, ["ticker", "symbol"], "ticker")
    firm_col = _find_column(firms, ["firm", "company", "name"], "firm name")
    lat_col = _find_column(
        firms,
        ["lat", "latitude", "hq_lat", "headquarters_lat"],
        "latitude",
    )
    lon_col = _find_column(
        firms,
        ["lon", "lng", "long", "longitude", "hq_lon", "headquarters_lon"],
        "longitude",
    )

    firms_norm = firms[[ticker_col, firm_col, lat_col, lon_col]].copy()
    firms_norm = firms_norm.rename(
        columns={
            ticker_col: "ticker",
            firm_col: "firm",
            lat_col: "lat",
            lon_col: "lon",
        }
    )
    firms_norm["lat"] = pd.to_numeric(firms_norm["lat"], errors="coerce")
    firms_norm["lon"] = pd.to_numeric(firms_norm["lon"], errors="coerce")
    firms_norm = firms_norm.dropna(subset=["lat", "lon"])

    # ---------- Deduplicate counties (one row per county) ----------
    counties = lights[
        ["id_2", "name_2", "name_1", "lat_round", "lon_round"]
    ].drop_duplicates("id_2")

    counties = counties.rename(
        columns={
            "id_2": "county_fips",
            "name_2": "county_name",
            "name_1": "state",
        }
    )

    counties["lat_round"] = pd.to_numeric(counties["lat_round"], errors="coerce")
    counties["lon_round"] = pd.to_numeric(counties["lon_round"], errors="coerce")
    counties = counties.dropna(subset=["lat_round", "lon_round"])

    if counties.empty:
        raise ValueError("No valid county centroids after numeric conversion.")

    # ---------- Compute nearest county for each HQ ----------
    def nearest_county(lat, lon):
        d = (counties["lat_round"] - lat) ** 2 + (counties["lon_round"] - lon) ** 2
        i = d.idxmin()
        return counties.loc[i]

    records = []
    for _, row in firms_norm.iterrows():
        c = nearest_county(row["lat"], row["lon"])
        records.append(
            {
                "ticker": row["ticker"],
                "firm": row["firm"],
                "county_fips": c["county_fips"],
                "county_name": c["county_name"],
                "state": c["state"],
                "lat": row["lat"],
                "lon": row["lon"],
            }
        )

    out = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"✅ Saved firm HQ → county mapping to {OUT}")
    print(f"Unique counties mapped: {out['county_fips'].nunique()}")
    print(out.head())


if __name__ == "__main__":
    build_firm_hq_to_county()
