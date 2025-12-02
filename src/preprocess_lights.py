# src/preprocess_lights.py

import pandas as pd

from .load_data import (
    load_raw_lights,
    save_lights_monthly_by_coord,
)

# Public county centroid file (FIPS + lat/lon)
COUNTY_CENTROIDS_URL = (
    "https://gist.githubusercontent.com/russellsamora/"
    "12be4f9f574e92413ea3f92ce1bc58e6/raw/us_county_latlng.csv"
)


def _load_county_centroids() -> pd.DataFrame:
    """
    Load US county centroids (FIPS, name, lat, lon) from a CSV.

    We don't trust the exact column names ahead of time, so we:
      - read with default CSV settings
      - auto-detect the FIPS column
      - auto-detect latitude and longitude columns
    and normalize to: ['fips_code', 'lat', 'lng'].
    """
    print(f"🗺 Fetching county centroids from {COUNTY_CENTROIDS_URL} ...")
    df = pd.read_csv(COUNTY_CENTROIDS_URL)

    print("🗺 County centroid raw columns:", df.columns.tolist())

    # Find FIPS-like column
    fips_candidates = [c for c in df.columns if "fip" in c.lower()]
    if not fips_candidates:
        raise ValueError(
            "Could not find a FIPS column in county centroid file. "
            f"Columns are: {df.columns.tolist()}"
        )
    fips_col = fips_candidates[0]

    # Find latitude and longitude columns
    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
    if not lat_candidates or not lon_candidates:
        raise ValueError(
            "Could not find latitude/longitude columns in county centroid file. "
            f"Columns are: {df.columns.tolist()}"
        )

    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]

    df["fips_code"] = df[fips_col].astype(str).str.zfill(5)
    df = df.rename(columns={lat_col: "lat", lon_col: "lng"})

    print("🗺 Normalized centroid columns:", df[["fips_code", "lat", "lng"]].head())

    return df[["fips_code", "lat", "lng"]]


def build_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Build a monthly county-level brightness panel for the US with coordinates.

    Output columns:
        ['iso', 'id_1', 'name_1', 'id_2', 'name_2',
         'date', 'avg_rad_month', 'lat_round', 'lon_round']
    """
    print("🚧 Building county-level nightlights panel...")
    print("📥 Loading raw nightlights data...")
    lights_raw = load_raw_lights().copy()

    print("👉 Raw columns:", list(lights_raw.columns))

    # Keep only USA
    if "iso" in lights_raw.columns:
        lights = lights_raw[lights_raw["iso"] == "USA"].copy()
    else:
        lights = lights_raw.copy()
        lights["iso"] = "USA"

    # Ensure year/month exist
    if "year" not in lights.columns or "month" not in lights.columns:
        raise ValueError(
            "Expected 'year' and 'month' columns in nightlights data, "
            f"found: {list(lights.columns)}"
        )

    # Build datetime month
    lights["year"] = pd.to_numeric(lights["year"], errors="coerce")
    lights["month"] = pd.to_numeric(lights["month"], errors="coerce")
    lights = lights.dropna(subset=["year", "month"])
    lights["year"] = lights["year"].astype(int)
    lights["month"] = lights["month"].astype(int)

    lights["date"] = pd.to_datetime(
        dict(year=lights["year"], month=lights["month"], day=1)
    )

    # *** IMPORTANT: use 2018+ window ***
    cutoff = pd.Timestamp("2018-01-01")
    lights = lights.loc[lights["date"] >= cutoff].copy()
    print(
        f"📆 Nightlights time window after filter: "
        f"{lights['date'].min()} → {lights['date'].max()}"
    )

    # Compute average radiance per area (nlsum / area)
    for col in ["nlsum", "area"]:
        if col not in lights.columns:
            raise ValueError(
                f"Expected column '{col}' in nightlights data. "
                f"Found: {list(lights.columns)}"
            )

    lights["avg_rad_month"] = lights["nlsum"] / lights["area"]

    # Aggregate by administrative unit + date
    group_cols = ["iso", "id_1", "name_1", "id_2", "name_2", "date"]
    agg = (
        lights.groupby(group_cols, as_index=False)["avg_rad_month"]
        .mean()
        .reset_index(drop=True)
    )

    print("📊 Aggregated nightlights rows:", len(agg))

    # Attach county centroids to get lat/lon for globe and distance matching
    county_centroids = _load_county_centroids()

    # We assume id_2 is a FIPS-like county code
    agg["id_2_str"] = agg["id_2"].astype(str).str.zfill(5)

    merged = agg.merge(
        county_centroids,
        left_on="id_2_str",
        right_on="fips_code",
        how="left",
    )

    missing_coord = merged["lat"].isna().sum()
    if missing_coord > 0:
        print(
            f"⚠️ Warning: {missing_coord} county-month rows have no centroid match "
            "(id_2 not found in county list). They will keep NaN lat/lon but are still usable."
        )

    merged = merged.drop(columns=["fips_code", "id_2_str"])
    merged = merged.rename(columns={"lat": "lat_round", "lng": "lon_round"})

    print(
        "💾 Writing lights_monthly_by_coord.csv with columns:",
        list(merged.columns),
    )
    save_lights_monthly_by_coord(merged)

    print("✅ Finished building lights_monthly_by_coord.csv (county-level, 2018+).")
    return merged
