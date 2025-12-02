# src/preprocess_lights.py

import pandas as pd
from pathlib import Path

from .load_data import load_raw_lights


# ---------------------------------------------------------
# Helper: ensure we have a proper datetime 'date' column
# ---------------------------------------------------------

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the dataframe has a 'date' column in datetime64 format.

    Supported input formats:
    - Separate columns: 'year', 'month'
    - A single column:  'date'
    - A single column:  'time'
    """

    # Case 1 — separate year + month columns
    if "year" in df.columns and "month" in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str),
            errors="coerce",
        )
        return df

    # Case 2 — already has a 'date' column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # Case 3 — has 'time' column
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce")
        return df

    raise ValueError(
        "Nightlights file must have either 'year'+'month', 'date', or 'time' columns."
    )


# ---------------------------------------------------------
# MAIN: Build trimmed lights grid for panel & globe
# ---------------------------------------------------------

def build_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Load raw VIIRS nightlights, heavily trim it (years + geography),
    then aggregate average radiance by rounded lat/lon and month.

    Output is saved to:
        data/intermediate/lights_monthly_by_coord.csv
    """

    print("📥 Loading raw nightlights data...")
    lights = load_raw_lights().copy()

    # 1) Ensure we have a proper datetime column
    print("🛠 Ensuring datetime 'date' column...")
    lights = ensure_date_column(lights)

    # 2) Trim to recent years ONLY (aggressive)
    #    This massively reduces data size & compute.
    CUTOFF_DATE = "2022-01-01"
    print(f"✂️ Trimming to dates >= {CUTOFF_DATE}...")
    lights = lights[lights["date"] >= CUTOFF_DATE]

    if lights.empty:
        raise ValueError(
            f"Nightlights data is empty after trimming to date >= {CUTOFF_DATE}. "
            "Check your VIIRS file's date columns."
        )

    # 3) Identify lat/lon columns (auto-detect common names)
    lat_col = None
    lon_col = None

    for c in lights.columns:
        cl = c.lower()
        if cl in ["lat", "latitude", "y"]:
            lat_col = c
        if cl in ["lon", "longitude", "x"]:
            lon_col = c

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Unable to identify latitude/longitude columns. "
            f"Columns found: {lights.columns.tolist()}"
        )

    # 4) Trim to (roughly) U.S. bounding box to avoid global grid
    #    Assumes lon is in [-180, 180].
    print("✂️ Trimming to approximate U.S. bounding box (lat 24–50, lon -130 to -60)...")
    lights = lights[
        (lights[lat_col] >= 24.0)
        & (lights[lat_col] <= 50.0)
        & (lights[lon_col] >= -130.0)
        & (lights[lon_col] <= -60.0)
    ]

    if lights.empty:
        raise ValueError(
            "Nightlights data is empty after trimming to U.S. lat/lon bounds. "
            "Check that your VIIRS file uses lat/lon in degrees in [-180, 180]."
        )

    # 5) Round coordinates for aggregation
    print("📍 Rounding coordinates...")
    lights["lat_round"] = lights[lat_col].round(1)
    lights["lon_round"] = lights[lon_col].round(1)

    # 6) Detect brightness / radiance column
    rad_col = None
    for c in lights.columns:
        cl = c.lower()
        if "rad" in cl or "brightness" in cl:
            rad_col = c
            break

    if rad_col is None:
        raise ValueError(
            f"Could not identify radiance / brightness column in VIIRS file. "
            f"Columns available: {lights.columns.tolist()}"
        )

    # 7) Aggregate by (lat_round, lon_round, date)
    print("📊 Aggregating monthly radiance on trimmed grid...")
    grouped = (
        lights.groupby(["lat_round", "lon_round", "date"], as_index=False)[rad_col]
        .mean()
        .rename(columns={rad_col: "avg_rad_month"})
    )

    # 8) Write to data/intermediate/
    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "intermediate"
        / "lights_monthly_by_coord.csv"
    )

    print(f"💾 Writing trimmed grid to {output_path}...")
    grouped.to_csv(output_path, index=False)

    print("✅ Finished building trimmed lights_monthly_by_coord.csv")

    return grouped

