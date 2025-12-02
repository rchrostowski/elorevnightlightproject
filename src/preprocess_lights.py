# src/preprocess_lights.py

import pandas as pd
import numpy as np

from .load_data import load_raw_lights


# ---------------------------------------------------------
# Helper: determine date from various possible formats
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

    # If none of the above found, raise clear error
    raise ValueError(
        "Nightlights file must have either 'year'+'month', 'date', or 'time' columns."
    )


# ---------------------------------------------------------
# MAIN FUNCTION: Aggregate nightlights by rounded coordinates & month
# ---------------------------------------------------------

def build_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Loads the raw VIIRS nightlights file (via load_raw_lights),
    trims the dataset to avoid blowing up Streamlit Cloud,
    converts dates, rounds lat/lon, and aggregates average radiance
    per grid cell per month.

    Returns the aggregated dataframe and also writes it to:
    data/intermediate/lights_monthly_by_coord.csv
    """

    # -----------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------
    print("📥 Loading raw nightlights data...")
    lights = load_raw_lights().copy()

    # -----------------------------------------------------
    # 2. Ensure we have a proper datetime column
    # -----------------------------------------------------
    print("🛠 Converting to datetime...")
    lights = ensure_date_column(lights)

    # -----------------------------------------------------
    # 3. TRIM YEARS to reduce dataset (massively reduces memory/time)
    # -----------------------------------------------------
    print("✂️ Trimming nightlights to years >= 2018...")
    lights = lights[lights["date"] >= "2018-01-01"]

    if lights.empty:
        raise ValueError(
            "Nightlights data is empty after trimming to year >= 2018. "
            "Check your VIIRS file's date columns."
        )

    # -----------------------------------------------------
    # 4. Round lat/lon for aggregation
    # -----------------------------------------------------
    # You might need to adjust column names depending on your VIIRS file
    lat_col = None
    lon_col = None

    # Try common names
    for c in lights.columns:
        if c.lower() in ["lat", "latitude", "y"]:
            lat_col = c
        if c.lower() in ["lon", "longitude", "x"]:
            lon_col = c

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Unable to identify latitude/longitude columns. Found columns: {lights.columns.tolist()}"
        )

    print("📍 Rounding coordinates...")
    lights["lat_round"] = lights[lat_col].round(1)
    lights["lon_round"] = lights[lon_col].round(1)

    # -----------------------------------------------------
    # 5. Identify brightness column (radiance)
    # -----------------------------------------------------
    rad_col = None
    for c in lights.columns:
        if "rad" in c.lower() or "brightness" in c.lower():
            rad_col = c
            break

    if rad_col is None:
        raise ValueError(
            f"Could not identify radiance / brightness column in VIIRS file. "
            f"Columns available: {lights.columns.tolist()}"
        )

    # -----------------------------------------------------
    # 6. Aggregate by (lat_round, lon_round, date)
    # -----------------------------------------------------
    print("📊 Aggregating monthly radiance...")

    grouped = (
        lights.groupby(["lat_round", "lon_round", "date"], as_index=False)[rad_col]
        .mean()
        .rename(columns={rad_col: "avg_rad_month"})
    )

    # -----------------------------------------------------
    # 7. Write to data/intermediate/
    # -----------------------------------------------------
    output_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "data"
        / "intermediate"
        / "lights_monthly_by_coord.csv"
    )

    print(f"💾 Writing intermediate grid to {output_path}...")
    grouped.to_csv(output_path, index=False)

    print("✅ Finished building lights_monthly_by_coord.csv")

    return grouped
