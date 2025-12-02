# src/build_panel.py

import pandas as pd

from .load_data import load_raw_sp500
from .preprocess_lights import build_lights_monthly_by_coord

# Mapping from 2-letter state codes to full uppercase names
STATE_ABBREV_TO_NAME = {
    "AL": "ALABAMA",
    "AK": "ALASKA",
    "AZ": "ARIZONA",
    "AR": "ARKANSAS",
    "CA": "CALIFORNIA",
    "CO": "COLORADO",
    "CT": "CONNECTICUT",
    "DE": "DELAWARE",
    "FL": "FLORIDA",
    "GA": "GEORGIA",
    "HI": "HAWAII",
    "ID": "IDAHO",
    "IL": "ILLINOIS",
    "IN": "INDIANA",
    "IA": "IOWA",
    "KS": "KANSAS",
    "KY": "KENTUCKY",
    "LA": "LOUISIANA",
    "ME": "MAINE",
    "MD": "MARYLAND",
    "MA": "MASSACHUSETTS",
    "MI": "MICHIGAN",
    "MN": "MINNESOTA",
    "MS": "MISSISSIPPI",
    "MO": "MISSOURI",
    "MT": "MONTANA",
    "NE": "NEBRASKA",
    "NV": "NEVADA",
    "NH": "NEW HAMPSHIRE",
    "NJ": "NEW JERSEY",
    "NM": "NEW MEXICO",
    "NY": "NEW YORK",
    "NC": "NORTH CAROLINA",
    "ND": "NORTH DAKOTA",
    "OH": "OHIO",
    "OK": "OKLAHOMA",
    "OR": "OREGON",
    "PA": "PENNSYLVANIA",
    "RI": "RHODE ISLAND",
    "SC": "SOUTH CAROLINA",
    "SD": "SOUTH DAKOTA",
    "TN": "TENNESSEE",
    "TX": "TEXAS",
    "UT": "UTAH",
    "VT": "VERMONT",
    "VA": "VIRGINIA",
    "WA": "WASHINGTON",
    "WV": "WEST VIRGINIA",
    "WI": "WISCONSIN",
    "WY": "WYOMING",
    "DC": "DISTRICT OF COLUMBIA",
}


def build_panel_firms_with_brightness() -> pd.DataFrame:
    """
    Build a firm × month panel by merging S&P 500 firms with
    STATE-LEVEL VIIRS brightness.

    Requirements:
    - data/raw/sp500_clean.csv must have:
        - 'ticker'
        - 'state'  (2-letter code, e.g., 'CA', 'NY', 'TX')
    - The preprocessed nightlights (build_lights_monthly_by_coord) must
      produce columns:
        - 'iso', 'name_1', 'date', 'avg_rad_month'
      where name_1 is full state name (e.g. 'California')
    """

    # ---------------------------------------------------------
    # 1. Load firm-level data
    # ---------------------------------------------------------
    sp500 = load_raw_sp500().copy()

    if "ticker" not in sp500.columns:
        raise ValueError(
            f"Expected 'ticker' column in sp500_clean.csv. "
            f"Columns found: {sp500.columns.tolist()}"
        )

    if "state" not in sp500.columns:
        raise ValueError(
            "sp500_clean.csv must have a 'state' column with 2-letter codes "
            "(e.g., CA, NY, TX). Please add it and rerun."
        )

    # Clean up state codes
    sp500["state_code"] = (
        sp500["state"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Drop any rows with invalid or unknown codes (like 'NAN')
    valid_codes = set(STATE_ABBREV_TO_NAME.keys())
    mask_valid = sp500["state_code"].isin(valid_codes)

    if not mask_valid.all():
        bad_codes = sp500.loc[~mask_valid, "state_code"].unique().tolist()
        print(
            f"⚠️ Dropping {len(sp500) - mask_valid.sum()} firms with unknown state codes: {bad_codes}"
        )
        sp500 = sp500[mask_valid].copy()

    # Map to full state names
    sp500["state_name"] = sp500["state_code"].map(STATE_ABBREV_TO_NAME)
    sp500["state_name"] = sp500["state_name"].str.upper().str.strip()

    # ---------------------------------------------------------
    # 2. Load region-based nightlights
    # ---------------------------------------------------------
    lights = build_lights_monthly_by_coord().copy()

    required_cols = {"iso", "name_1", "date", "avg_rad_month"}
    missing = required_cols - set(lights.columns)
    if missing:
        raise ValueError(
            f"Region-based lights missing columns: {missing}. "
            f"Columns found: {lights.columns.tolist()}"
        )

    # Keep USA only, just in case
    lights = lights[lights["iso"] == "USA"].copy()

    # Standardize state names in lights
    lights["state_name"] = lights["name_1"].astype(str).str.upper().str.strip()

    # ---------------------------------------------------------
    # 3. Aggregate to STATE × DATE
    # ---------------------------------------------------------
    lights_state = (
        lights.groupby(["state_name", "date"], as_index=False)["avg_rad_month"]
        .mean()
    )

    # ---------------------------------------------------------
    # 4. Build firm × month panel: merge on state_name
    # ---------------------------------------------------------
    panel = sp500.merge(
        lights_state,
        on="state_name",
        how="left",   # many firms per state, many dates
        validate="m:m",
    )

    # Ensure date is datetime
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    return panel
