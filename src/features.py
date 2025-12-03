# src/features.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .load_data import (
    load_sp500_clean,
    load_lights_monthly_by_coord,
    load_returns_standardized,
    save_model_data,
)

# -------------------------------------------------------------------
# State name <-> postal code mapping (for joining lights to HQ state)
# -------------------------------------------------------------------

STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Puerto Rico": "PR",
}

# reverse map: postal -> full name (if needed later)
ABBR_STATE = {v: k for k, v in STATE_ABBR.items()}


def _check_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a helpful error if required columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} must have columns {required}. Missing: {missing}")


def build_features_and_model_data(
    save: bool = True,
    path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build the final firm × month dataset for the project.

    Core idea:
      - Take S&P 500 firms with HQ state.
      - Take VIIRS nightlights aggregated by *state* and month.
      - Take monthly stock returns with next-month returns.
      - Link each firm to brightness in its HQ state.
      - Build brightness_change and forward return features.

    The resulting CSV is what the Streamlit app should use.
    """

    print("🏗  Building features and model data...")

    # ---------------------------------------------------------------
    # 1) Load base datasets
    # ---------------------------------------------------------------
    firms = load_sp500_clean()
    print(f"🏢 Loaded {len(firms)} firms from sp500_clean.csv")

    lights = load_lights_monthly_by_coord()
    print(f"💡 Loaded {len(lights)} rows from lights_monthly_by_coord.csv")

    returns = load_returns_standardized()
    print(f"📈 Loaded {len(returns)} rows from sp500_monthly_returns.csv")

    # ---------------------------------------------------------------
    # 2) Clean / standardize dates
    # ---------------------------------------------------------------
    # Firms should have ticker + state
    _check_columns(firms, {"ticker", "state"}, "sp500_clean")

    # Lights: keep USA only, ensure datetime
    _check_columns(
        lights,
        {"iso", "name_1", "date", "avg_rad_month"},
        "lights_monthly_by_coord",
    )
    lights = lights[lights["iso"] == "USA"].copy()
    lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
    lights = lights.dropna(subset=["date"])

    # Returns: ensure datetime and required columns
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns = returns.dropna(subset=["date"])

    # If your loader still names it "return", fix here
    if "ret" not in returns.columns and "return" in returns.columns:
        returns = returns.rename(columns={"return": "ret"})

    _check_columns(returns, {"ticker", "date", "ret"}, "returns")
    if "ret_fwd_1m" not in returns.columns:
        # if not already created, build a simple next-month return per ticker
        returns = returns.sort_values(["ticker", "date"]).copy()
        returns["ret_fwd_1m"] = returns.groupby("ticker")["ret"].shift(-1)

    # Force both sides to true datetime type (fixing your earlier merge error)
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")

    # ---------------------------------------------------------------
    # 3) Aggregate nightlights to HQ *state* × month
    # ---------------------------------------------------------------
    print("🗺  Aggregating nightlights to state × month...")

    # Map state full name -> 2-letter postal code
    lights["state_name"] = lights["name_1"].astype(str)
    lights["state"] = lights["state_name"].map(STATE_ABBR)

    before = len(lights)
    lights = lights.dropna(subset=["state"]).copy()
    dropped = before - len(lights)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} rows in lights with unmapped state_name.")

    # Restrict to 2018+ window
    lights = lights[lights["date"] >= pd.Timestamp("2018-01-01")].copy()

    # Average brightness within state and month
    state_lights = (
        lights.groupby(["state", "date"], as_index=False)["avg_rad_month"]
        .mean()
        .rename(columns={"avg_rad_month": "brightness_state"})
    )

    print(
        "📆 State lights date range:",
        state_lights["date"].min(),
        "→",
        state_lights["date"].max(),
    )

    # ---------------------------------------------------------------
    # 4) Build firm × month panel: attach HQ state, brightness, returns
    # ---------------------------------------------------------------
    print("🧩 Merging returns with HQ state...")

    # Attach HQ state to each return record
    panel = returns.merge(
        firms[["ticker", "state"]],
        on="ticker",
        how="left",
        validate="many_to-one",
    )

    missing_state = panel["state"].isna().sum()
    if missing_state > 0:
        print(
            f"⚠️ {missing_state} return rows have no HQ state. "
            "They will drop out once we merge lights."
        )

    # Now attach state brightness
    print("💡 Merging in state-level nightlights...")
    # Ensure datetime type on both sides for the merge
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    state_lights["date"] = pd.to_datetime(state_lights["date"], errors="coerce")

    panel = panel.merge(
        state_lights,
        on=["state", "date"],
        how="left",
        validate="many_to-one",
    )

    # ---------------------------------------------------------------
    # 5) Build brightness_change at the HQ state *for each firm*
    # ---------------------------------------------------------------
    print("🔁 Computing brightness_change at HQ state per firm...")

    panel = panel.sort_values(["ticker", "date"]).copy()
    panel["brightness_state_lag"] = (
        panel.groupby("ticker")["brightness_state"].shift(1)
    )
    panel["brightness_change"] = (
        panel["brightness_state"] - panel["brightness_state_lag"]
    )

    # year-month string for fixed effects: C(year_month)
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)

    # ---------------------------------------------------------------
    # 6) Final cleaning / model dataset
    # ---------------------------------------------------------------
    print("🧹 Cleaning for model-ready dataset...")

    model_cols = [
        "ticker",
        "state",
        "date",
        "year_month",
        "brightness_state",
        "brightness_state_lag",
        "brightness_change",
        "ret",
        "ret_fwd_1m",
    ]

    # Keep only columns that exist (robust if we tweak upstream)
    model_cols = [c for c in model_cols if c in panel.columns]

    df = panel[model_cols].copy()

    # Drop rows where we can't define both brightness_change and future return
    df = df.dropna(subset=["brightness_change", "ret_fwd_1m"])

    # Optional: restrict again to 2018+ just in case
    df = df[df["date"] >= pd.Timestamp("2018-02-01")].copy()

    print(f"✅ Final model rows: {len(df)}")
    if not df.empty:
        print(
            "📆 Model date range:",
            df["date"].min(),
            "→",
            df["date"].max(),
        )
        corr = df["brightness_change"].corr(df["ret_fwd_1m"])
        print(f"📊 Corr(brightness_change, next-month return): {corr:.4f}")

    # ---------------------------------------------------------------
    # 7) Save to CSV for the app
    # ---------------------------------------------------------------
    if save:
        out_path: Path = Path(path) if path is not None else Path(
            "data/final/nightlights_model_data.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_model_data(df, path=out_path)
        print(f"💾 Saved model data to {out_path}")

    return df
