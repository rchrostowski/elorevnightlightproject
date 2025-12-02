# src/load_data.py

import pandas as pd
import numpy as np
from pathlib import Path

from .config import (
    DATA_RAW,
    DATA_INTER,
    DATA_FINAL,
    LIGHTS_RAW_FILE,
    SP500_RAW_FILE,
    RETURNS_RAW_FILE,
    MODEL_DATA_FILE,
    LIGHTS_URL,
)

# ---------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------

def load_raw_lights() -> pd.DataFrame:
    """
    Load the raw VIIRS nightlights data.

    Priority:
    1) Use local file if present (data/raw/...)
    2) Otherwise load directly from the Dropbox URL (LIGHTS_URL)
    """
    local_path = DATA_RAW / LIGHTS_RAW_FILE

    # Use local version if it exists
    if local_path.exists():
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass  # fallback to URL

    # Load directly from Dropbox (huge file)
    if not LIGHTS_URL:
        raise FileNotFoundError(
            "Nightlights file missing locally and LIGHTS_URL is not set."
        )

    print("🔗 Loading nightlights directly from Dropbox URL...")
    df = pd.read_csv(LIGHTS_URL)
    return df


def load_raw_sp500() -> pd.DataFrame:
    """
    Load the raw S&P 500 firm file (lat/long).
    Must exist locally in data/raw/.
    """
    path = DATA_RAW / SP500_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"SP500 firm file not found: {path}\n"
            "→ Put sp500_clean.csv in data/raw/"
        )
    df = pd.read_csv(path)
    return df


def load_raw_returns() -> pd.DataFrame:
    """
    Load monthly stock returns.
    Must exist locally in data/raw/.
    """
    path = DATA_RAW / RETURNS_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {path}\n"
            "→ Put sp500_monthly_returns.csv in data/raw/"
        )

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Returns file must contain 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------
# Processed / final dataset loaders
# ---------------------------------------------------------

def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load final modeling dataset used by the Streamlit app.

    If missing and fallback=True → generates a small synthetic dataset
    so the app still loads.
    """
    path = DATA_FINAL / MODEL_DATA_FILE

    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    if not fallback_if_missing:
        raise FileNotFoundError(
            f"Final model data not found: {path}\n"
            "→ Run: python scripts/build_all.py"
        )

    # ---------------------------------------------------
    # Fallback synthetic dataset (app won't crash)
    # ---------------------------------------------------
    print("⚠️ nightlights_model_data.csv missing — using synthetic demo data.")

    dates = pd.date_range("2018-01-01", periods=24, freq="MS")
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    rng = np.random.default_rng(42)

    for t in tickers:
        level = 10 + rng.normal(0, 1)
        for d in dates:
            level *= (1 + rng.normal(0, 0.02))
            brightness_change = rng.normal(0, 0.1)
            ret_fwd = 0.01 + 0.05*brightness_change + rng.normal(0, 0.05)
            rows.append({
                "ticker": t,
                "date": d,
                "avg_rad_month": level,
                "avg_rad_month_lag1": level / (1 + brightness_change + 1e-6),
                "brightness_change": brightness_change,
                "avg_rad_z_within_firm": rng.normal(),
                "ret": ret_fwd + rng.normal(0, 0.02),
                "ret_fwd": ret_fwd,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Intermediate lights (globe + Overview page)
# ---------------------------------------------------------

def load_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Load the processed VIIRS grid (lat_round, lon_round, date, avg_rad_month)
    produced by preprocess_lights.build_lights_monthly_by_coord().
    """
    path = DATA_INTER / "lights_monthly_by_coord.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"lights_monthly_by_coord.csv not found: {path}\n"
            "→ Run: python scripts/build_all.py"
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


