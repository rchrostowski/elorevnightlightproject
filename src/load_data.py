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
)

# ---------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------

def load_raw_lights() -> pd.DataFrame:
    """
    Load the raw VIIRS nightlights CSV.
    Expects the file specified by LIGHTS_RAW_FILE in data/raw/.
    """
    path = DATA_RAW / LIGHTS_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Nightlights file not found: {path}")
    df = pd.read_csv(path)
    return df


def load_raw_sp500() -> pd.DataFrame:
    """
    Load the raw S&P 500 firm file with lat/long.
    Expects SP500_RAW_FILE in data/raw/.
    """
    path = DATA_RAW / SP500_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"SP500 firm file not found: {path}")
    df = pd.read_csv(path)
    return df


def load_raw_returns() -> pd.DataFrame:
    """
    Load monthly stock returns.

    Expected minimum columns:
    - ticker
    - date  (YYYY-MM-DD or similar; will be parsed to datetime)
    - ret   (monthly return, decimal, e.g. 0.02 for 2%)
    """
    path = DATA_RAW / RETURNS_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Returns file not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Returns file must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------
# Processed / final loaders
# ---------------------------------------------------------

def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the final modeling dataset (nightlights_model_data.csv)
    produced by the pipeline.

    If missing and fallback_if_missing=True, returns a small synthetic
    dataset so the Streamlit app can still run.
    """
    path = DATA_FINAL / MODEL_DATA_FILE
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    if not fallback_if_missing:
        raise FileNotFoundError(f"Final model data file not found: {path}")

    # -----------------------------------------------------
    # Synthetic fallback so the app doesn't crash
    # -----------------------------------------------------
    dates = pd.date_range("2018-01-01", periods=24, freq="MS")
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    rng = np.random.default_rng(42)

    for t in tickers:
        level = 10 + rng.normal(0, 1)
        for d in dates:
            level = level * (1 + rng.normal(0, 0.02))
            brightness_change = rng.normal(0, 0.1)
            ret_fwd = 0.01 + 0.05 * brightness_change + rng.normal(0, 0.05)
            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "avg_rad_month": level,
                    "avg_rad_month_lag1": level / (1 + brightness_change + 1e-6),
                    "brightness_change": brightness_change,
                    "avg_rad_z_within_firm": rng.normal(),
                    "ret": ret_fwd + rng.normal(0, 0.02),
                    "ret_fwd": ret_fwd,
                }
            )

    df = pd.DataFrame(rows)
    return df


def load_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Load the aggregated lights grid: one row per (lat_round, lon_round, date)
    with avg_rad_month.

    This file is produced by preprocess_lights.build_lights_monthly_by_coord()
    and saved to data/intermediate/lights_monthly_by_coord.csv.
    """
    path = DATA_INTER / "lights_monthly_by_coord.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"lights_monthly_by_coord.csv not found at {path}. "
            "Run the pipeline (scripts/build_all.py) first."
        )

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

