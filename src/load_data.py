# src/load_data.py

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from .config import (
    DATA_RAW,
    DATA_FINAL,
    LIGHTS_URL,
    SP500_RAW_FILE,
)


def load_raw_lights() -> pd.DataFrame:
    """
    Load the raw VIIRS nightlights CSV from the URL specified in config.LIGHTS_URL.

    This is the big 'level2' file:
        VIIRS-nighttime-lights-2013m1to2024m5-level2.csv
    """
    if LIGHTS_URL is None:
        raise ValueError("LIGHTS_URL is not set in config.py")

    try:
        print(f"🔗 Loading VIIRS nightlights from {LIGHTS_URL} ...")
        df = pd.read_csv(LIGHTS_URL)
    except Exception as e:
        raise RuntimeError(f"Error loading nightlights data from {LIGHTS_URL}: {e}")

    return df


def _ensure_exists(path: Path, message_name: Optional[str] = None) -> None:
    """
    Helper: raise a clear FileNotFoundError if a file is missing.
    """
    if not path.exists():
        label = message_name or path.name
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Upload {label} to {path.parent}/"
        )


def load_raw_sp500() -> pd.DataFrame:
    """
    Load the raw S&P 500 firm file from data/raw/SP500_RAW_FILE.

    Expected to have at least:
        - ticker
        - company
        - lat
        - lon
        - state   (added by your script)
    """
    path = DATA_RAW / SP500_RAW_FILE
    _ensure_exists(path, SP500_RAW_FILE)
    df = pd.read_csv(path)

    return df


def load_raw_returns() -> pd.DataFrame:
    """
    Load raw monthly returns from data/raw/sp500_monthly_returns.csv.

    Expected columns:
        - ticker
        - date
        - return  (or ret; later standardized in preprocess_stocks)
    """
    path = DATA_RAW / "sp500_monthly_returns.csv"
    _ensure_exists(path, "sp500_monthly_returns.csv")

    df = pd.read_csv(path)
    return df


def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the final modeling dataset used by the Streamlit app.

    File:
        data/final/nightlights_model_data.csv

    Expected columns include:
        - ticker
        - company (if carried from sp500_clean)
        - state / state_name
        - date
        - avg_rad_month
        - avg_rad_month_lag1
        - brightness_change
        - ret
        - ret_fwd_1m
    """
    path = DATA_FINAL / "nightlights_model_data.csv"

    if path.exists():
        df = pd.read_csv(path)

        # Try to parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    # If we get here, file is missing
    if not fallback_if_missing:
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run `python scripts/build_all.py` to generate it."
        )

    # Fallback mode: warn (if in Streamlit) and return empty DataFrame
    if HAS_STREAMLIT:
        st.warning(
            "Final model data not found at "
            f"`{path}`. Returning empty DataFrame. "
            "Run `python scripts/build_all.py` in your repo to build it."
        )

    return pd.DataFrame()
