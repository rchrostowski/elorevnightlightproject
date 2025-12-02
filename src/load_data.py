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
    DATA_INTER,
    LIGHTS_URL,
    SP500_RAW_FILE,
)


def load_raw_lights() -> pd.DataFrame:
    """
    Load the raw VIIRS nightlights CSV from the URL specified in config.LIGHTS_URL.

    This is the big 'level2' file:
        VIIRS-nighttime-lights-2013m1to2024m5-level2.csv

    We first try UTF-8, and if that fails (UnicodeDecodeError), we fall back
    to a more permissive encoding (latin1).
    """
    if LIGHTS_URL is None:
        raise ValueError("LIGHTS_URL is not set in config.py")

    print("📥 Loading raw nightlights data...")
    print(f"🔗 Loading VIIRS nightlights from {LIGHTS_URL} ...")

    try:
        df = pd.read_csv(LIGHTS_URL)
    except UnicodeDecodeError:
        # Retry with a more permissive encoding
        print("⚠️ UTF-8 decode failed, retrying with latin1 encoding...")
        df = pd.read_csv(LIGHTS_URL, encoding="latin1")
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


def load_lights_monthly_by_coord(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the preprocessed region/state-level lights panel:

        data/intermediate/lights_monthly_by_coord.csv

    This is used by the Globe page. We DO NOT rebuild it here; we just
    read the CSV that was created by scripts/build_all.py.
    """
    path = DATA_INTER / "lights_monthly_by_coord.csv"

    if path.exists():
        df = pd.read_csv(path)
        # Parse date if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # File missing
    if not fallback_if_missing:
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run `python scripts/build_all.py` to generate it, "
            "then commit/push the CSV."
        )

    if HAS_STREAMLIT:
        st.warning(
            "lights_monthly_by_coord.csv not found. "
            "Globe view will be empty. "
            "Run `python scripts/build_all.py` to build it."
        )

    return pd.DataFrame()


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
        - ret_fwd_1m   (we'll also alias this to 'ret_fwd' for older pages)
    """
    path = DATA_FINAL / "nightlights_model_data.csv"

    if path.exists():
        df = pd.read_csv(path)

        # Try to parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 🔁 Alias for older code: ret_fwd_1m -> ret_fwd
        if "ret_fwd" not in df.columns and "ret_fwd_1m" in df.columns:
            df["ret_fwd"] = df["ret_fwd_1m"]

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

