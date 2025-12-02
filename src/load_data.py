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
# RAW NIGHTLIGHTS LOADER (with encoding fallback)
# ---------------------------------------------------------

def load_raw_lights() -> pd.DataFrame:
    """
    Load the raw VIIRS nightlights directly from Dropbox (or local file if present).

    We try multiple encodings because the VIIRS CSV is huge and may contain
    characters outside UTF-8.

    Priority:
    1) Local file in data/raw/
    2) Direct Dropbox URL (LIGHTS_URL)
    """

    local_path = DATA_RAW / LIGHTS_RAW_FILE

    # --- 1. Try local version first (rarely used)
    if local_path.exists():
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass  # if local is corrupted, fall through to URL

    # --- 2. Use Dropbox URL
    if not LIGHTS_URL:
        raise FileNotFoundError(
            "Nightlights file missing and LIGHTS_URL is not set in config.py"
        )

    print("🔗 Loading VIIRS nightlights from Dropbox (this may take a while)...")

    # Try multiple encodings
    for enc in [None, "utf-8", "latin1"]:
        try:
            df = pd.read_csv(
                LIGHTS_URL,
                encoding=enc,
                on_bad_lines="skip",  # skip malformed rows instead of crashing
            )
            return df
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "Unable to decode the nightlights CSV using UTF-8 or latin1."
    )


# ---------------------------------------------------------
# RAW SP500 LOADER
# ---------------------------------------------------------

def load_raw_sp500() -> pd.DataFrame:
    """
    Load local sp500_clean.csv from data/raw.
    Must contain columns: ticker, lat, lon (or latitude/longitude).
    """
    path = DATA_RAW / SP500_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nUpload sp500_clean.csv to data/raw/"
        )
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------
# RAW RETURNS LOADER
# ---------------------------------------------------------

def load_raw_returns() -> pd.DataFrame:
    """
    Load monthly stock returns from data/raw/sp500_monthly_returns.csv.
    Must contain: ticker, date, ret
    """
    path = DATA_RAW / RETURNS_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nUpload sp500_monthly_returns.csv to data/raw/"
        )

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("The returns file must include a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------
# FINAL MODEL DATA LOADER (AUTO-BUILDS PIPELINE IF MISSING)
# ---------------------------------------------------------

def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Loads data/final/nightlights_model_data.csv.

    Behavior:
    • If file exists → load it.
    • If missing and fallback_if_missing = False → auto-run pipeline.
    • If missing and fallback_if_missing = True → create AAA/BBB/CCC synthetic demo.
    """
    path = DATA_FINAL / MODEL_DATA_FILE

    # --- Case 1: Final file exists → just load it
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # --- Case 2: Missing and we want REAL DATA → auto build pipeline
    if not fallback_if_missing:
        print("⚠️ nightlights_model_data.csv missing → Building pipeline now...")

        # Lazy import fixes circular deps
        from .features import build_features_and_model_data

        df = build_features_and_model_data()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # --- Case 3: Missing and fallback=True → synthetic AAA/BBB/CCC
    print("⚠️ Using synthetic AAA/BBB/CCC data (fallback enabled).")

    dates = pd.date_range("2018-01-01", periods=24, freq="MS")
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(42)

    rows = []
    for t in tickers:
        level = 10 + rng.normal()
        for d in dates:
            level *= (1 + rng.normal(0, 0.02))
            change = rng.normal(0, 0.1)
            ret_fwd = 0.01 + 0.05 * change + rng.normal(0, 0.03)

            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "avg_rad_month": level,
                    "avg_rad_month_lag1": level / (1 + change + 1e-6),
                    "brightness_change": change,
                    "avg_rad_z_within_firm": rng.normal(),
                    "ret": ret_fwd + rng.normal(0, 0.02),
                    "ret_fwd": ret_fwd,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# INTERMEDIATE GRID (used by Globe + Overview)
# ---------------------------------------------------------

def load_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Load data/intermediate/lights_monthly_by_coord.csv
    required by the Globe page.
    """
    path = DATA_INTER / "lights_monthly_by_coord.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "Run the pipeline or let load_model_data(auto-build) generate it."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


