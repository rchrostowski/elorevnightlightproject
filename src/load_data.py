# src/load_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from .config import DATA_RAW, DATA_FINAL, MODEL_DATA_FILE

def load_raw_lights() -> pd.DataFrame:
    path = DATA_RAW / "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"
    df = pd.read_csv(path)
    return df

def load_raw_sp500() -> pd.DataFrame:
    path = DATA_RAW / "sp500_clean.csv"
    df = pd.read_csv(path)
    return df

def load_raw_returns() -> pd.DataFrame:
    """
    Expects monthly returns with columns at least:
    'ticker', 'date', 'ret'
    """
    path = DATA_RAW / "sp500_monthly_returns.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load final modeling dataset for the Streamlit app.
    If missing and fallback=True, generate small synthetic sample.
    """
    path = DATA_FINAL / MODEL_DATA_FILE
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    if not fallback_if_missing:
        raise FileNotFoundError(f"{path} not found.")

    # Fallback synthetic data so app can at least run
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

