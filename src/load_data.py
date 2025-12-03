# src/load_data.py

import pandas as pd
from pathlib import Path


DATA_FINAL_PATH = Path("data/final/nightlights_model_data.csv")


def _read_csv_lower(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df


def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    # Master loader for the final nightlights × returns dataset.
    # Returns a DataFrame with at least:
    #   ['ticker','date','brightness_change','ret','ret_fwd','ret_fwd_1m']
    # plus any other columns already present in the CSV.
    path = DATA_FINAL_PATH

    if not path.exists():
        if fallback_if_missing:
            print(f"⚠️ WARNING: {path} not found. Returning empty DataFrame.")
            return pd.DataFrame()
        raise FileNotFoundError(
            f"Final model dataset not found at {path}. "
            "Make sure nightlights_model_data.csv is committed there."
        )

    df = _read_csv_lower(path)

    # --- Date parsing ---
    if "date" not in df.columns:
        raise ValueError("nightlights_model_data.csv must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # --- Ticker sanity ---
    if "ticker" not in df.columns:
        raise ValueError("nightlights_model_data.csv must have a 'ticker' column.")

    # --- Brightness / light-change column standardization ---
    brightness_candidates = [
        "brightness_change",
        "delta_brightness",
        "d_brightness",
        "dlight",
        "dl",
        "brightness_delta",
    ]
    found_brightness = None
    for col in brightness_candidates:
        if col in df.columns:
            found_brightness = col
            break

    if found_brightness is None:
        # Fall back to a level column if that's all we have
        level_candidates = ["avg_rad_month", "brightness", "light", "rad"]
        for col in level_candidates:
            if col in df.columns:
                found_brightness = col
                break

    if found_brightness is None:
        raise ValueError(
            "Could not find a brightness / brightness_change column in "
            "nightlights_model_data.csv"
        )

    if found_brightness != "brightness_change":
        df = df.rename(columns={found_brightness: "brightness_change"})

    # --- Return columns ---
    # Base monthly return
    if "ret" not in df.columns:
        # Try to infer from alternative names
        ret_aliases = ["return", "returns", "ret_monthly", "monthly_return", "excess_ret"]
        for col in ret_aliases:
            if col in df.columns:
                df = df.rename(columns={col: "ret"})
                break

    if "ret" not in df.columns:
        raise ValueError(
            "nightlights_model_data.csv must have a 'ret' (monthly return) column "
            "or an alternative like 'return' / 'returns'."
        )

    # Forward 1-month return
    if "ret_fwd_1m" not in df.columns and "ret_fwd" in df.columns:
        df = df.rename(columns={"ret_fwd": "ret_fwd_1m"})

    if "ret_fwd_1m" not in df.columns:
        # Compute from ret if not supplied
        df = df.sort_values(["ticker", "date"])
        df["ret_fwd_1m"] = df.groupby("ticker")["ret"].shift(-1)

    # Keep a 'ret_fwd' alias for backwards compatibility with existing pages
    if "ret_fwd" not in df.columns:
        df["ret_fwd"] = df["ret_fwd_1m"]

    # --- Month-year key for fixed effects ---
    df["ym"] = df["date"].dt.to_period("M")

    return df


def load_returns_standardized(fallback_if_missing: bool = True) -> pd.DataFrame:
    # Convenience wrapper that returns just the standardized returns view:
    #   ['ticker','date','ret','ret_fwd_1m']
    # Sourced directly from the final model dataset.
    df = load_model_data(fallback_if_missing=fallback_if_missing)
    if df.empty:
        return df

    cols = ["ticker", "date", "ret", "ret_fwd_1m"]
    cols_present = [c for c in cols if c in df.columns]
    return df[cols_present].copy()


def add_ym(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a YYYY-MM period column `ym` if missing."""
    if "date" not in df.columns:
        raise ValueError("add_ym expects a 'date' column.")
    df["ym"] = df["date"].dt.to_period("M")
    return df
