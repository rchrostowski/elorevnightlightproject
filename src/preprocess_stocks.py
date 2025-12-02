# src/preprocess_stocks.py

import pandas as pd

from .load_data import load_raw_returns


def load_returns_standardized() -> pd.DataFrame:
    """
    Load raw S&P 500 monthly returns and standardize column names / types
    so the rest of the pipeline (features.py) can rely on:

        - 'ticker'
        - 'date'  (datetime64[ns])
        - 'ret'   (float, monthly return)

    Expected raw file (from load_raw_returns):
        data/raw/sp500_monthly_returns.csv

    with at least:
        - 'ticker'
        - 'date'
        - 'return'  (or already 'ret')
    """

    returns = load_raw_returns().copy()

    # Standardize column names: we want 'ticker', 'date', 'ret'
    cols = {c.lower(): c for c in returns.columns}

    # Map flexible input names to canonical ones
    # Allow 'return' or 'ret' for the returns column
    # Allow 'date' for the date column
    # Allow 'ticker' for ticker
    col_map = {}

    # Ticker
    if "ticker" in cols:
        col_map[cols["ticker"]] = "ticker"
    else:
        raise ValueError(
            f"Expected a 'ticker' column in returns file. "
            f"Found columns: {returns.columns.tolist()}"
        )

    # Date
    if "date" in cols:
        col_map[cols["date"]] = "date"
    else:
        raise ValueError(
            f"Expected a 'date' column in returns file. "
            f"Found columns: {returns.columns.tolist()}"
        )

    # Return / ret
    if "ret" in cols:
        col_map[cols["ret"]] = "ret"
    elif "return" in cols:
        col_map[cols["return"]] = "ret"
    else:
        raise ValueError(
            f"Expected a 'return' or 'ret' column in returns file. "
            f"Found columns: {returns.columns.tolist()}"
        )

    # Rename columns to canonical names
    returns = returns.rename(columns=col_map)

    # Keep only what we need
    returns = returns[["ticker", "date", "ret"]].copy()

    # Clean types
    returns["ticker"] = returns["ticker"].astype(str).str.strip().str.upper()
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns["ret"] = pd.to_numeric(returns["ret"], errors="coerce")

    # Drop rows with missing key fields
    returns = returns.dropna(subset=["ticker", "date", "ret"])

    # Sort nicely
    returns = returns.sort_values(["ticker", "date"]).reset_index(drop=True)

    return returns

def load_returns_standardized() -> pd.DataFrame:
    returns = load_raw_returns().copy()
    returns["date"] = pd.to_datetime(returns["date"])
    return returns

