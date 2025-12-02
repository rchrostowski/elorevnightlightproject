# src/preprocess_stocks.py
import pandas as pd
from .config import DATA_INTER
from .load_data import load_raw_sp500, load_raw_returns

def build_sp500_with_round_coords() -> pd.DataFrame:
    sp500 = load_raw_sp500().copy()

    # Standardize coordinate columns
    rename_map = {}
    if "latitude" in sp500.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in sp500.columns:
        rename_map["longitude"] = "lon"

    sp500 = sp500.rename(columns=rename_map)

    sp500["lat_round"] = sp500["lat"].round(2)
    sp500["lon_round"] = sp500["lon"].round(2)

    out_path = DATA_INTER / "sp500_clean_with_round_coords.csv"
    sp500.to_csv(out_path, index=False)
    return sp500

def load_returns_standardized() -> pd.DataFrame:
    returns = load_raw_returns().copy()
    returns["date"] = pd.to_datetime(returns["date"])
    return returns

