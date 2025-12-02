# src/features.py

import pandas as pd
from pathlib import Path

from .build_panel import build_panel_firms_with_brightness
from .preprocess_stocks import load_returns_standardized
from .config import DATA_FINAL


def build_features_and_model_data() -> pd.DataFrame:
    """
    Build the final firm × month dataset used for modeling and the app.

    Pipeline:
    1) Build firm × state × month brightness panel.
    2) Load monthly stock returns.
    3) Merge on (ticker, date).
    4) Create features:
        - avg_rad_month_lag1
        - brightness_change
        - ret (current month return)
        - ret_fwd_1m (next month return)
    5) Save to data/final/nightlights_model_data.csv
    """

    # ---------------------------------------------------------
    # 1. Build firm × month brightness panel
    # ---------------------------------------------------------
    print("📊 Building firm × month brightness panel...")
    panel = build_panel_firms_with_brightness().copy()

    if "ticker" not in panel.columns:
        raise ValueError(
            f"Panel is missing 'ticker' column. Columns: {panel.columns.tolist()}"
        )
    if "date" not in panel.columns:
        raise ValueError(
            f"Panel is missing 'date' column. Columns: {panel.columns.tolist()}"
        )
    if "avg_rad_month" not in panel.columns:
        raise ValueError(
            f"Panel is missing 'avg_rad_month' column. Columns: {panel.columns.tolist()}"
        )

    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    # ---------------------------------------------------------
    # 2. Load standardized monthly returns
    # ---------------------------------------------------------
    print("📈 Loading standardized monthly returns...")
    returns = load_returns_standardized().copy()

    # Be robust: accept either 'ret' or 'return' and rename if needed
    cols = list(returns.columns)
    print(f"ℹ️ Returns columns before fix: {cols}")

    if "ret" not in returns.columns and "return" in returns.columns:
        returns = returns.rename(columns={"return": "ret"})

    if not {"ticker", "date", "ret"}.issubset(returns.columns):
        raise ValueError(
            f"Returns must have ['ticker','date','ret'] columns. "
            f"Found: {returns.columns.tolist()}"
        )

    returns["ticker"] = returns["ticker"].astype(str).str.upper().str.strip()
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")

    # ---------------------------------------------------------
    # 3. Merge brightness with returns
    # ---------------------------------------------------------
    print("🔗 Merging panel with returns on (ticker, date)...")
    df = panel.merge(
        returns[["ticker", "date", "ret"]],
        on=["ticker", "date"],
        how="inner",
        validate="m:1",  # many firm-rows per (ticker,date), 1 return per (ticker,date)
    )

    # Sort for lag/lead operations
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Feature engineering
    # ---------------------------------------------------------
    print("🧪 Creating lagged brightness and forward returns...")

    # Lagged brightness (previous month)
    df["avg_rad_month_lag1"] = df.groupby("ticker")["avg_rad_month"].shift(1)

    # Month-over-month change in brightness
    df["brightness_change"] = df["avg_rad_month"] - df["avg_rad_month_lag1"]

    # Forward 1-month return
    df["ret_fwd_1m"] = df.groupby("ticker")["ret"].shift(-1)

    # Drop rows where we can't define key features
    df = df.dropna(
        subset=[
            "avg_rad_month",
            "avg_rad_month_lag1",
            "brightness_change",
            "ret",
            "ret_fwd_1m",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. Save final dataset
    # ---------------------------------------------------------
    output_path = DATA_FINAL / "nightlights_model_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"✅ Saved final model data to {output_path} with {len(df):,} rows "
        f"and {df.shape[1]} columns."
    )

    return df

