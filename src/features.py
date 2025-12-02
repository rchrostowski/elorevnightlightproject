# src/features.py
import pandas as pd
from .config import DATA_INTER, DATA_FINAL, MODEL_DATA_FILE
from .build_panel import build_panel_firms_with_brightness
from .preprocess_stocks import load_returns_standardized

def build_features_and_model_data() -> pd.DataFrame:
    panel = build_panel_firms_with_brightness().copy()
    panel["date"] = pd.to_datetime(panel["date"])

    panel = panel.sort_values(["ticker", "date"])

    # Lagged brightness and change
    panel["avg_rad_month_lag1"] = panel.groupby("ticker")["avg_rad_month"].shift(1)
    panel["brightness_change"] = (
        (panel["avg_rad_month"] - panel["avg_rad_month_lag1"])
        / panel["avg_rad_month_lag1"]
    )

    panel["avg_rad_z_within_firm"] = (
        panel.groupby("ticker")["avg_rad_month"]
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    )

    panel_features = panel.dropna(subset=["avg_rad_month_lag1", "brightness_change"])

    # Attach returns
    returns = load_returns_standardized()
    panel_ret = panel_features.merge(
        returns[["ticker", "date", "ret"]],
        on=["ticker", "date"],
        how="left",
        validate="m:1",
    )

    panel_ret = panel_ret.sort_values(["ticker", "date"])
    panel_ret["ret_fwd"] = panel_ret.groupby("ticker")["ret"].shift(-1)

    model_data = panel_ret.dropna(subset=["ret_fwd", "brightness_change"]).copy()

    out_path = DATA_FINAL / MODEL_DATA_FILE
    model_data.to_csv(out_path, index=False)
    return model_data

