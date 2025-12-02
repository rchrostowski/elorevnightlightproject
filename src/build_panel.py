# src/build_panel.py
import pandas as pd
from .config import DATA_INTER
from .preprocess_lights import build_lights_monthly_by_coord
from .preprocess_stocks import build_sp500_with_round_coords

def build_panel_firms_with_brightness() -> pd.DataFrame:
    lights_monthly = build_lights_monthly_by_coord()
    sp500 = build_sp500_with_round_coords()

    panel = sp500.merge(
        lights_monthly,
        on=["lat_round", "lon_round"],
        how="left",
        validate="m:m",
    )

    out_path = DATA_INTER / "panel_firms_with_brightness.csv"
    panel.to_csv(out_path, index=False)
    return panel

