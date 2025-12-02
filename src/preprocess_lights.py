# src/preprocess_lights.py
import pandas as pd
from .config import DATA_INTER
from .load_data import load_raw_lights

def build_lights_monthly_by_coord() -> pd.DataFrame:
    lights = load_raw_lights().copy()

    # Handle date
    if "date" in lights.columns:
        lights["date"] = pd.to_datetime(lights["date"])
    elif {"year", "month"}.issubset(lights.columns):
        lights["date"] = pd.to_datetime(
            dict(year=lights["year"], month=lights["month"], day=1)
        )
    else:
        raise ValueError("Lights CSV must have either 'date' or 'year' + 'month'.")

    # Standardize column names (adjust if your actual names are different)
    rename_map = {}
    if "latitude" in lights.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in lights.columns:
        rename_map["longitude"] = "lon"
    if "avg_rad" in lights.columns:
        rename_map["avg_rad"] = "avg_rad"

    lights = lights.rename(columns=rename_map)

    # Round coordinates
    lights["lat_round"] = lights["lat"].round(2)
    lights["lon_round"] = lights["lon"].round(2)

    # Monthly average brightness by rounded coord
    lights_monthly = (
        lights.groupby(["lat_round", "lon_round", "date"], as_index=False)["avg_rad"]
        .mean()
        .rename(columns={"avg_rad": "avg_rad_month"})
    )

    out_path = DATA_INTER / "lights_monthly_by_coord.csv"
    lights_monthly.to_csv(out_path, index=False)
    return lights_monthly

