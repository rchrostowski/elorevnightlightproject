# src/preprocess_lights.py

import pandas as pd
from pathlib import Path

from .load_data import load_raw_lights
from .config import DATA_INTER


def build_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Build a region-based (state-level) monthly nightlights panel from the
    raw VIIRS level-2 file.

    Input (from load_raw_lights / LIGHTS_URL) is expected to have columns:
        - iso        (country code, e.g. 'USA')
        - id_1       (first admin level id)
        - name_1     (first admin level name, e.g. 'California')
        - id_2       (second admin level id)
        - name_2     (second admin level name, e.g. county)
        - year       (int)
        - month      (int)
        - nlsum      (sum of radiance over the region)
        - area       (area of the region)

    We:
        1) Filter to USA
        2) Create a proper datetime 'date' column
        3) Trim to dates >= 2018-01-01 (your chosen analysis window)
        4) Compute avg_rad_month = nlsum / area
        5) Aggregate by (iso, id_1, name_1, id_2, name_2, date)
        6) Save to data/intermediate/lights_monthly_by_coord.csv

    Note: despite the function name, this is now a REGION-based panel
    (state / subregion), not raw lat/lon coordinates.
    """

    print("📥 Loading raw nightlights data...")
    lights = load_raw_lights().copy()
    print(f"👉 Raw columns: {lights.columns.tolist()}")

    expected_cols = {"iso", "id_1", "name_1", "id_2", "name_2", "year", "month", "nlsum", "area"}
    missing = expected_cols - set(lights.columns)
    if missing:
        raise ValueError(
            f"Raw lights data is missing columns: {missing}. "
            f"Found columns: {lights.columns.tolist()}"
        )

    # 1) Filter to USA only
    print("🌎 Filtering to iso == 'USA'...")
    lights = lights[lights["iso"] == "USA"].copy()

    # 2) Ensure datetime 'date' column from year/month
    print("🛠 Ensuring datetime 'date' column...")
    lights["year"] = pd.to_numeric(lights["year"], errors="coerce").astype("Int64")
    lights["month"] = pd.to_numeric(lights["month"], errors="coerce").astype("Int64")

    lights["date"] = pd.to_datetime(
        dict(year=lights["year"], month=lights["month"], day=1),
        errors="coerce",
    )

    # 3) Trim to your real analysis window: 2018+
    cutoff = pd.Timestamp("2018-01-01")
    print(f"✂️ Trimming to dates >= {cutoff.date()}...")
    lights = lights[lights["date"] >= cutoff].copy()

    # 4) Compute average radiance per area
    print("💡 Computing avg_rad_month = nlsum / area...")
    lights["avg_rad_month"] = lights["nlsum"] / lights["area"]

    # 5) Aggregate to a stable region-month panel
    print("📊 Aggregating by ['iso', 'id_1', 'name_1', 'id_2', 'name_2', 'date']...")
    group_cols = ["iso", "id_1", "name_1", "id_2", "name_2", "date"]
    lights_panel = (
        lights.groupby(group_cols, as_index=False)["avg_rad_month"]
        .mean()
    )

    # 6) Save to data/intermediate
    DATA_INTER.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTER / "lights_monthly_by_coord.csv"
    print(f"💾 Writing region-based lights panel to {out_path}...")
    lights_panel.to_csv(out_path, index=False)
    print("✅ Finished building lights_monthly_by_coord.csv (region-based, 2018+).")

    return lights_panel

