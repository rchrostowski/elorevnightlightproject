# src/preprocess_lights.py

import pandas as pd
from pathlib import Path

from .load_data import load_raw_lights
from .config import DATA_INTER


def build_lights_monthly_by_coord() -> pd.DataFrame:
    """
    Build a monthly nightlights panel from the 'level2' VIIRS file:

    Columns in raw file (from your CSV):
    ['iso', 'id_1', 'name_1', 'id_2', 'name_2', 'year', 'month', 'nlsum', 'area', 'date']

    We will:
    - Keep USA only (iso == 'USA')
    - Ensure a proper datetime 'date' column
    - Optionally trim to recent years (e.g. 2020+)
    - Compute brightness as nlsum / area
    - Aggregate by (iso, id_1, name_1, id_2, name_2, date)
    - Save to data/intermediate/lights_monthly_by_coord.csv

    NOTE: Despite the function name mentioning "coord", this file is
    region-based (admin level), not lat/lon grid based.
    """

    print("📥 Loading raw nightlights data...")
    lights = load_raw_lights().copy()

    # ---------------------------------------------------------
    # 1. Filter to USA only (you can remove this if you want global)
    # ---------------------------------------------------------
    if "iso" in lights.columns:
        print("🌎 Filtering to iso == 'USA'...")
        lights = lights[lights["iso"] == "USA"]

    if lights.empty:
        raise ValueError("Nightlights data is empty after filtering iso == 'USA'.")

    # ---------------------------------------------------------
    # 2. Ensure we have a proper datetime 'date' column
    # ---------------------------------------------------------
    print("🛠 Ensuring datetime 'date' column...")

    if "date" in lights.columns:
        lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
    elif "year" in lights.columns and "month" in lights.columns:
        lights["date"] = pd.to_datetime(
            lights["year"].astype(str) + "-" + lights["month"].astype(str),
            errors="coerce",
        )
    else:
        raise ValueError(
            "VIIRS file must have either 'date' or 'year'+'month' columns."
        )

    lights = lights.dropna(subset=["date"])

    # ---------------------------------------------------------
    # 3. Aggressive TRIM to recent years only (to keep pipeline light)
    #    Adjust CUTOFF_DATE if you want more/less history.
    # ---------------------------------------------------------
    CUTOFF_DATE = "2020-01-01"
    print(f"✂️ Trimming to dates >= {CUTOFF_DATE}...")
    lights = lights[lights["date"] >= CUTOFF_DATE]

    if lights.empty:
        raise ValueError(
            f"Nightlights data is empty after trimming to date >= {CUTOFF_DATE}."
        )

    # ---------------------------------------------------------
    # 4. Compute brightness measure from nlsum and area
    # ---------------------------------------------------------
    # Your file has 'nlsum' (sum of lights) and 'area' (area of polygon).
    # We'll define avg_rad_month = nlsum / area.
    if "nlsum" not in lights.columns or "area" not in lights.columns:
        raise ValueError(
            "Expected columns 'nlsum' and 'area' in VIIRS file. "
            f"Found columns: {lights.columns.tolist()}"
        )

    print("💡 Computing avg_rad_month = nlsum / area...")
    lights["avg_rad_month"] = lights["nlsum"] / lights["area"]

    # ---------------------------------------------------------
    # 5. Aggregate by region and date
    # ---------------------------------------------------------
    group_cols = ["iso", "id_1", "name_1", "id_2", "name_2", "date"]

    print(f"📊 Aggregating by {group_cols}...")
    grouped = (
        lights.groupby(group_cols, as_index=False)["avg_rad_month"]
        .mean()
    )

    # ---------------------------------------------------------
    # 6. Save to data/intermediate/lights_monthly_by_coord.csv
    #    (name kept for compatibility with rest of pipeline)
    # ---------------------------------------------------------
    output_path = DATA_INTER / "lights_monthly_by_coord.csv"

    print(f"💾 Writing region-based lights panel to {output_path}...")
    grouped.to_csv(output_path, index=False)

    print("✅ Finished building lights_monthly_by_coord.csv (region-based).")

    return grouped

