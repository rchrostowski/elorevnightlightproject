# src/build_panel.py

import numpy as np
import pandas as pd

from .load_data import (
    load_sp500_clean,
    load_lights_monthly_by_coord,
)


def _assign_nearest_county(sp500: pd.DataFrame, counties: pd.DataFrame) -> pd.DataFrame:
    """
    For each firm HQ (lat, lon) in sp500, assign the nearest county
    based on county centroids (lat_round, lon_round).

    Adds:
        - county_id_2 : the county id_2 used in lights
        - county_name : human-readable county name (name_2)
    """
    # Drop any weird firms without coordinates
    sp = sp500.dropna(subset=["lat", "lon"]).copy()

    if sp.empty or counties.empty:
        raise ValueError(
            "No valid firms or county centroids to match. "
            "Check that sp500_clean has lat/lon and lights_monthly_by_coord has lat_round/lon_round."
        )

    county_coords = counties[["lat_round", "lon_round"]].to_numpy()
    firm_coords = sp[["lat", "lon"]].to_numpy()

    nearest_ids = []
    nearest_names = []

    for (flat, flon) in firm_coords:
        # squared Euclidean distance in lat/lon space (good enough for this use case)
        d2 = (county_coords[:, 0] - flat) ** 2 + (county_coords[:, 1] - flon) ** 2
        idx = int(np.argmin(d2))
        nearest_ids.append(counties.iloc[idx]["id_2"])
        nearest_names.append(counties.iloc[idx]["name_2"])

    sp["county_id_2"] = nearest_ids
    sp["county_name"] = nearest_names

    # Put county info back into the full sp500 (in case there were NaNs)
    result = sp500.copy()
    result = result.merge(
        sp[["ticker", "county_id_2", "county_name"]],
        on="ticker",
        how="left",
    )

    return result


def build_panel_firms_with_brightness() -> pd.DataFrame:
    """
    Build a firm × month panel with localized brightness.

    Steps:
        1. Load sp500_clean (tickers + HQ lat/lon)
        2. Load county-level nightlights (2018+)
        3. Assign each firm to nearest county by distance
        4. Merge to get brightness per firm-month

    Returns a DataFrame with at least:
        ['ticker', 'company', 'date',
         'county_id_2', 'county_name',
         'avg_rad_month', 'brightness',
         'lat_round', 'lon_round', ...]
    """
    print("🚧 Building firm × month brightness panel (county-level)...")

    # 1) Firms
    sp500 = load_sp500_clean().copy()
    if not {"ticker", "lat", "lon"}.issubset(sp500.columns):
        raise ValueError(
            "sp500_clean must have columns ['ticker','lat','lon']. "
            f"Found: {list(sp500.columns)}"
        )

    print(f"🏢 Loaded {len(sp500)} firms from sp500_clean.csv")

    # 2) County-level lights
    lights = load_lights_monthly_by_coord().copy()
    expected_l_cols = {"id_2", "name_2", "date", "avg_rad_month", "lat_round", "lon_round"}
    if not expected_l_cols.issubset(lights.columns):
        raise ValueError(
            "lights_monthly_by_coord.csv must have columns "
            f"{expected_l_cols}. Found: {list(lights.columns)}"
        )

    lights["date"] = pd.to_datetime(lights["date"])
    print(
        "💡 lights_monthly_by_coord time window:",
        lights["date"].min(),
        "→",
        lights["date"].max(),
    )

    # Unique county centroids from lights
    counties = (
        lights[["id_2", "name_2", "lat_round", "lon_round"]]
        .dropna(subset=["lat_round", "lon_round"])
        .drop_duplicates("id_2")
        .reset_index(drop=True)
    )
    print(f"🗺 Using {len(counties)} distinct counties for nearest-HQ mapping")

    # 3) Assign nearest county for each firm HQ
    sp500_with_county = _assign_nearest_county(sp500, counties)

    missing_county = sp500_with_county["county_id_2"].isna().sum()
    if missing_county > 0:
        print(
            f"⚠️ Warning: {missing_county} firms could not be matched to any county. "
            "They will be dropped from the brightness panel."
        )

    sp500_with_county = sp500_with_county.dropna(subset=["county_id_2"]).copy()

    # 4) Merge firms with county-month brightness
    panel = sp500_with_county.merge(
        lights[
            [
                "id_2",
                "date",
                "avg_rad_month",
                "lat_round",
                "lon_round",
            ]
        ],
        left_on="county_id_2",
        right_on="id_2",
        how="inner",
    )

    panel = panel.rename(columns={"id_2": "county_fips"})

    # Alias for features / plots
    panel["brightness"] = panel["avg_rad_month"]

    print(f"✅ Built firm × month brightness panel with {len(panel)} rows")
    print(
        "   Date range:",
        panel["date"].min(),
        "→",
        panel["date"].max(),
    )
    print(
        "   Example columns:",
        [c for c in panel.columns if c in ("ticker", "company", "date", "county_name", "brightness")],
    )

    return panel
