# scripts/add_state_to_sp500.py

import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "raw" / "sp500_clean.csv"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "sp500_clean_with_state.csv"


# Approximate latitude/longitude for each US state centroid (degrees)
STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130),
    "AK": (64.200841, -149.493673),
    "AZ": (34.048928, -111.093731),
    "AR": (35.201050, -91.831833),
    "CA": (36.778261, -119.417932),
    "CO": (39.550051, -105.782067),
    "CT": (41.603221, -73.087749),
    "DE": (38.910832, -75.527670),
    "FL": (27.664827, -81.515754),
    "GA": (32.165622, -82.900075),
    "HI": (19.896766, -155.582782),
    "ID": (44.068202, -114.742041),
    "IL": (40.633125, -89.398528),
    "IN": (40.551217, -85.602364),
    "IA": (41.878003, -93.097702),
    "KS": (39.011902, -98.484246),
    "KY": (37.839333, -84.270018),
    "LA": (30.984298, -91.962333),
    "ME": (45.253783, -69.445469),
    "MD": (39.045755, -76.641271),
    "MA": (42.407211, -71.382437),
    "MI": (44.314844, -85.602364),
    "MN": (46.729553, -94.685900),
    "MS": (32.354668, -89.398528),
    "MO": (37.964253, -91.831833),
    "MT": (46.879682, -110.362566),
    "NE": (41.492537, -99.901813),
    "NV": (38.802610, -116.419389),
    "NH": (43.193852, -71.572395),
    "NJ": (40.058324, -74.405661),
    "NM": (34.519940, -105.870090),
    "NY": (43.299428, -74.217933),
    "NC": (35.759573, -79.019300),
    "ND": (47.551493, -101.002012),
    "OH": (40.417287, -82.907123),
    "OK": (35.467560, -97.516428),
    "OR": (43.804133, -120.554201),
    "PA": (41.203322, -77.194525),
    "RI": (41.580095, -71.477429),
    "SC": (33.836081, -81.163725),
    "SD": (43.969515, -99.901813),
    "TN": (35.517491, -86.580447),
    "TX": (31.968599, -99.901813),
    "UT": (39.320980, -111.093731),
    "VT": (44.558803, -72.577841),
    "VA": (37.431573, -78.656894),
    "WA": (47.751074, -120.740139),
    "WV": (38.597626, -80.454903),
    "WI": (43.784440, -88.787868),
    "WY": (43.075968, -107.290284),
    "DC": (38.907192, -77.036871),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two points on Earth (in kilometers).
    """
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def coord_to_state(lat, lon):
    """
    Roughly map (lat, lon) to the nearest US state centroid.
    If outside a broad US bounding box, return None.
    """
    # Rough bounding box for continental US (plus a bit of margin)
    if not (24.0 <= lat <= 50.5 and -125.0 <= lon <= -66.0):
        # Might be Alaska, Hawaii, or international HQ
        # Try AK/HI specifically, then give up if still far.
        candidates = ["AK", "HI"]
        best_state = None
        best_dist = float("inf")
        for st in candidates:
            slat, slon = STATE_CENTROIDS[st]
            dist = haversine_distance(lat, lon, slat, slon)
            if dist < best_dist:
                best_dist = dist
                best_state = st

        # If it's within 500km of AK/HI centroid, accept; else treat as non-US
        if best_dist < 500:
            return best_state
        return None

    # Continental US: pick nearest centroid
    best_state = None
    best_dist = float("inf")
    for st, (slat, slon) in STATE_CENTROIDS.items():
        dist = haversine_distance(lat, lon, slat, slon)
        if dist < best_dist:
            best_dist = dist
            best_state = st

    return best_state


def main():
    print(f"📥 Loading {IN_PATH} ...")
    df = pd.read_csv(IN_PATH)

    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected '{col}' column in sp500_clean.csv. "
                f"Found columns: {df.columns.tolist()}"
            )

    print("📍 Mapping lat/lon to nearest US state...")
    df["state"] = df.apply(lambda row: coord_to_state(row["lat"], row["lon"]), axis=1)

    missing = df["state"].isna().sum()
    if missing > 0:
        bad_rows = df[df["state"].isna()][["ticker", "company", "lat", "lon"]]
        print(f"⚠️ WARNING: {missing} firms could not be mapped to a US state (likely non-US HQs).")
        print(bad_rows.head())

    print(f"💾 Writing updated file with 'state' column to {OUT_PATH} ...")
    df.to_csv(OUT_PATH, index=False)
    print("✅ Done. Open sp500_clean_with_state.csv to inspect, then replace the original if you're happy.")


if __name__ == "__main__":
    main()
