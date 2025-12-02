# pages/4_Globe.py

import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_lights_monthly_by_coord

st.title("🌎 Nightlights by State")

# ---------------------------------------------------------
# 1. Load preprocessed lights panel
# ---------------------------------------------------------
df = load_lights_monthly_by_coord(fallback_if_missing=True)

if df.empty:
    st.error(
        "lights_monthly_by_coord.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` to rebuild it, commit the CSV in "
        "`data/intermediate/`, and redeploy."
    )
    st.stop()

required_cols = {"iso", "name_1", "date", "avg_rad_month"}
missing = required_cols - set(df.columns)
if missing:
    st.error(
        f"Missing columns in lights_monthly_by_coord.csv: {missing}\n\n"
        f"Found columns: {df.columns.tolist()}"
    )
    st.stop()

# Keep only USA and clean dates
df = df[df["iso"] == "USA"].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

if df.empty:
    st.error("No valid USA + date rows in lights panel after cleaning.")
    st.stop()

# ---------------------------------------------------------
# 2. Map state names -> postal abbreviations for choropleth
# ---------------------------------------------------------
STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Puerto Rico": "PR",
}

# name_1 is the first admin level name (e.g. state)
df["state_name"] = df["name_1"].astype(str)
df["state"] = df["state_name"].map(STATE_ABBR)

df = df.dropna(subset=["state"])

if df.empty:
    st.error(
        "Could not map any state names in `name_1` to US postal codes. "
        "Check that `name_1` contains US state names like 'California', "
        "'Texas', etc."
    )
    st.stop()

# ---------------------------------------------------------
# 3. Choose month to visualize
# ---------------------------------------------------------
st.sidebar.header("Globe / Map Filters")

unique_dates = sorted(df["date"].unique())
default_date = unique_dates[-1]  # latest month

selected_date = st.sidebar.selectbox(
    "Select month:",
    options=unique_dates,
    index=len(unique_dates) - 1,
    format_func=lambda d: d.strftime("%Y-%m"),
)

df_month = df[df["date"] == selected_date].copy()

if df_month.empty:
    st.warning(
        f"No data for selected month {selected_date.strftime('%Y-%m')}. "
        "Try another month."
    )
    st.stop()

# Aggregate to one value per state for that month
state_df = (
    df_month.groupby(["state", "state_name"], as_index=False)["avg_rad_month"]
    .mean()
)

st.caption(
    f"Average nighttime lights by state for {selected_date.strftime('%Y-%m')} "
    f"({len(state_df)} states)."
)

# ---------------------------------------------------------
# 4. Plot choropleth over USA
# ---------------------------------------------------------
fig = px.choropleth(
    state_df,
    locations="state",
    locationmode="USA-states",
    color="avg_rad_month",
    hover_name="state_name",
    scope="usa",
)

st.plotly_chart(fig, use_container_width=True)


