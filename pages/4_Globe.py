# pages/4_Globe.py

import streamlit as st
import plotly.express as px
import numpy as np

from src.load_data import load_lights_monthly_by_coord

st.title("🌍 Nightlights Globe")

# Load lights grid
lights = load_lights_monthly_by_coord().copy()

required = {"lat_round", "lon_round", "date", "avg_rad_month"}
missing = required - set(lights.columns)
if missing:
    st.error(f"Missing columns in lights_monthly_by_coord.csv: {missing}")
    st.stop()

lights["date"] = lights["date"].astype("datetime64[ns]")

# Sidebar controls
st.sidebar.header("Globe Controls")

min_date, max_date = lights["date"].min(), lights["date"].max()
selected_date = st.sidebar.slider(
    "Month",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=max_date.to_pydatetime(),
    format="YYYY-MM",
)

# Filter to that month (same year & month)
mask_month = (
    (lights["date"].dt.year == selected_date.year)
    & (lights["date"].dt.month == selected_date.month)
)
grid = lights[mask_month].copy()

if grid.empty:
    st.warning("No data for this month.")
    st.stop()

# Option: keep only the brightest points to avoid overplotting
quantile = st.sidebar.slider(
    "Show top X% brightest points",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
)

thresh = grid["avg_rad_month"].quantile(1 - quantile / 100)
grid = grid[grid["avg_rad_month"] >= thresh].copy()

# Normalize brightness for marker size
grid["marker_size"] = np.interp(
    grid["avg_rad_month"],
    (grid["avg_rad_month"].min(), grid["avg_rad_month"].max()),
    (3, 12),
)

st.caption(
    f"Showing {len(grid):,} points for "
    f"{selected_date.strftime('%Y-%m')} (top {quantile}% brightest)."
)

fig = px.scatter_geo(
    grid,
    lat="lat_round",
    lon="lon_round",
    size="marker_size",
    color="avg_rad_month",
    color_continuous_scale="viridis",
    projection="orthographic",
    hover_name=None,
    hover_data={
        "lat_round": True,
        "lon_round": True,
        "avg_rad_month": True,
        "marker_size": False,
    },
)

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar_title="Brightness",
)

st.plotly_chart(fig, use_container_width=True)

