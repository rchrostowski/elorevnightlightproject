# pages/4_Globe.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.load_data import load_lights_monthly_by_coord

st.set_page_config(
    page_title="Night Lights Anomalia Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #050710;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='margin-bottom: 1rem;'>Night Lights Anomalia Dashboard</h1>",
    unsafe_allow_html=True,
)

# ------------------ LOAD COUNTY LIGHTS ------------------

lights = load_lights_monthly_by_coord(fallback_if_missing=True)
if lights.empty:
    st.error(
        "lights_monthly_by_coord.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` and commit "
        "`data/intermediate/lights_monthly_by_coord.csv`."
    )
    st.stop()

lights = lights.copy()
lights.columns = [c.strip().lower() for c in lights.columns]

required = {"iso", "date", "avg_rad_month"}
if not required.issubset(lights.columns):
    st.error(
        f"lights_monthly_by_coord.csv must have columns {required}.\n"
        f"Found: {lights.columns.tolist()}"
    )
    st.stop()

# lat / lon columns from the county-centroid build
lat_col = None
lon_col = None
for cand in ["lat_round", "lat"]:
    if cand in lights.columns:
        lat_col = cand
        break
for cand in ["lon_round", "lon"]:
    if cand in lights.columns:
        lon_col = cand
        break

if lat_col is None or lon_col is None:
    st.error(
        "lights_monthly_by_coord.csv must have county coordinates "
        "(lat_round / lon_round). Rebuild the panel with "
        "`build_lights_monthly_by_coord`."
    )
    st.stop()

# Clean and restrict to USA
lights = lights[lights["iso"].str.upper() == "USA"].copy()
lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
lights = lights.dropna(subset=["date", lat_col, lon_col, "avg_rad_month"])

lights["year_month"] = lights["date"].dt.to_period("M").astype(str)

# ------------------ SIDEBAR ------------------

st.sidebar.header("Globe controls")

ym_options = sorted(lights["year_month"].unique())
selected_ym = st.sidebar.selectbox("Year-month", ym_options, index=len(ym_options) - 1)

st.sidebar.caption("Trim extreme brightness outliers")
p_low, p_high = st.sidebar.slider(
    "Brightness percentile range",
    0.0,
    100.0,
    (1.0, 99.0),
    step=1.0,
)

# ------------------ FILTER FOR MONTH ------------------

df_m = lights[lights["year_month"] == selected_ym].copy()
if df_m.empty:
    st.warning(f"No county lights for {selected_ym}.")
    st.stop()

b = df_m["avg_rad_month"]
low, high = np.percentile(b, [p_low, p_high])
df_m = df_m[df_m["avg_rad_month"].between(low, high)]

if df_m.empty:
    st.warning("No points after percentile filter.")
    st.stop()

df_m = df_m.rename(columns={lat_col: "lat", lon_col: "lon"})

# ------------------ GLOBE FIGURE ------------------

fig = px.scatter_geo(
    df_m,
    lat="lat",
    lon="lon",
    color="avg_rad_month",
    size="avg_rad_month",
    color_continuous_scale="ice",  # dark blue → light blue
    size_max=7,
    projection="orthographic",
    hover_data={"avg_rad_month": ":.4f", "lat": False, "lon": False},
)

fig.update_layout(
    geo=dict(
        showland=False,
        showcountries=False,
        showcoastlines=False,
        showlakes=False,
        showocean=True,
        oceancolor="rgb(0,0,0)",
    ),
    paper_bgcolor="rgb(0,0,0)",
    plot_bgcolor="rgb(0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    height=650,
)

fig.update_geos(
    resolution=50,
    lataxis_showgrid=False,
    lonaxis_showgrid=False,
)

left, right = st.columns([3.2, 1.2])

with left:
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Month summary")
    st.markdown(f"**Selected month:** {selected_ym}")

    avg_b = df_m["avg_rad_month"].mean()
    st.metric("Avg brightness", f"{avg_b:.3f}")

    st.markdown("#### Brightest counties (by avg_rad_month)")
    name_col = "name_2" if "name_2" in df_m.columns else None
    cols = ["avg_rad_month"]
    if name_col:
        cols = [name_col] + cols

    top = (
        df_m.sort_values("avg_rad_month", ascending=False)
        .head(10)[cols]
        .rename(columns={"name_2": "County", "avg_rad_month": "Brightness"})
    )
    st.table(top)

st.caption(
    "Globe uses VIIRS county-level night-lights (avg_rad_month) for the U.S. "
    "Color and size both scale with brightness. Drag to rotate."
)




