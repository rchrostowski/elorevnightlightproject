# pages/3_County_Explorer.py

import pandas as pd
import streamlit as st
import altair as alt

from src.load_data import load_model_data

st.set_page_config(page_title="County Explorer", page_icon="🗺️")

st.title("🗺️ County / Area Explorer")

df = load_model_data(fallback_if_missing=True).copy()
df.columns = [c.strip().lower() for c in df.columns]

if "date" not in df.columns or "ret" not in df.columns:
    st.error("Dataset must have 'date' and 'ret' columns.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year_month"] = df["date"].dt.to_period("M").astype(str)

# Use county-level info if present, else state
area_col = None
for cand in ["county_name", "name_2", "state"]:
    if cand in df.columns:
        area_col = cand
        break

if area_col is None:
    st.error("No county/state column found (e.g. 'county_name', 'name_2', 'state').")
    st.stop()

brightness_col = None
for cand in ["brightness", "avg_rad_month", "avg_brightness"]:
    if cand in df.columns:
        brightness_col = cand
        break

if brightness_col is None:
    st.error("No brightness column found (e.g. 'brightness', 'avg_rad_month').")
    st.stop()

with st.sidebar:
    st.header("Filters")

    # pick a single calendar month to avoid mixing seasonality
    ym_options = sorted(df["year_month"].dropna().unique())
    ym = st.selectbox("Year-Month (fixed calendar month)", ym_options)

    # optional: focus on a subset of areas
    areas = sorted(df[area_col].dropna().unique())
    picked_areas = st.multiselect(
        f"{area_col} filter (optional)",
        options=areas,
        default=areas[:20],  # first 20 as default
    )

# filter to that month
df_m = df[df["year_month"] == ym].copy()
if picked_areas:
    df_m = df_m[df_m[area_col].isin(picked_areas)]

# aggregate at county/month level
g = (
    df_m.groupby(area_col, as_index=False)
    .agg(
        mean_ret=("ret", "mean"),
        mean_brightness=(brightness_col, "mean"),
    )
)

if g.empty:
    st.warning("No data for this selection.")
    st.stop()

st.markdown(
    f"""
For **{ym}**, each point below is one **{area_col}**, with:

- X-axis: average brightness in that area  
- Y-axis: average stock return (ret) for firms mapped to that area  
- This is basically the cross-sectional relationship the regression is using.
"""
)

chart = (
    alt.Chart(g)
    .mark_circle(size=60, opacity=0.7)
    .encode(
        x=alt.X("mean_brightness:Q", title="Brightness (mean in area for this month)"),
        y=alt.Y("mean_ret:Q", title="Monthly return (ret)"),
        tooltip=[area_col, "mean_brightness:Q", "mean_ret:Q"],
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)


