# pages/1_Overview.py

import streamlit as st
import pandas as pd
import altair as alt

from src.load_data import load_model_data

st.title("Overview")

df = load_model_data(fallback_if_missing=True).copy()

if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Filter to rows with at least one of brightness_change / ret_fwd
mask = df[["brightness_change", "ret_fwd"]].notna().any(axis=1)
df = df[mask].sort_values("date")

# Sidebar filters
st.sidebar.header("Filters")
min_year = int(df["date"].dt.year.min())
max_year = int(df["date"].dt.year.max())

year_range = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

df = df[(df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])]

# Metrics
n_firms = df["ticker"].nunique()
n_counties = df[["state_full", "county_name"]].drop_duplicates().shape[0]
date_min = df["date"].min()
date_max = df["date"].max()

corr = df["brightness_change"].corr(df["ret_fwd"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Firms", f"{n_firms}")
c2.metric("HQ counties", f"{n_counties}")
c3.metric("Date range", f"{date_min:%Y-%m} → {date_max:%Y-%m}")
c4.metric("Corr(ΔLight, next-month ret)", f"{corr:.3f}" if pd.notna(corr) else "N/A")

st.markdown("### Average across all firms & HQ counties")

group = df.groupby("date").agg(
    avg_brightness=("avg_rad_month", "mean"),
    avg_dlight=("brightness_change", "mean"),
    avg_ret=("ret", "mean"),
    avg_ret_fwd=("ret_fwd", "mean"),
).reset_index()

group["avg_ret_fwd_pct"] = group["avg_ret_fwd"] * 100

base = alt.Chart(group).encode(x="date:T")

chart1 = base.mark_line(color="#f58518").encode(
    y=alt.Y("avg_dlight:Q", title="Average ΔBrightness")
)

chart2 = base.mark_line(color="#4c78a8").encode(
    y=alt.Y("avg_ret_fwd_pct:Q", title="Average next-month return (%)"),
)

st.altair_chart(
    alt.layer(chart1, chart2).resolve_scale(y="independent"),
    use_container_width=True,
)

st.markdown("### Cross-section: ΔBrightness vs next-month returns")

scatter = (
    alt.Chart(df)
    .mark_circle(opacity=0.35)
    .encode(
        x=alt.X("brightness_change:Q", title="ΔBrightness (county HQ)"),
        y=alt.Y("ret_fwd:Q", title="Next-month return"),
        color=alt.Color("date:T", legend=None),
        tooltip=["ticker", "firm", "state_full", "county_name", "date", "brightness_change", "ret_fwd"],
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

st.caption(
    "This overview page shows how changes in nighttime lights around firm HQs "
    "line up with *next-month* stock returns, aggregated across the whole sample."
)
