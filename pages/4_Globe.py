# pages/4_Globe.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.load_data import load_model_data

st.title("HQ Globe – Night Lights & Returns")

df = load_model_data(fallback_if_missing=True).copy()
if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# keep rows where we have coordinates
df = df.dropna(subset=["lat", "lon"])

# Sidebar: month selection
st.sidebar.header("Globe controls")
unique_dates = sorted(df["date"].dt.to_period("M").unique())
unique_ts = [pd.Period(p, freq="M").to_timestamp() for p in unique_dates]

default_idx = len(unique_ts) - 1
selected_month = st.sidebar.selectbox(
    "Month",
    options=unique_ts,
    index=default_idx,
    format_func=lambda d: d.strftime("%Y-%m"),
)

# Filter to that month
df_m = df[df["date"].dt.to_period("M") == selected_month.to_period("M")].copy()
if df_m.empty:
    st.warning("No data for this month.")
    st.stop()

# Aggregate per HQ (ticker)
agg = (
    df_m.groupby(["ticker", "firm", "state_full", "county_name", "lat", "lon"], as_index=False)
    .agg(
        brightness=("avg_rad_month", "mean"),
        dlight=("brightness_change", "mean"),
        ret_fwd=("ret_fwd", "mean"),
    )
)

# Use ret_fwd as main visual driver, fallback to dlight if missing
use_ret = agg["ret_fwd"].notna().any()
if use_ret:
    metric = agg["ret_fwd"].fillna(0.0)
    metric_label = "Next-month return"
else:
    metric = agg["dlight"].fillna(0.0)
    metric_label = "ΔBrightness"

max_abs = metric.abs().max()
if max_abs == 0 or pd.isna(max_abs):
    norm = pd.Series(0.5, index=metric.index)
else:
    norm = (metric / max_abs + 1) / 2.0  # map [-max, max] → [0, 1]

marker_sizes = 5 + 25 * norm
marker_intensity = norm

hover_text = (
    agg["ticker"]
    + " – "
    + agg["firm"]
    + "<br>"
    + agg["county_name"].astype(str)
    + ", "
    + agg["state_full"].astype(str)
    + "<br>"
    + metric_label
    + ": "
    + metric.round(3).astype(str)
)

fig = go.Figure()

fig.add_trace(
    go.Scattergeo(
        lon=agg["lon"],
        lat=agg["lat"],
        text=hover_text,
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_intensity,
            colorscale=[
                [0.0, "rgb(2, 6, 23)"],
                [0.3, "rgb(13, 37, 88)"],
                [0.6, "rgb(37, 99, 235)"],
                [1.0, "rgb(191, 219, 254)"],
            ],
            cmin=0,
            cmax=1,
            opacity=0.95,
        ),
    )
)

fig.update_geos(
    projection_type="orthographic",
    projection_rotation=dict(lon=-95, lat=35, roll=0),
    showcountries=True,
    showcoastlines=False,
    showland=True,
    landcolor="rgb(0, 0, 0)",
    showocean=True,
    oceancolor="rgb(0, 0, 0)",
    bgcolor="rgba(0, 0, 0, 0)",
)

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
)

left, right = st.columns([3, 1.2])

with left:
    st.plotly_chart(fig, use_container_width=True, height=650)

with right:
    st.markdown(f"### {selected_month:%Y-%m} summary")

    avg_metric = metric.mean()
    st.metric(f"Avg {metric_label}", f"{avg_metric:.3f}")

    st.markdown("#### Top 5 (positive)")
    top_pos = agg.sort_values("ret_fwd" if use_ret else "dlight", ascending=False).head(5)
    st.table(
        top_pos[["ticker", "firm", "county_name", "state_full", "ret_fwd" if use_ret else "dlight"]]
    )

    st.markdown("#### Bottom 5 (negative)")
    top_neg = agg.sort_values("ret_fwd" if use_ret else "dlight", ascending=True).head(5)
    st.table(
        top_neg[["ticker", "firm", "county_name", "state_full", "ret_fwd" if use_ret else "dlight"]]
    )

st.caption(
    f"The globe shows firm HQ locations for {selected_month:%Y-%m}. "
    f"Marker size and brightness reflect {metric_label.lower()}."
)
