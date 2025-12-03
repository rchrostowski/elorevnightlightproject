# pages/1_Overview.py

import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.set_page_config(
    page_title="Overview – Night Lights Anomalia",
    layout="wide",
)

# ---------- Helpers ----------

def _get_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


# ---------- Load & prep ----------

df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

brightness_col = _get_col(df, ["brightness_change", "d_light", "delta_light"], required=True)
level_col = _get_col(df, ["brightness_hq", "avg_rad_hq", "avg_rad_month", "light_level"])
ret_col = _get_col(df, ["ret_excess", "ret", "return"], required=True)
ret_fwd_col = _get_col(df, ["ret_fwd_1m", "ret_fwd", "ret_forward_1m"])

st.markdown("## Overview")

# ---------- KPI row ----------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Firm-months", f"{len(df):,}")
with c2:
    st.metric("Tickers", f"{df['ticker'].nunique():,}")
with c3:
    st.metric("HQ counties", f"{df['county_name'].nunique():,}")
with c4:
    st.metric("Date range", f"{df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}")

# ---------- Average time-series ----------

st.markdown("### Cross-sectional averages over time")

agg = (
    df.groupby("date")
    .agg(
        mean_dlight=(brightness_col, "mean"),
        mean_ret=(ret_col, "mean"),
    )
    .reset_index()
)

if ret_fwd_col:
    agg["mean_ret_fwd"] = df.groupby("date")[ret_fwd_col].mean().values

col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    fig1 = px.line(
        agg,
        x="date",
        y="mean_dlight",
        labels={"date": "Date", "mean_dlight": "Avg ΔLight (HQ county)"},
        title="Average ΔLight across firms (by calendar month)",
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_ts2:
    y_cols = ["mean_ret"]
    labels = {"mean_ret": "Avg same-month return"}
    if "mean_ret_fwd" in agg.columns:
        y_cols.append("mean_ret_fwd")
        labels["mean_ret_fwd"] = "Avg next-month return"

    fig2 = px.line(
        agg,
        x="date",
        y=y_cols,
        labels={"date": "Date", **labels},
        title="Average returns over time",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Scatter ΔLight vs next-month return ----------

st.markdown("### ΔLight vs next-month return (cross-sectional)")

if not ret_fwd_col:
    st.info("Next-month return column not found – skipping this plot.")
else:
    sample = df[[brightness_col, ret_fwd_col]].dropna()
    if sample.empty:
        st.info("Not enough non-missing observations to plot.")
    else:
        fig_sc = px.scatter(
            sample.sample(min(len(sample), 8000), random_state=42),
            x=brightness_col,
            y=ret_fwd_col,
            trendline="ols",
            labels={
                brightness_col: "ΔLight (change in HQ county brightness)",
                ret_fwd_col: "Next-month return",
            },
            title="Does a brighter HQ county predict higher next-month returns?",
            opacity=0.6,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ---------- Distribution panel ----------

st.markdown("### Distributions")

c_hist1, c_hist2 = st.columns(2)

with c_hist1:
    fig_h1 = px.histogram(
        df,
        x=brightness_col,
        nbins=60,
        title="Distribution of ΔLight across firm-months",
        labels={brightness_col: "ΔLight (HQ county)"},
    )
    st.plotly_chart(fig_h1, use_container_width=True)

with c_hist2:
    fig_h2 = px.histogram(
        df,
        x=ret_fwd_col if ret_fwd_col else ret_col,
        nbins=60,
        title="Distribution of next-month returns" if ret_fwd_col else "Distribution of returns",
        labels={ret_fwd_col if ret_fwd_col else ret_col: "Return"},
    )
    st.plotly_chart(fig_h2, use_container_width=True)

st.caption(
    "This overview page is purely descriptive. The causal / predictive story is tested formally in the Regression page."
)


