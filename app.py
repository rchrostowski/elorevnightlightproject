# app.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

# ---------- Page config ----------
st.set_page_config(
    page_title="Night Lights & Returns – Main Dashboard",
    layout="wide",
)

st.title("Night Lights & Stock Returns – Dashboard")

# ---------- Load data ----------
df = load_model_data(fallback_if_missing=True).copy()

if df.empty:
    st.error(
        "nightlights_model_data.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` to rebuild it and commit "
        "data/final/nightlights_model_data.csv."
    )
    st.stop()

# Basic cleaning
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df[df["date"].notna()]

# restrict to years where we have both lights & returns
mask = df[["brightness_change", "ret_fwd"]].notna().any(axis=1)
df_use = df[mask].copy()
df_use = df_use.sort_values("date")

# ---------- Top-level metrics ----------
n_firms = df_use["ticker"].nunique()
n_counties = df_use[["state_full", "county_name"]].drop_duplicates().shape[0]
date_min = df_use["date"].min()
date_max = df_use["date"].max()

corr = df_use["brightness_change"].corr(df_use["ret_fwd"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Number of firms", f"{n_firms}")
col2.metric("HQ counties", f"{n_counties}")
col3.metric("Date range", f"{date_min:%Y-%m} → {date_max:%Y-%m}")
col4.metric("Corr(ΔLight, next-month ret)", f"{corr:.3f}" if pd.notna(corr) else "N/A")

st.markdown(
    """
Below you have:

- **Overview** (this page): big-picture summary of the dataset  
- **Ticker Explorer**: focus on one firm’s HQ brightness vs its returns  
- **County Explorer**: zoom into one HQ county and see all firms there  
- **Globe**: interactive spinning map of HQ locations & hotspots  
- **Regression**: the main **Ret ~ ΔBrightness + month fixed effects** result
    """
)

# ---------- Time-series summary ----------
st.subheader("Average ΔBrightness and Average Next-Month Return Over Time")

group = df_use.groupby("date").agg(
    avg_dlight=("brightness_change", "mean"),
    avg_ret_fwd=("ret_fwd", "mean"),
).reset_index()

# Scale returns to percentages for plotting
group["avg_ret_fwd_pct"] = group["avg_ret_fwd"] * 100

import altair as alt

base = alt.Chart(group).encode(x="date:T")

line_dlight = base.mark_line().encode(
    y=alt.Y("avg_dlight:Q", title="Average ΔBrightness")
)

line_ret = base.mark_line(color="#4c78a8").encode(
    y=alt.Y("avg_ret_fwd_pct:Q", title="Avg next-month return (%)"),
).interactive()

st.altair_chart(
    alt.layer(
        line_dlight.encode(color=alt.value("#f58518")),
        line_ret,
    ).resolve_scale(y="independent"),
    use_container_width=True,
)

# ---------- Distribution ----------
st.subheader("Distribution of HQ Brightness Changes")

hist = (
    alt.Chart(df_use)
    .mark_bar()
    .encode(
        x=alt.X("brightness_change:Q", bin=alt.Bin(maxbins=50), title="ΔBrightness"),
        y=alt.Y("count()", title="Count"),
    )
)

st.altair_chart(hist, use_container_width=True)

st.caption(
    "Note: Returns used in this app are *raw monthly stock returns*, "
    "not market- or risk-adjusted excess returns. We highlight the relation "
    "between changes in local nightlights and future returns."
)


