# pages/2_Ticker_Explorer.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.load_data import load_model_data

st.set_page_config(page_title="Ticker Explorer", page_icon="📈")

st.title("📈 Ticker Explorer")

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = load_model_data(fallback_if_missing=True).copy()
df.columns = [c.strip().lower() for c in df.columns]

if "date" not in df.columns:
    st.error("Dataset is missing 'date' column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Assume 'ret' is your **excess return** measure.
# If it's raw returns, just talk about it as 'monthly return' in class.
required = {"ticker", "date", "ret"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns in dataset: {missing}")
    st.stop()

# Use brightness column if present
brightness_col = None
for cand in ["brightness", "avg_rad_month", "avg_brightness"]:
    if cand in df.columns:
        brightness_col = cand
        break

# Compute forward return if needed
if "ret_fwd" not in df.columns:
    df = df.sort_values(["ticker", "date"])
    df["ret_fwd"] = df.groupby("ticker")["ret"].shift(-1)

# Year-month factor for regression tab later if you want
df["year_month"] = df["date"].dt.to_period("M").astype(str)

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    tickers = sorted(df["ticker"].dropna().unique())
    ticker = st.selectbox("Ticker", tickers)

    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

start_date, end_date = [pd.to_datetime(d) for d in date_range]

df_t = df[(df["ticker"] == ticker) & df["date"].between(start_date, end_date)].copy()
df_t = df_t.sort_values("date")

if df_t.empty:
    st.warning("No data for this ticker in the selected date range.")
    st.stop()

# -------------------------------------------------------------------
# Summary metrics
# -------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Obs", len(df_t))
with c2:
    st.metric("Avg monthly return", f"{df_t['ret'].mean():.2%}")
with c3:
    if df_t["ret_fwd"].notna().any():
        st.metric("Avg next-month return", f"{df_t['ret_fwd'].mean():.2%}")
    else:
        st.metric("Avg next-month return", "n/a")

# -------------------------------------------------------------------
# 1) Time series: **x = date, y = return**
# -------------------------------------------------------------------
st.subheader(f"Returns over time: {ticker}")

ts_chart = (
    alt.Chart(df_t)
    .mark_line()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("ret:Q", title="Monthly return (ret)"),
        tooltip=["date:T", "ret:Q"],
    )
    .properties(height=300)
)

st.altair_chart(ts_chart, use_container_width=True)

# -------------------------------------------------------------------
# 2) Brightness vs return: **y = returns, x = brightness**
# -------------------------------------------------------------------
if brightness_col is not None:
    st.subheader(f"Brightness vs return: {ticker}")
    st.caption(
        "Y-axis is the stock's monthly return. "
        "X-axis is localized nightlights brightness near HQ."
    )

    scatter_df = df_t[[brightness_col, "ret"]].dropna()
    if not scatter_df.empty:
        scat = (
            alt.Chart(scatter_df)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X(f"{brightness_col}:Q", title="Brightness (avg_rad_month)"),
                y=alt.Y("ret:Q", title="Monthly return (ret)"),
                tooltip=[f"{brightness_col}:Q", "ret:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(scat, use_container_width=True)
    else:
        st.info("No non-missing brightness data to plot for this ticker.")

# -------------------------------------------------------------------
# 3) Simple within-ticker correlation table
# -------------------------------------------------------------------
st.subheader("Correlation summary (this ticker)")

rows = []

if df_t["ret_fwd"].notna().sum() > 2:
    rows.append(["ret vs next-month ret", df_t["ret"].corr(df_t["ret_fwd"])])

if brightness_col is not None:
    tmp = df_t[[brightness_col, "ret"]].dropna()
    if tmp.shape[0] > 2:
        rows.append([f"{brightness_col} vs same-month ret", tmp[brightness_col].corr(tmp["ret"])])

    tmp2 = df_t[[brightness_col, "ret_fwd"]].dropna()
    if tmp2.shape[0] > 2:
        rows.append([f"{brightness_col} vs next-month ret", tmp2[brightness_col].corr(tmp2["ret_fwd"])])

if rows:
    corr_df = pd.DataFrame(rows, columns=["Pair", "Correlation"])
    corr_df["Correlation"] = corr_df["Correlation"].round(3)
    st.dataframe(corr_df, use_container_width=True)
else:
    st.info("Not enough data to compute correlations for this ticker.")



