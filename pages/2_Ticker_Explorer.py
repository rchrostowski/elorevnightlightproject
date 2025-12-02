# pages/2_Ticker_Explorer.py

import numpy as np
import pandas as pd
import streamlit as st

from src.load_data import load_model_data

st.set_page_config(page_title="Ticker Explorer", page_icon="📈")

st.title("📈 Ticker Explorer")

# -------------------------------------------------------------------
# Load data and normalize columns
# -------------------------------------------------------------------
df = load_model_data(fallback_if_missing=True).copy()

# normalize colnames
df.columns = [c.strip().lower() for c in df.columns]

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

required = {"ticker", "date", "ret"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns in dataset: {missing}")
    st.stop()

# -------------------------------------------------------------------
# Ensure ret_fwd exists (compute if missing)
# -------------------------------------------------------------------
if "ret_fwd" not in df.columns:
    df = df.sort_values(["ticker", "date"])
    df["ret_fwd"] = df.groupby("ticker")["ret"].shift(-1)
    # You can choose to drop the last row per ticker where ret_fwd is NaN,
    # but we'll just let it be NaN for plotting / stats.
    
# Optional: make sure brightness column is named consistently
brightness_col = None
for cand in ["brightness", "avg_rad_month", "avg_brightness"]:
    if cand in df.columns:
        brightness_col = cand
        break

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

# Filter by ticker + date range
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

df_t = df[(df["ticker"] == ticker) & (df["date"].between(start_date, end_date))].copy()
df_t = df_t.sort_values("date")

if df_t.empty:
    st.warning("No data for this ticker in the selected date range.")
    st.stop()

# -------------------------------------------------------------------
# Summary metrics
# -------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Obs",
        len(df_t),
    )

with col2:
    avg_ret = df_t["ret"].mean()
    st.metric("Avg monthly return", f"{avg_ret:.2%}")

with col3:
    if df_t["ret_fwd"].notna().any():
        avg_fwd = df_t["ret_fwd"].mean()
        st.metric("Avg next-month return", f"{avg_fwd:.2%}")
    else:
        st.metric("Avg next-month return", "n/a")

# -------------------------------------------------------------------
# Time-series plots
# -------------------------------------------------------------------
st.subheader(f"Returns over time for {ticker}")

ret_chart = df_t.set_index("date")[["ret", "ret_fwd"]]
st.line_chart(ret_chart)

if brightness_col is not None:
    st.subheader(f"Brightness vs. returns for {ticker}")
    st.caption(
        "Brightness is localized (county-level) nightlights near HQ; "
        "this is the same brightness used in the regression."
    )

    # Simple scatter: brightness vs same-month return
    scatter_df = df_t[[brightness_col, "ret"]].dropna()
    if not scatter_df.empty:
        st.scatter_chart(scatter_df.rename(columns={brightness_col: "brightness"}))
    else:
        st.info("No non-missing brightness data available for this ticker.")

# -------------------------------------------------------------------
# Correlation summary
# -------------------------------------------------------------------
st.subheader("Correlation summary")

rows = []

# corr(ret, ret_fwd)
if df_t["ret_fwd"].notna().sum() > 2:
    corr_ret_ret_fwd = df_t["ret"].corr(df_t["ret_fwd"])
    rows.append(["ret vs next-month ret", corr_ret_ret_fwd])

# corr(brightness, ret) and brightness vs ret_fwd if available
if brightness_col is not None:
    if df_t[[brightness_col, "ret"]].dropna().shape[0] > 2:
        corr_b_ret = df_t[brightness_col].corr(df_t["ret"])
        rows.append([f"{brightness_col} vs same-month ret", corr_b_ret])

    if df_t["ret_fwd"].notna().sum() > 2:
        tmp = df_t[[brightness_col, "ret_fwd"]].dropna()
        if tmp.shape[0] > 2:
            corr_b_ret_fwd = tmp[brightness_col].corr(tmp["ret_fwd"])
            rows.append([f"{brightness_col} vs next-month ret", corr_b_ret_fwd])

if rows:
    corr_df = pd.DataFrame(rows, columns=["Pair", "Correlation"])
    corr_df["Correlation"] = corr_df["Correlation"].round(3)
    st.dataframe(corr_df, use_container_width=True)
else:
    st.info("Not enough data to compute correlations in this window.")


