# pages/1_Overview.py

import streamlit as st
import pandas as pd
import numpy as np

from src.load_data import load_model_data

st.title("📊 Overview: Nightlights & Returns")

df = load_model_data(fallback_if_missing=False)

if df.empty:
    st.error(
        "Final model data is empty or missing.\n\n"
        "Run `python scripts/build_all.py` locally or in Codespaces to generate "
        "`data/final/nightlights_model_data.csv`, commit it, and redeploy."
    )
    st.stop()

# Basic cleaning / types
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"])

# --- Sidebar / filters -------------------------------------------------
st.sidebar.header("Filters")

# Ticker filter
tickers = sorted(df["ticker"].unique()) if "ticker" in df.columns else []
selected_tickers = st.sidebar.multiselect(
    "Select tickers (leave empty for all):",
    options=tickers,
    default=[],
)

# Date range filter
min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.slider(
    "Date range:",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
)

mask = (df["date"] >= pd.to_datetime(date_range[0])) & (
    df["date"] <= pd.to_datetime(date_range[1])
)

if selected_tickers:
    mask &= df["ticker"].isin(selected_tickers)

df_filt = df[mask].copy()

st.caption(
    f"Filtered to **{len(df_filt):,}** rows "
    f"({df_filt['ticker'].nunique() if 'ticker' in df_filt.columns else 0} tickers) "
    f"from {df_filt['date'].min().date()} to {df_filt['date'].max().date()}."
)

# --- High-level metrics ------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Tickers", df_filt["ticker"].nunique() if "ticker" in df_filt.columns else 0)

with col2:
    if "state_name" in df_filt.columns:
        st.metric("States", df_filt["state_name"].nunique())
    else:
        st.metric("States", "—")

with col3:
    if "ret" in df_filt.columns:
        avg_ret = df_filt["ret"].mean()
        st.metric("Avg monthly return", f"{avg_ret*100:.2f}%")
    else:
        st.metric("Avg monthly return", "—")

with col4:
    if "brightness_change" in df_filt.columns:
        avg_bchg = df_filt["brightness_change"].mean()
        st.metric("Avg brightness change", f"{avg_bchg:.4f}")
    else:
        st.metric("Avg brightness change", "—")

st.divider()

# --- Time series: average brightness over time -------------------------
st.subheader("Average Nightlights Over Time")

if "avg_rad_month" in df_filt.columns:
    if "state_name" in df_filt.columns:
        group_level = st.radio(
            "Aggregate level:",
            ["All firms", "By state"],
            horizontal=True,
        )
    else:
        group_level = "All firms"

    if group_level == "All firms" or "state_name" not in df_filt.columns:
        ts = (
            df_filt.groupby("date")["avg_rad_month"]
            .mean()
            .reset_index()
            .sort_values("date")
        )
        ts = ts.set_index("date")
        st.line_chart(ts, height=300)
    else:
        # Show top 5 states by average brightness and plot them
        state_avg = (
            df_filt.groupby("state_name")["avg_rad_month"]
            .mean()
            .sort_values(ascending=False)
        )
        top_states = state_avg.head(5).index.tolist()

        st.caption(f"Showing top 5 states by average brightness: {', '.join(top_states)}")

        subset = df_filt[df_filt["state_name"].isin(top_states)].copy()
        ts_multi = (
            subset.groupby(["date", "state_name"])["avg_rad_month"]
            .mean()
            .reset_index()
            .sort_values(["state_name", "date"])
        )

        # Pivot for multi-line chart
        pivot = ts_multi.pivot(index="date", columns="state_name", values="avg_rad_month")
        st.line_chart(pivot, height=350)
else:
    st.info("Column `avg_rad_month` not found in the dataset.")

st.divider()

# --- Relationship: brightness_change vs future returns -----------------
st.subheader("Brightness Changes vs Next-Month Returns")

if {"brightness_change", "ret_fwd_1m"}.issubset(df_filt.columns):
    corr = df_filt[["brightness_change", "ret_fwd_1m"]].corr().iloc[0, 1]
    st.write(f"Correlation between **brightness_change** and **next-month return**: `{corr:.3f}`")

    st.caption("Each point is a (ticker, month) observation in your filtered sample.")
    st.scatter_chart(
        df_filt[["brightness_change", "ret_fwd_1m"]].rename(
            columns={"brightness_change": "Brightness change", "ret_fwd_1m": "Next-month return"}
        )
    )
else:
    st.info(
        "Need columns `brightness_change` and `ret_fwd_1m` to show this chart. "
        "Make sure your pipeline created those features."
    )

st.divider()

# --- Single-ticker detail view -----------------------------------------
st.subheader("Single Ticker View")

if "ticker" in df_filt.columns:
    unique_tickers = sorted(df_filt["ticker"].unique())
    default_ticker = selected_tickers[0] if selected_tickers else unique_tickers[0]
    default_index = unique_tickers.index(default_ticker)

    tkr = st.selectbox("Choose a ticker:", options=unique_tickers, index=default_index)

    df_tkr = df_filt[df_filt["ticker"] == tkr].sort_values("date").copy()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{tkr}: Brightness vs Time**")
        if "avg_rad_month" in df_tkr.columns:
            series_b = df_tkr.set_index("date")["avg_rad_month"]
            st.line_chart(series_b, height=250)
        else:
            st.info("`avg_rad_month` missing.")

    with c2:
        st.markdown(f"**{tkr}: Monthly Returns**")
        if "ret" in df_tkr.columns:
            series_r = df_tkr.set_index("date")["ret"]
            st.line_chart(series_r, height=250)
        else:
            st.info("`ret` missing.")

    st.caption("Use this to tell the story for specific companies in your writeup.")
else:
    st.info("No `ticker` column found in the dataset.")

