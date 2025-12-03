# pages/3_County_Explorer.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.markdown("## 🏙 County Explorer")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset is missing. Run `python scripts/build_all.py` first.")
    st.stop()

required = {"county_name", "state", "date"}
if not required.issubset(df.columns):
    st.error(f"`nightlights_model_data` must contain columns: {required}")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["county_label"] = df["county_name"].astype(str) + ", " + df["state"].astype(str)

# Sidebar: pick county
st.sidebar.header("County filters")

county_options = sorted(df["county_label"].unique())
default_label = "Santa Clara County, CA" if any(
    "Santa Clara" in c and "CA" in c for c in county_options
) else county_options[0]

county_sel = st.sidebar.selectbox(
    "HQ county",
    options=county_options,
    index=county_options.index(default_label) if default_label in county_options else 0,
)

df_c = df[df["county_label"] == county_sel].copy()
if df_c.empty:
    st.warning("No observations for that county.")
    st.stop()

date_min = df_c["date"].min()
date_max = df_c["date"].max()
start, end = st.sidebar.slider(
    "Date window",
    min_value=date_min.to_pydatetime(),
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    format="YYYY-MM",
)

df_c = df_c[df_c["date"].between(start, end)].copy()
if df_c.empty:
    st.warning("No observations in that date window for this county.")
    st.stop()

# Summary metrics
n_tickers = df_c["ticker"].nunique() if "ticker" in df_c.columns else 0
avg_bright = df_c["avg_rad_month"].mean() if "avg_rad_month" in df_c.columns else float("nan")
avg_dlight = df_c["brightness_change"].mean() if "brightness_change" in df_c.columns else float("nan")
avg_ret = df_c["ret_fwd_1m"].mean() if "ret_fwd_1m" in df_c.columns else float("nan")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tickers HQ'd here", n_tickers)
with col2:
    if pd.notna(avg_dlight):
        st.metric("Avg Δbrightness", f"{avg_dlight:.2f}")
    else:
        st.metric("Avg Δbrightness", "n/a")
with col3:
    if pd.notna(avg_ret):
        st.metric("Avg next-month return", f"{avg_ret:.2%}")
    else:
        st.metric("Avg next-month return", "n/a")

# Firms in this county
st.markdown("### Firms headquartered in this county")

if {"ticker", "firm"}.issubset(df_c.columns):
    firms_table = (
        df_c[["ticker", "firm"]]
        .drop_duplicates()
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    st.table(firms_table)
else:
    st.info("Ticker / firm information not available in this dataset.")

# Time series for the county
st.markdown("### County brightness and returns over time")

plot_cols = []
labels = {}
if "avg_rad_month" in df_c.columns:
    plot_cols.append("avg_rad_month")
    labels["avg_rad_month"] = "avg_rad_month (level)"
if "brightness_change" in df_c.columns:
    plot_cols.append("brightness_change")
    labels["brightness_change"] = "brightness_change (Δ vs prev. month)"
if "ret_fwd_1m" in df_c.columns:
    plot_cols.append("ret_fwd_1m")
    labels["ret_fwd_1m"] = "ret_fwd_1m (next-month return)"

if plot_cols:
    ts = df_c[["date"] + plot_cols].set_index("date")
    st.line_chart(ts.rename(columns=labels))
    st.caption(
        "Lines show how night-lights and next-month returns evolve for HQ firms in this county."
    )
else:
    st.info("No brightness / return columns available to plot.")

# Raw table
st.markdown("### Underlying observations in this county")

show_cols = [c for c in [
    "ticker", "firm", "county_name", "state",
    "date", "avg_rad_month", "brightness_change",
    "ret", "ret_fwd_1m"
] if c in df_c.columns]

st.dataframe(
    df_c.sort_values(["date", "ticker"])[show_cols],
    use_container_width=True,
)
