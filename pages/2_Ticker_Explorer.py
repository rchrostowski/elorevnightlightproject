# pages/2_Ticker_Explorer.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.markdown("## 🔍 Ticker Explorer (HQ County)")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset is missing. Run `python scripts/build_all.py` first.")
    st.stop()

required = {"ticker", "date"}
if not required.issubset(df.columns):
    st.error(f"`nightlights_model_data` must contain at least: {required}")
    st.stop()

# Basic cleaning
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# --- Figure out column names for HQ county/state ---

county_col = None
for c in ["county_name", "county", "county_label"]:
    if c in df.columns:
        county_col = c
        break

state_col = None
for c in ["state", "state_abbr", "state_code", "state_name"]:
    if c in df.columns:
        state_col = c
        break

# Ticker selection
tickers = sorted(df["ticker"].unique())
default_ticker = "AAPL" if "AAPL" in tickers else tickers[0]

st.sidebar.header("Ticker filters")
ticker_sel = st.sidebar.selectbox("Ticker", options=tickers, index=tickers.index(default_ticker))

# Date window
date_min = df["date"].min()
date_max = df["date"].max()
start, end = st.sidebar.slider(
    "Date window",
    min_value=date_min.to_pydatetime(),
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    format="YYYY-MM",
)

df_t = df[(df["ticker"] == ticker_sel) & (df["date"].between(start, end))].copy()

if df_t.empty:
    st.warning("No observations for that ticker / date window.")
    st.stop()

# --- Header metrics ---

firm_name = df_t["firm"].iloc[0] if "firm" in df_t.columns else ticker_sel

county_val = df_t[county_col].iloc[0] if county_col else "n/a"
state_val = df_t[state_col].iloc[0] if state_col and state_col in df_t.columns else None

# Avoid displaying ", n/a"
if state_val is None or pd.isna(state_val) or str(state_val).lower() in ["nan", "none", "n/a", ""]:
    hq_label = str(county_val)
else:
    hq_label = f"{county_val}, {state_val}"

n_obs = len(df_t)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ticker", ticker_sel)
with col2:
    st.metric("Firm", firm_name)
with col3:
    st.metric("HQ county", hq_label)

# --- Returns vs brightness over time ---

st.markdown("### Time Series: county brightness vs next-month returns")

plot_cols = []
labels = {}

if "brightness_change" in df_t.columns:
    plot_cols.append("brightness_change")
    labels["brightness_change"] = "ΔBrightness (HQ county)"

if "ret_fwd_1m" in df_t.columns:
    plot_cols.append("ret_fwd_1m")
    labels["ret_fwd_1m"] = "Next-month return"

if not plot_cols:
    st.warning("No `brightness_change` or `ret_fwd_1m` column in the dataset.")
else:
    ts = df_t[["date"] + plot_cols].set_index("date").rename(columns=labels)
    st.line_chart(ts)

    st.caption(
        "**ret_fwd_1m** is the ticker’s **next-month total return** (not excess over the risk-free rate). "
        "`ΔBrightness` is the month-to-month change in VIIRS night-lights for the HQ county."
    )

# --- Scatter: returns vs brightness (per ticker) ---

if {"brightness_change", "ret_fwd_1m"}.issubset(df_t.columns):
    st.markdown("### Scatter: ΔBrightness vs next-month return")

    scat = df_t[["brightness_change", "ret_fwd_1m"]].dropna()
    if scat.empty:
        st.info("No non-missing pairs of (brightness_change, ret_fwd_1m) for this ticker.")
    else:
        corr = scat["brightness_change"].corr(scat["ret_fwd_1m"])
        st.write(f"Correlation for **{ticker_sel}** in this window: **{corr:.3f}**")

        st.scatter_chart(scat, x="brightness_change", y="ret_fwd_1m")
        st.caption(
            "Each point is a month for this ticker. "
            "X-axis: change in HQ county night-lights; "
            "Y-axis: **next-month stock return**."
        )

# --- Raw observations table ---

st.markdown("### Underlying observations")

show_cols = [c for c in [
    "ticker", "firm", county_col, state_col,
    "date", "avg_rad_month", "brightness_change",
    "ret", "ret_fwd_1m"
] if c and c in df_t.columns]

st.dataframe(
    df_t.sort_values("date")[show_cols],
    use_container_width=True,
)
