# pages/2_Ticker_Explorer.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.markdown("## 🔍 Ticker Explorer")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset is missing. Run `python scripts/build_all.py` first.")
    st.stop()

if "date" not in df.columns or "ticker" not in df.columns:
    st.error("`nightlights_model_data` must contain at least `ticker` and `date`.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Sidebar controls
tickers = sorted(df["ticker"].unique())
default_ticker = "AAPL" if "AAPL" in tickers else tickers[0]

st.sidebar.header("Ticker filters")
ticker_sel = st.sidebar.selectbox("Ticker", options=tickers, index=tickers.index(default_ticker))

# Optional date range
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

firm_name = df_t["firm"].iloc[0] if "firm" in df_t.columns else ticker_sel
county = df_t["county_name"].iloc[0] if "county_name" in df_t.columns else "n/a"
state = df_t["state"].iloc[0] if "state" in df_t.columns else "n/a"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Firm", firm_name)
with col2:
    st.metric("HQ county", f"{county}, {state}")
with col3:
    st.metric("# months in window", len(df_t))

# Time series of brightness and returns
st.markdown("### Time series: brightness vs. returns")

plot_cols = []
if "brightness_change" in df_t.columns:
    plot_cols.append("brightness_change")
if "ret_fwd_1m" in df_t.columns:
    plot_cols.append("ret_fwd_1m")

if not plot_cols:
    st.warning("Missing `brightness_change` / `ret_fwd_1m` columns in the dataset.")
else:
    ts = df_t[["date"] + plot_cols].set_index("date")
    st.line_chart(ts)

    st.caption(
        "- **brightness_change**: change in county night-lights vs previous month\n"
        "- **ret_fwd_1m**: next-month stock return for this ticker"
    )

# Scatter: ΔBrightness vs next-month return
if {"brightness_change", "ret_fwd_1m"}.issubset(df_t.columns):
    st.markdown("### Scatter: ΔBrightness vs. next-month return")

    scat = df_t[["brightness_change", "ret_fwd_1m"]].dropna()
    if scat.empty:
        st.info("No non-missing pairs of (brightness_change, ret_fwd_1m) for this ticker.")
    else:
        corr = scat["brightness_change"].corr(scat["ret_fwd_1m"])
        st.write(f"Correlation in this window: **{corr:.3f}**")

        st.scatter_chart(scat, x="brightness_change", y="ret_fwd_1m")
        st.caption(
            "Each point is a month for this ticker. X-axis: change in night-lights at HQ county; "
            "Y-axis: next-month stock return."
        )

# Raw table
st.markdown("### Underlying observations")
show_cols = [c for c in [
    "ticker", "firm", "county_name", "state",
    "date", "avg_rad_month", "brightness_change",
    "ret", "ret_fwd_1m"
] if c in df_t.columns]

st.dataframe(
    df_t.sort_values("date")[show_cols],
    use_container_width=True,
)
