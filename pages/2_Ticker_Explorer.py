# pages/2_Ticker_Explorer.py

import streamlit as st
import pandas as pd
import altair as alt

from src.load_data import load_model_data

st.title("Ticker Explorer")

df = load_model_data(fallback_if_missing=True).copy()
if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Only keep rows where we have at least brightness_change or returns
mask = df[["brightness_change", "ret", "ret_fwd"]].notna().any(axis=1)
df = df[mask].sort_values("date")

tickers = sorted(df["ticker"].unique())
default_index = tickers.index("AAPL") if "AAPL" in tickers else 0

st.sidebar.header("Ticker selection")
ticker = st.sidebar.selectbox("Ticker", options=tickers, index=default_index)

df_t = df[df["ticker"] == ticker].copy()
if df_t.empty:
    st.warning("No data for this ticker.")
    st.stop()

firm_name = df_t["firm"].iloc[0]
state = df_t["state_full"].iloc[0]
county = df_t["county_name"].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Ticker", ticker)
c2.metric("Firm", firm_name)
c3.metric("HQ county", f"{county}, {state}")

st.markdown("### Time series: HQ ΔBrightness vs next-month returns")

df_t["ret_fwd_pct"] = df_t["ret_fwd"] * 100

base = alt.Chart(df_t).encode(x="date:T")

line_dlight = base.mark_line(color="#f58518").encode(
    y=alt.Y("brightness_change:Q", title="ΔBrightness (HQ county)"),
)

line_ret = base.mark_line(color="#4c78a8").encode(
    y=alt.Y("ret_fwd_pct:Q", title="Next-month return (%)"),
)

st.altair_chart(
    alt.layer(line_dlight, line_ret).resolve_scale(y="independent"),
    use_container_width=True,
)

st.markdown("### Cross-section for this ticker: ΔBrightness vs next-month return")

corr_t = df_t["brightness_change"].corr(df_t["ret_fwd"])
st.write(f"Correlation (ΔBrightness, next-month return) for **{ticker}**: "
         f"**{corr_t:.3f}**" if pd.notna(corr_t) else "Correlation not available.")

scatter = (
    alt.Chart(df_t)
    .mark_circle(size=80, opacity=0.6)
    .encode(
        x=alt.X("brightness_change:Q", title="ΔBrightness (HQ county)"),
        y=alt.Y("ret_fwd:Q", title="Next-month return"),
        color=alt.Color("date:T", legend=None),
        tooltip=["date", "brightness_change", "ret_fwd"],
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

st.caption(
    "This page focuses on a single firm. We look at how changes in nightlights "
    "around its HQ county line up with the firm's subsequent monthly returns."
)

