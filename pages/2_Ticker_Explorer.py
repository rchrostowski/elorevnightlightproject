# pages/2_Ticker_Explorer.py

import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.set_page_config(
    page_title="Ticker Explorer – Night Lights Anomalia",
    layout="wide",
)

def _get_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None

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

st.markdown("## Ticker Explorer – HQ County vs Returns")

# ---------- Sidebar controls ----------

tickers = sorted(df["ticker"].unique().tolist())

selected_ticker = st.sidebar.selectbox("Select ticker", options=tickers, index=0)

df_t = df[df["ticker"] == selected_ticker].copy().sort_values("date")

firm_name = df_t["firm"].iloc[0] if "firm" in df_t.columns else selected_ticker
state = df_t["state"].iloc[0] if "state" in df_t.columns else "N/A"
county = df_t["county_name"].iloc[0] if "county_name" in df_t.columns else "N/A"

st.markdown(
    f"### {selected_ticker} – {firm_name}  \n"
    f"**HQ county:** {county}, {state}"
)

# ---------- Top row: time-series ----------

c1, c2 = st.columns(2)

with c1:
    ycol = brightness_col if brightness_col in df_t.columns else level_col
    ylab = "ΔLight (HQ county)" if ycol == brightness_col else "Brightness level (HQ county)"

    fig_b = px.line(
        df_t,
        x="date",
        y=ycol,
        labels={"date": "Date", ycol: ylab},
        title="Night-time light at HQ county over time",
    )
    st.plotly_chart(fig_b, use_container_width=True)

with c2:
    ycols = [ret_col]
    labels = {ret_col: "Return (this month)"}
    title = "Returns over time"

    if ret_fwd_col and ret_fwd_col in df_t.columns:
        ycols.append(ret_fwd_col)
        labels[ret_fwd_col] = "Next-month return"
        title = "Same-month vs next-month returns"

    fig_r = px.line(
        df_t,
        x="date",
        y=ycols,
        labels={"date": "Date", **labels},
        title=title,
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ---------- Scatter ΔLight vs next-month returns ----------

st.markdown("### ΔLight vs future returns for this firm")

if not ret_fwd_col or ret_fwd_col not in df_t.columns:
    st.info("Next-month return column not found – cannot plot predictive relationship.")
else:
    tmp = df_t[[brightness_col, ret_fwd_col, "date"]].dropna()
    if tmp.empty:
        st.info("No non-missing pairs of ΔLight and next-month returns for this ticker.")
    else:
        fig_sc = px.scatter(
            tmp,
            x=brightness_col,
            y=ret_fwd_col,
            trendline="ols",
            labels={
                brightness_col: "ΔLight (HQ county)",
                ret_fwd_col: "Next-month return",
            },
            title=f"{selected_ticker}: ΔLight vs next-month return",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ---------- Table ----------

st.markdown("### Data table (this ticker only)")
show_cols = ["date", brightness_col, ret_col]
if ret_fwd_col and ret_fwd_col in df_t.columns:
    show_cols.append(ret_fwd_col)

extra_cols = [c for c in ["state", "county_name"] if c in df_t.columns]
show_cols = extra_cols + show_cols

st.dataframe(
    df_t[show_cols].sort_values("date").reset_index(drop=True),
    use_container_width=True,
)




