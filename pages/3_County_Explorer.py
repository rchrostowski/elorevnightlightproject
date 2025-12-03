# pages/3_County_Explorer.py

import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.set_page_config(
    page_title="County Explorer – Night Lights Anomalia",
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
ret_col = _get_col(df, ["ret_excess", "ret", "return"], required=True)
ret_fwd_col = _get_col(df, ["ret_fwd_1m", "ret_fwd", "ret_forward_1m"])

st.markdown("## County Explorer – HQ Counties Across Firms")

# Build a county selector: "(County, ST) – N tickers"
if not {"county_name", "state"}.issubset(df.columns):
    st.error("Expected 'county_name' and 'state' columns to exist for HQ counties.")
    st.stop()

county_stats = (
    df.groupby(["county_name", "state"])
    .agg(n_tickers=("ticker", "nunique"), n_obs=("ticker", "size"))
    .reset_index()
)

county_stats["label"] = (
    county_stats["county_name"] + ", " + county_stats["state"] +
    "  –  " + county_stats["n_tickers"].astype(str) + " tickers"
)

county_stats = county_stats.sort_values("n_tickers", ascending=False)

selected_label = st.sidebar.selectbox(
    "Select HQ county",
    options=county_stats["label"].tolist(),
)

row = county_stats[county_stats["label"] == selected_label].iloc[0]
county_name, state = row["county_name"], row["state"]

st.markdown(f"### {county_name}, {state}")
st.caption(
    f"This county hosts {int(row['n_tickers'])} tickers, {int(row['n_obs'])} firm-months in total."
)

df_c = df[(df["county_name"] == county_name) & (df["state"] == state)].copy()

# ---------- Time-series: avg across firms in this county ----------

agg = (
    df_c.groupby("date")
    .agg(
        mean_dlight=(brightness_col, "mean"),
        mean_ret=(ret_col, "mean"),
    )
    .reset_index()
)

c1, c2 = st.columns(2)

with c1:
    fig1 = px.line(
        agg,
        x="date",
        y="mean_dlight",
        labels={"date": "Date", "mean_dlight": "Avg ΔLight (HQ county)"},
        title="County-level ΔLight over time (averaged across HQ firms)",
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    ycs = ["mean_ret"]
    labels = {"mean_ret": "Avg same-month return"}
    if ret_fwd_col and ret_fwd_col in df_c.columns:
        agg["mean_ret_fwd"] = df_c.groupby("date")[ret_fwd_col].mean().values
        ycs.append("mean_ret_fwd")
        labels["mean_ret_fwd"] = "Avg next-month return"

    fig2 = px.line(
        agg,
        x="date",
        y=ycs,
        labels={"date": "Date", **labels},
        title="Average returns for firms HQ’d in this county",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Cross-section inside county ----------

st.markdown("### Tickers in this county")

c_sc1, c_sc2 = st.columns(2)

with c_sc1:
    if ret_fwd_col and ret_fwd_col in df_c.columns:
        tmp = df_c[[brightness_col, ret_fwd_col, "ticker"]].dropna()
        if tmp.empty:
            st.info("No ΔLight / next-month pairs for regression-style scatter.")
        else:
            fig_sc = px.scatter(
                tmp,
                x=brightness_col,
                y=ret_fwd_col,
                color="ticker",
                trendline="ols",
                opacity=0.6,
                labels={
                    brightness_col: "ΔLight (HQ county)",
                    ret_fwd_col: "Next-month return",
                },
                title="Within this HQ county: ΔLight vs next-month return",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Next-month return not available in dataset.")

with c_sc2:
    top_tickers = (
        df_c.groupby("ticker")
        .agg(
            firm=("firm", "first"),
            n_obs=("date", "size"),
        )
        .reset_index()
        .sort_values("n_obs", ascending=False)
        .head(15)
    )
    st.dataframe(top_tickers, use_container_width=True)

st.caption(
    "This page lets you see how one HQ county’s light and returns have evolved, "
    "and which firms are driving the activity."
)
