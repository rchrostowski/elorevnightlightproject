# pages/3_County_Explorer.py

import streamlit as st
import pandas as pd
import altair as alt

from src.load_data import load_model_data

st.title("County Explorer")

df = load_model_data(fallback_if_missing=True).copy()
if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

mask = df[["brightness_change", "ret_fwd"]].notna().any(axis=1)
df = df[mask].sort_values("date")

# Sidebar selection
st.sidebar.header("County selection")

state_list = sorted(df["state_full"].dropna().unique())
state = st.sidebar.selectbox("State", options=state_list)

df_state = df[df["state_full"] == state]
county_list = sorted(df_state["county_name"].dropna().unique())
county = st.sidebar.selectbox("County", options=county_list)

df_c = df_state[df_state["county_name"] == county].copy()
if df_c.empty:
    st.warning("No data for this county.")
    st.stop()

st.subheader(f"{county}, {state}")

# Firms in this county
firms = df_c[["ticker", "firm"]].drop_duplicates().sort_values("ticker")
st.markdown("**Firms headquartered in this county:**")
st.table(firms)

# Aggregate over firms in the county
agg = (
    df_c.groupby("date")
    .agg(
        avg_brightness=("avg_rad_month", "mean"),
        avg_dlight=("brightness_change", "mean"),
        avg_ret_fwd=("ret_fwd", "mean"),
        n_firms=("ticker", "nunique"),
    )
    .reset_index()
)

agg["avg_ret_fwd_pct"] = agg["avg_ret_fwd"] * 100

c1, c2 = st.columns(2)
c1.metric("Average number of firms per month", f"{agg['n_firms'].mean():.1f}")
c2.metric("Obs (month-county)", f"{len(agg)}")

st.markdown("### County average: ΔBrightness and next-month returns")

base = alt.Chart(agg).encode(x="date:T")

line_dlight = base.mark_line(color="#f58518").encode(
    y=alt.Y("avg_dlight:Q", title="Average ΔBrightness"),
)

line_ret = base.mark_line(color="#4c78a8").encode(
    y=alt.Y("avg_ret_fwd_pct:Q", title="Avg next-month return (%)"),
)

st.altair_chart(
    alt.layer(line_dlight, line_ret).resolve_scale(y="independent"),
    use_container_width=True,
)

st.markdown("### Firm-level scatter in this county")

scatter = (
    alt.Chart(df_c)
    .mark_circle(opacity=0.4)
    .encode(
        x=alt.X("brightness_change:Q", title="ΔBrightness (firm's HQ county)"),
        y=alt.Y("ret_fwd:Q", title="Next-month return"),
        color=alt.Color("ticker:N", legend=None),
        tooltip=["ticker", "firm", "date", "brightness_change", "ret_fwd"],
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

st.caption(
    "This page zooms into a single HQ county and shows both the average "
    "relationship and the firm-by-firm scatter of ΔBrightness vs next-month returns."
)

