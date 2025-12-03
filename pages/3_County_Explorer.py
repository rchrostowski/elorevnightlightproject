import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.markdown("## 3. County Explorer – which HQ counties and how they behave")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset `nightlights_model_data.csv` is missing or empty.")
    st.stop()

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

needed = {"ticker", "firm", "county_name", "state", "brightness_change", "ret_fwd_1m"}
missing = needed - set(df.columns)
if missing:
    st.error(f"`nightlights_model_data` must contain: {needed}. Missing: {missing}")
    st.stop()

# Clean county junk
df["county_name"] = df["county_name"].astype(str)
df = df[df["county_name"].str.lower() != "n/a"]

df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")
df = df.dropna(subset=["brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error("No usable rows after cleaning `brightness_change` and `ret_fwd_1m`.")
    st.stop()

# Build county key
df["county_key"] = df["county_name"] + ", " + df["state"].astype(str)
county_keys = sorted(df["county_key"].unique())

default_key = "Santa Clara County, CA" if "Santa Clara County, CA" in county_keys else county_keys[0]

county_key = st.selectbox("Select HQ county:", options=county_keys, index=county_keys.index(default_key))

county_name, state = county_key.split(", ", 1)
df_c = df[(df["county_name"] == county_name) & (df["state"] == state)].sort_values("date").copy()

st.markdown(
    f"""
**Selected county:** {county_name}, {state}  

This page answers:

- Which **tickers** have HQs in this county?  
- How do their **next-month returns** behave when the **HQ lights brighten or dim**?  
- How does this county fit into the overall regression story.
"""
)

st.markdown("---")

# --- 1. Firms in this county ---

st.markdown("### A. Firms headquartered in this county")

firm_summary = (
    df_c.groupby("ticker", as_index=False)
        .agg(
            firm=("firm", "first"),
            n_obs=("date", "size"),
            avg_ret=("ret_fwd_1m", "mean"),
            avg_brightness_change=("brightness_change", "mean"),
        )
        .sort_values("n_obs", ascending=False)
)

st.dataframe(
    firm_summary.rename(
        columns={
            "ticker": "Ticker",
            "firm": "Firm",
            "n_obs": "# months",
            "avg_ret": "Avg next-month return",
            "avg_brightness_change": "Avg Δ brightness",
        }
    ),
    use_container_width=True,
)

st.markdown(
    """
**How to read this table:**  

- Each row is a **ticker headquartered in this county**.  
- `Avg next-month return` is the **mean of `ret_fwd_1m`** for that ticker in this county.  
- `Avg Δ brightness` is the **average change in HQ night-lights** over the sample.  

This links the **spatial unit** (county) to the **economic outcome** (stock returns).
"""
)

st.markdown("---")

# --- 2. Time series: county brightness vs county average return ---

st.markdown("### B. County time series: average return vs HQ brightness change")

df_c_month = (
    df_c.groupby("date", as_index=False)
        .agg(
            avg_ret=("ret_fwd_1m", "mean"),
            avg_brightness_change=("brightness_change", "mean"),
        )
        .sort_values("date")
)

if len(df_c_month) < 3:
    st.warning("Not enough data for time-series visualization in this county.")
else:
    colL, colR = st.columns(2)

    with colL:
        fig_ret = px.line(
            df_c_month,
            x="date",
            y="avg_ret",
            title=f"{county_key}: average next-month total return over time",
            labels={"date": "Month", "avg_ret": "Avg next-month total return"},
        )
        fig_ret.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ret, use_container_width=True)

    with colR:
        fig_bright = px.line(
            df_c_month,
            x="date",
            y="avg_brightness_change",
            title=f"{county_key}: average Δ brightness over time",
            labels={"date": "Month", "avg_brightness_change": "Avg Δ brightness"},
        )
        fig_bright.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bright, use_container_width=True)

    st.markdown(
        """
**What this pair of plots shows:**  

- Left: when this county’s HQ firms, on average, have **high next-month returns**.  
- Right: when the **night-time lights in this county spike or drop**.  

The fixed-effects regression in the next tab basically asks:  
> *Across all counties and months, do months where HQ counties brighten more tend to be followed by months with higher average returns?*
"""
    )

st.markdown("---")

# --- 3. County-level scatter: ΔBrightness vs next-month return (all ticker-months) ---

st.markdown("### C. County scatter: ΔBrightness vs next-month returns (all firms)")

if len(df_c) >= 10:
    fig_sc = px.scatter(
        df_c,
        x="brightness_change",
        y="ret_fwd_1m",
        color="ticker",
        labels={
            "brightness_change": "Δ brightness (HQ county)",
            "ret_fwd_1m": "Next-month total return",
            "ticker": "Ticker",
        },
        title=f"{county_key}: ΔBrightness vs next-month returns across tickers",
    )
    fig_sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown(
        """
**How this ties into the overall regression:**  

- Here we ignore fixed effects and just look at **all ticker–months in this one county**.  
- In the **Regression** tab, we pool **all counties and tickers** and add **year–month fixed effects**:  
  - that gives a clean estimate of whether **unusually bright HQ months** are linked to **unusually high next-month returns**,  
  - after controlling for **time effects** common to all firms.
"""
    )
else:
    st.info("Too few observations in this county to show a meaningful scatter.")
