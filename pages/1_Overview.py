import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.markdown("## 1. Overview – sample, variables, and big picture")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset `nightlights_model_data.csv` is missing or empty.")
    st.stop()

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Clean county junk
if "county_name" in df.columns:
    df["county_name"] = df["county_name"].astype(str)
    df = df[df["county_name"].str.lower() != "n/a"]

# Keep core regression vars
needed = {"brightness_change", "ret_fwd_1m"}
missing = needed - set(df.columns)
if missing:
    st.error(f"`nightlights_model_data` is missing columns: {missing}")
    st.stop()

df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")
df = df.dropna(subset=["brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error("No rows remain after cleaning brightness and return columns.")
    st.stop()

# --- Summary KPIs ---

date_min = df["date"].min()
date_max = df["date"].max()
n_obs = len(df)
n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
n_counties = df["county_name"].nunique() if "county_name" in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ticker–month obs.", f"{n_obs:,}")
with col2:
    st.metric("Tickers", f"{n_tickers:,}")
with col3:
    st.metric("HQ counties", f"{n_counties:,}")
with col4:
    st.metric(
        "Sample window",
        f"{date_min.strftime('%Y-%m')} → {date_max.strftime('%Y-%m')}",
    )

st.markdown(
    """
**Interpretation for class:**  
This is the *working sample* used in the regression tab. Every point is a **ticker–month** with:
- an HQ county brightness change (`brightness_change`), and  
- that ticker’s **next-month total return** (`ret_fwd_1m`).
"""
)

st.markdown("---")

# --- Distribution plots: brightness_change and returns ---

colL, colR = st.columns(2)

with colL:
    fig_b = px.histogram(
        df,
        x="brightness_change",
        nbins=40,
        title="Distribution of HQ brightness changes",
    )
    fig_b.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown(
        """
**What this shows:**  
Most HQ counties have relatively small changes in brightness from month to month,  
with a few months where the HQ county gets much brighter or much dimmer.
"""
    )

with colR:
    fig_r = px.histogram(
        df,
        x="ret_fwd_1m",
        nbins=40,
        title="Distribution of next-month total returns",
    )
    fig_r.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown(
        """
**What this shows:**  
This is the distribution of **one-month ahead total stock returns**.  
We use **total returns (not market-excess)** in the regression, and make that explicit in the text.
"""
    )

st.markdown("---")

# --- Simple scatter: brightness_change vs next-month return ---

st.markdown("### Brightness vs next-month returns (raw relationship)")

fig_scatter = px.scatter(
    df.sample(min(3000, len(df)), random_state=42),
    x="brightness_change",
    y="ret_fwd_1m",
    opacity=0.4,
    trendline="ols",
    labels={
        "brightness_change": "Δ brightness (HQ county)",
        "ret_fwd_1m": "Next-month total return",
    },
    title="Raw relationship: ΔBrightness vs next-month total returns",
)
fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """
**How this connects to the regression:**  

- This scatter is like a **raw correlation view** – it ignores **seasonality** and macro shocks.  
- The regression on the **Regression** tab adds **year–month fixed effects** `C(year-month)`:
  - that means we compare **brighter vs darker HQ counties *within the same calendar month***.  
  - So the regression coefficient on `brightness_change` is a **seasonality-adjusted version** of what you see here.
"""
)

