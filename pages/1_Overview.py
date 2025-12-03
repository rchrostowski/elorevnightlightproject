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

# ----- Summary KPIs -----
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
**How to explain this in class:**  
Each row in our final dataset is a **ticker–month observation**. For every ticker in the S&P 500,  
we know:

- the **county where its HQ is located**,  
- how much that county’s **night-time brightness changed** in that month (`brightness_change`), and  
- the ticker’s **total return in the **next** month** (`ret_fwd_1m`).

This is the exact sample that feeds the regression in the Regression tab.
"""
)

st.markdown("---")

# ----- Distributions: brightness_change and returns -----
colL, colR = st.columns(2)

with colL:
    fig_b = px.histogram(
        df,
        x="brightness_change",
        nbins=40,
        title="Distribution of HQ brightness changes (Δ brightness)",
    )
    fig_b.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown(
        """
**Narrative:**  
This shows how large the **changes in night-lights** around firm HQs typically are.

- Most observations have **small** `brightness_change` (near zero).  
- A smaller number of months show **big jumps or drops** in brightness, which we interpret as big local activity changes.
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
**Narrative:**  
This is the distribution of the **next-month total stock returns** we are trying to explain.

- Returns are centered somewhere around zero, with both positive and negative outliers.  
- This is **total return**, not market-excess — we are explicit about that.
"""
    )

st.markdown("---")

# ----- Raw scatter + correlation -----
st.markdown("### Brightness vs next-month returns (raw relationship)")

corr = df["brightness_change"].corr(df["ret_fwd_1m"])
colA, colB = st.columns([1.1, 2.9])
with colA:
    st.metric("Corr(ΔBrightness, next-month return)", f"{corr:.3f}")
with colB:
    st.caption(
        "This is the plain, unconditional correlation between HQ brightness changes and next-month returns.\n"
        "The regression later adds month fixed effects to *clean this up* for seasonality and macro shocks."
    )

fig_scatter = px.scatter(
    df.sample(min(4000, len(df)), random_state=42),
    x="brightness_change",
    y="ret_fwd_1m",
    opacity=0.35,
    trendline="ols",
    labels={
        "brightness_change": "Δ brightness (HQ county)",
        "ret_fwd_1m": "Next-month total return",
    },
    title="Raw relationship: ΔBrightness vs next-month total returns (no controls)",
)
fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """
**How this connects to the regression:**  

- This chart is like a **visual correlation**: when HQ brightness changes a lot, do returns move?  
- However, it **ignores the calendar month**. It mixes together:
  - seasonal patterns (e.g., holiday lights), and  
  - common macro shocks (e.g., COVID or Fed events).

In the **Regression** tab, we add `C(year-month)` fixed effects:

- That means we are comparing **bright vs dark HQ counties *within the same year–month***.  
- The coefficient on `brightness_change` in that regression is basically a **seasonality-adjusted version** of what you see here.
"""
)
