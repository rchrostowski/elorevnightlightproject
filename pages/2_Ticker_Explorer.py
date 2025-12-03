import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.markdown("## 2. Ticker Explorer – HQ lights vs that ticker’s returns")

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

tickers = sorted(df["ticker"].unique())
default_ticker = "AAPL" if "AAPL" in tickers else tickers[0]

ticker = st.selectbox("Select ticker:", options=tickers, index=tickers.index(default_ticker))
df_t = df[df["ticker"] == ticker].sort_values("date").copy()

firm_name = df_t["firm"].iloc[0] if not df_t["firm"].isna().all() else ticker
county_name = df_t["county_name"].iloc[0]
state = df_t["state"].iloc[0]

st.markdown(
    f"""
**Firm:** `{ticker}` – {firm_name}  
**HQ county:** {county_name}, {state}  

- **Y-axis** in the main chart is **next-month total return** (`ret_fwd_1m`).  
- We explicitly use **total returns (not market-excess)**, and we say that out loud here.
"""
)

st.markdown("---")

# --- 1. Time series: returns with brightness shading ---

st.markdown("### A. Time series: next-month returns vs HQ brightness change")

if len(df_t) < 3:
    st.warning("Not enough data points for this ticker to make a meaningful chart.")
else:
    # Normalize brightness change for color
    b = df_t["brightness_change"]
    if b.nunique() > 1:
        b_norm = (b - b.min()) / (b.max() - b.min())
    else:
        b_norm = pd.Series(0.5, index=b.index)

    df_t["brightness_norm"] = b_norm

    fig_ts = px.bar(
        df_t,
        x="date",
        y="ret_fwd_1m",
        color="brightness_norm",
        color_continuous_scale="Blues",
        labels={
            "date": "Month",
            "ret_fwd_1m": "Next-month total return",
            "brightness_norm": "Δ brightness (normalized)",
        },
        title=f"{ticker}: Next-month returns, colored by HQ ΔBrightness",
    )
    fig_ts.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown(
        """
**How to read this:**  

- Each bar is a **month** (x-axis) and the **next-month total return** for this ticker (y-axis).  
- The **color** encodes the **change in HQ county brightness**:
  - darker bars → smaller brightness change;  
  - lighter bars → large positive changes in night-time lights.  

The regression in the **Regression** tab essentially asks:  
> *Across all firms and months, do the light-colored bars (brightening HQs) tend to have systematically higher next-month returns?*
"""
    )

st.markdown("---")

# --- 2. Scatter: ΔBrightness vs next-month return for this ticker ---

st.markdown("### B. Within-ticker scatter: ΔBrightness vs next-month return")

if len(df_t) >= 5:
    fig_sc = px.scatter(
        df_t,
        x="brightness_change",
        y="ret_fwd_1m",
        trendline="ols",
        labels={
            "brightness_change": "Δ brightness (HQ county)",
            "ret_fwd_1m": "Next-month total return",
        },
        title=f"{ticker}: ΔBrightness vs next-month total return (raw within-ticker)",
    )
    fig_sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown(
        """
**How this ties into the regression:**  

- This is a **within-ticker** view of the same relationship:  
  - x-axis: change in HQ county brightness this month,  
  - y-axis: next-month total return.  
- The **regression tab** pools these relationships across **all tickers** and adds **year–month fixed effects** `C(year-month)`, which:
  - compares bright vs dark HQ counties *within the same calendar month*,  
  - removes common seasonal and macro effects.
"""
    )
else:
    st.info("Too few observations for this ticker to show a meaningful scatter.")
