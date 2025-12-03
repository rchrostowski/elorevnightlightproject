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

needed = {"ticker", "firm", "county_name", "brightness_change", "ret_fwd_1m"}
missing = needed - set(df.columns)
if missing:
    st.error(f"`nightlights_model_data` must contain: {needed}. Missing: {missing}")
    st.stop()

# Optional state column (if available)
state_col = None
for cand in ["state", "hq_state", "state_name"]:
    if cand in df.columns:
        state_col = cand
        break

# Clean county
df["county_name"] = df["county_name"].astype(str)
df = df[df["county_name"].str.lower() != "n/a"]

df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")
df = df.dropna(subset=["brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error("No usable rows after cleaning `brightness_change` and `ret_fwd_1m`.")
    st.stop()

tickers = sorted(df["ticker"].unique())
default_ticker = "AAPL" if "AAPL" in tickers else tickers[0]

ticker = st.selectbox("Select ticker:", options=tickers, index=tickers.index(default_ticker))
df_t = df[df["ticker"] == ticker].sort_values("date").copy()

if df_t.empty:
    st.error("No observations for this ticker in the final dataset.")
    st.stop()

firm_name = df_t["firm"].iloc[0] if not df_t["firm"].isna().all() else ticker
county_name = df_t["county_name"].iloc[0]
if state_col:
    state_val = df_t[state_col].iloc[0]
    county_label = f"{county_name}, {state_val}"
else:
    state_val = ""
    county_label = county_name

st.markdown(
    f"""
**Firm selected:** `{ticker}` – {firm_name}  
**HQ county:** {county_label}  

This tab answers:  
- *For this specific firm, when its HQ county brightens or dims, how do its **next-month returns** behave?*  

We keep the **y-axis as next-month total return** (`ret_fwd_1m`) to stay consistent with the regression.
"""
)

st.markdown("---")

# ----- A. Time series: returns colored by HQ brightness change -----
st.markdown("### A. Time series – next-month returns, colored by Δ brightness")

if len(df_t) < 3:
    st.warning("Not enough data points for this ticker to show a meaningful time series.")
else:
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
        title=f"{ticker}: Next-month returns, colored by HQ Δ brightness",
    )
    fig_ts.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown(
        """
**How to explain this chart:**  

- Each bar is one **month** on the x-axis.  
- The **height** of the bar is the **next-month total return** for that ticker.  
- The **color** of the bar encodes the **change in brightness** in the HQ county in that month:
  - lighter bars = **bigger positive brightness changes**,  
  - darker bars = **small or negative brightness changes**.

The regression later asks this question in a pooled way across all firms:  
> *Do the “light blue bars” (months with big positive Δ brightness) tend to be followed by higher returns, once we control for the calendar month?*
"""
    )

st.markdown("---")

# ----- B. Within-ticker scatter: ΔBrightness vs next-month return -----
st.markdown("### B. Within-ticker scatter – ΔBrightness vs next-month returns")

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
        title=f"{ticker}: ΔBrightness vs next-month total return (within-ticker)",
    )
    fig_sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown(
        """
**Interpretation:**  

- Each dot is a **month** for this firm:
  - x-axis = how much the HQ county’s night-lights changed,  
  - y-axis = the return in the **following month**.

- The **trend line** is like a mini regression just for this ticker, with **no fixed effects**.

In the full regression (Regression tab), we:

- pool **all tickers together**, and  
- add **year–month fixed effects** `C(year-month)` to compare bright vs dark HQ counties within each month.
"""
    )
else:
    st.info("Too few observations for this ticker to show a meaningful scatter.")

