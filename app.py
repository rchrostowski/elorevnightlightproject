import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.set_page_config(
    page_title="Night Lights & Stock Returns – FIN 377",
    layout="wide",
)

st.markdown("## 🌌 Night Lights & Stock Returns – FIN 377 Project")

df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "Final dataset `nightlights_model_data.csv` is missing or empty.\n\n"
        "Run `python scripts/build_all.py`, commit the CSV in `data/final/`, and redeploy."
    )
    st.stop()

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Clean out junk counties (e.g. 'n/a')
if "county_name" in df.columns:
    df["county_name"] = df["county_name"].astype(str)
    df = df[df["county_name"].str.lower() != "n/a"]

# Basic sample stats
n_obs = len(df)
n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
n_counties = df["county_name"].nunique() if "county_name" in df.columns else 0
date_min = df["date"].min()
date_max = df["date"].max()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ticker–month observations", f"{n_obs:,}")
with col2:
    st.metric("Unique tickers", f"{n_tickers:,}")
with col3:
    st.metric("HQ counties", f"{n_counties:,}")
with col4:
    st.metric(
        "Sample window",
        f"{date_min.strftime('%Y-%m')} → {date_max.strftime('%Y-%m')}",
    )

st.markdown("---")

st.markdown("### 🎯 Project goal (what this dashboard is actually testing)")

st.markdown(
    r"""
We test whether **changes in night-time light around firm headquarters** predict **next-month stock returns**:

\[
\text{ret\_{fwd, i,t}} = \alpha + \beta \cdot \Delta \text{Brightness}\_{i,t}
+ \gamma_{\text{year-month}(t)} + \varepsilon_{i,t}
\]

Where:

- **`ret_fwd_1m`** (our dependent variable) is the **total next-month return** of ticker \(i\).  
  - We use **total returns**, not market-excess or risk-free-excess returns (this is noted on the plots).
- **`brightness_change`** is the **change in VIIRS night-lights** for the ticker’s HQ **county** from month \(t-1\) to \(t\).
- \(\gamma_{\text{year-month}(t)}\) are **year–month fixed effects**:
  - They compare **brighter vs darker HQ counties *within the same calendar month***.
  - This removes **seasonality** and common macro shocks in that month.

The rest of the app is organized as:

1. **Overview** – high-level distribution of brightness and returns.  
2. **Ticker Explorer** – zoom in on a single ticker: its HQ county’s lights vs its returns.  
3. **County Explorer** – zoom in on a county: which firms sit there and how they behave.  
4. **Globe** – interactive map of HQ counties and their light / return signals.  
5. **Regression** – the main econometric result: \( \text{ret\_{fwd}} \sim \Delta \text{Brightness} + C(\text{year-month}) \).
"""
)
