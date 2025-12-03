import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.set_page_config(
    page_title="Night Lights & Stock Returns – FIN 377",
    layout="wide",
)

st.markdown("## 🌌 Night Lights & Stock Returns – FIN 377 Project")

# ---------- Load and clean data ----------
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

# Clean counties (some rows had county_name 'n/a' earlier)
if "county_name" in df.columns:
    df["county_name"] = df["county_name"].astype(str)
    df = df[df["county_name"].str.lower() != "n/a"]

# Core columns
for col in ["brightness_change", "ret_fwd_1m"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- High-level summary ----------
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

# ---------- Project goal explanation ----------
st.markdown("### 🎯 What this project is testing")

st.markdown(
    r"""
We test whether **changes in night-time light around firm headquarters** are informative about
**next-month stock returns**.

The core regression is:

\[
\text{ret\_{fwd, i,t}} = \alpha + \beta \cdot \Delta \text{Brightness}\_{i,t}
+ \gamma_{\text{year-month}(t)} + \varepsilon_{i,t}
\]

Where:

- **Dependent variable**: `ret_fwd_1m`  
  - This is the **total next-month return** of ticker \(i\) (we are *not* subtracting the market or risk-free rate).
- **Key regressor**: `brightness_change`  
  - The **change in VIIRS night-lights** in the ticker’s **HQ county** from month \(t-1\) to \(t\).
- **Fixed effects**: \(\gamma_{\text{year-month}(t)}\)  
  - These are **year–month fixed effects**, implemented as `C(year-month)`.  
  - They compare **brighter vs darker HQ counties *within the same calendar month***.  
  - This removes **seasonality** and common macro shocks.

So, the coefficient **β on `brightness_change`** answers:

> “Holding the calendar month fixed, do firms in HQ counties that brighten more tend to have higher next-month returns?”

The rest of the app is structured to support this story:

1. **Overview** – what the sample looks like and how brightness/returns are distributed.  
2. **Ticker Explorer** – for a single firm: how its HQ lights move vs its own returns.  
3. **County Explorer** – for a single county: which firms sit there and how they behave.  
4. **Globe** – interactive HQ map to visualize where the signals come from.  
5. **Regression** – the main econometric test: `ret_fwd_1m ~ brightness_change + C(year-month)`.
"""
)
