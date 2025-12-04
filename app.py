# app.py

import streamlit as st
import pandas as pd
import numpy as np

from src.load_data import load_model_data

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="🌌 Night Lights & Returns – Overview",
    layout="wide",
)

st.title("🌌 Night Lights & Stock Returns – Project Overview")

st.caption(
    "S&P 500 firms × VIIRS night-time lights × monthly returns\n"
    "FIN 377 • Nightlight Trading Project"
)

# ----------------------------
# 1. Load and sanity-check data
# ----------------------------
panel = load_model_data(fallback_if_missing=True)

if panel.empty:
    st.error("nightlights_model_data.csv is missing or empty. "
             "Make sure build_all.py has been run and the final CSV is committed.")
    st.stop()

panel = panel.copy()
panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
panel = panel.dropna(subset=["date"])

# ----------------------------
# 2. Basic sample statistics
# ----------------------------
st.subheader("🧱 Sample Overview")

n_obs = len(panel)
n_firms = panel["ticker"].nunique() if "ticker" in panel.columns else None
n_counties = panel["county_name"].nunique() if "county_name" in panel.columns else None
date_min = panel["date"].min()
date_max = panel["date"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Firm–month observations", f"{n_obs:,}")
if n_firms is not None:
    c2.metric("Unique tickers", f"{n_firms:,}")
if n_counties is not None:
    c3.metric("Unique HQ counties", f"{n_counties:,}")
c4.metric("Date range", f"{date_min.date()} → {date_max.date()}")

with st.expander("See raw sample snapshot (first 10 rows)"):
    st.dataframe(panel.head(10), use_container_width=True)

# ----------------------------
# 3. How strong is the raw brightness–return link?
# ----------------------------
st.subheader("🔍 Raw Relationship: Brightness Change vs. Next-Month Return")

corr_text = "N/A"
corr_val = None

if {"brightness_change", "ret_fwd_1m"}.issubset(panel.columns):
    tmp = panel[["brightness_change", "ret_fwd_1m"]].dropna()
    if len(tmp) > 0 and tmp["brightness_change"].std() > 0 and tmp["ret_fwd_1m"].std() > 0:
        corr_val = float(tmp["brightness_change"].corr(tmp["ret_fwd_1m"]))
        corr_text = f"{corr_val:.3f}"
    else:
        corr_text = "undefined (no variation)"

c1, c2 = st.columns(2)
c1.metric("Simple Pearson correlation (ΔLight vs. next-month return)", corr_text)

if corr_val is not None:
    if abs(corr_val) < 0.05:
        interpretation = (
            "The raw correlation is extremely close to zero, which already hints that "
            "brightness shocks on their own don’t line up strongly with next-month stock returns."
        )
    elif corr_val > 0:
        interpretation = (
            "The raw correlation is positive, but we still need regression with controls "
            "to see if this signal survives once we remove broad market and seasonal effects."
        )
    else:
        interpretation = (
            "The raw correlation is negative, suggesting some reversal, but we still need "
            "regression with controls to know if this is statistically reliable."
        )
else:
    interpretation = (
        "We couldn’t compute a clean correlation because one or both series had no variation."
    )

c2.markdown(f"**Interpretation:** {interpretation}")

# Optional quick plot: average ΔLight and average next-month return over time
if {"brightness_change", "ret_fwd_1m"}.issubset(panel.columns):
    st.markdown("#### Time-Series: Average ΔLight and Next-Month Return")

    ts = (
        panel.groupby("date", as_index=False)[["brightness_change", "ret_fwd_1m"]]
        .mean()
        .sort_values("date")
    )

    ts = ts.rename(
        columns={
            "brightness_change": "Avg ΔLight (HQ counties)",
            "ret_fwd_1m": "Avg next-month return",
        }
    )

    st.line_chart(
        ts.set_index("date"),
        use_container_width=True,
    )
else:
    st.info(
        "Time-series comparison requires both `brightness_change` and `ret_fwd_1m` "
        "columns to be present."
    )

# ----------------------------
# 4. How the regression fits into the story
# ----------------------------
st.subheader("📐 How the Regression Fits Into the Story")

st.markdown("""
We use this overview page to **frame the whole project**:

1. We match **S&P 500 firms** to their **HQ counties**.  
2. We pull **VIIRS night-time light intensity** for those counties, by month.  
3. For each firm-month, we compute:
   - a **brightness level** and a **brightness change (ΔLight)**, and  
   - the **following month’s stock return**.  
4. The raw correlation between ΔLight and next-month return is **very small**, which already suggests
   that any predictive power, if it exists at all, is weak.

The important question is:

> After controlling for **market-wide month effects** (year–month fixed effects),  
> do firms in counties that light up more than usual actually earn **higher or lower returns next month**?

That question is answered rigorously on the **Regression** page, which reports:

- the **β coefficient** on `brightness_change`,  
- its **t-statistic and p-value**, and  
- the **incremental R²** contributed by brightness over and above month fixed effects.

In our results, the β coefficient is **very close to zero** and **not statistically significant**, and the incremental R² is **essentially zero**.  
Thus the regression shows that **night-lights do *not* generate a tradable predictive edge** once we strip out broad market and seasonal forces.
""")

# ----------------------------
# 5. Navigation help – what each page does
# ----------------------------
st.subheader("🧭 How to Use the Rest of the Dashboard")

st.markdown("""
**1. Overview (this page)**  
- Big-picture sample stats  
- Raw correlation between brightness shocks and next-month returns  
- Setup for the main research question  

**2. Ticker Explorer**  
- Drill into a **single firm** at a time  
- See its HQ county, brightness series, and return series  
- Visualize how specific firms behave around “light surprises”  

**3. County Explorer**  
- Analysis at the **county level**  
- Which HQ counties have the strongest brightness–return relationships?  
- County-level leaderboards and summary stats  

**4. Globe**  
- 3D map of the U.S. with **brightness and return overlays**  
- Explore which regions are lighting up and how that relates to returns  

**5. Regression**  
- The **core econometric result**  
- Month fixed-effects regression of next-month returns on brightness change  
- Full β, t-stat, p-value, R² and incremental R², plus a presentation-ready write-up  
""")

st.success(
    "Use this page to introduce the data and intuition, then rely on the other tabs "
    "to tell the detailed story: firm-level, county-level, map-based, and regression-based evidence."
)


