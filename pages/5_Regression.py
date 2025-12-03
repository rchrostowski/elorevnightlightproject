# pages/5_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.load_data import load_model_data

st.markdown("## 📊 Regression: Returns on Night-Lights (with Month Fixed Effects)")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset is missing. Run `python scripts/build_all.py` first.")
    st.stop()

required = {"date", "brightness_change", "ret_fwd_1m"}
if not required.issubset(df.columns):
    st.error(f"`nightlights_model_data` must contain at least: {required}")
    st.stop()

# Clean
df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "brightness_change", "ret_fwd_1m"])

# Explain what "ret" is
st.markdown(
    """
We estimate the regression:

\\[
\\text{ret\_{fwd, i,t}} = \\alpha + \\beta \\cdot \\Delta \\text{Brightness}\_{i,t}
+ \\gamma_{\\text{month}(t)} + \\varepsilon\_{i,t}
\\]

- **ret_fwd_1m** = ticker's **next-month total return** (not excess over the risk-free rate).  
- **brightness_change** = change in VIIRS night-lights for the HQ county from month \\(t-1\\) to \\(t\\).  
- \\(\\gamma_{\\text{month}(t)}\\) are **calendar month fixed effects** (C(year-month)), which:
  - compare **brighter vs darker counties *within the same calendar month***,
  - soak up seasonality, business-cycle effects, etc.
"""
)

# --- Raw correlation (no controls) ---

corr = df["brightness_change"].corr(df["ret_fwd_1m"])
st.markdown("### 1. Raw correlation (no controls)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Corr(ΔBrightness, next-month return)", f"{corr:.3f}")
with col2:
    st.caption(
        "This is just the plain correlation across all ticker-months, with **no controls**.\n"
        "The professor's FE regression is the 'seasonality-adjusted' version of this."
    )

# --- 2. Fixed-effects regression: ret_fwd_1m ~ brightness_change + C(year-month) ---

st.markdown("### 2. Fixed-effects regression (C(year-month))")

df["ym"] = df["date"].dt.to_period("M").astype(str)

# Build design matrix: brightness + month dummies
month_dummies = pd.get_dummies(df["ym"], prefix="ym", drop_first=True)
X = pd.concat([df[["brightness_change"]], month_dummies], axis=1)
X = sm.add_constant(X)
y = df["ret_fwd_1m"]

model = sm.OLS(y, X)
results = model.fit()

# Extract main coefficient (brightness_change)
beta = results.params.get("brightness_change", np.nan)
se_beta = results.bse.get("brightness_change", np.nan)
t_beta = results.tvalues.get("brightness_change", np.nan)
p_beta = results.pvalues.get("brightness_change", np.nan)

coef_df = pd.DataFrame(
    {
        "term": ["brightness_change"],
        "Coef.": [beta],
        "Std.Err.": [se_beta],
        "t": [t_beta],
        "P>|t|": [p_beta],
    }
)

colA, colB = st.columns([1.2, 2.8])

with colA:
    st.markdown("#### Key coefficient (within-month comparison)")
    st.dataframe(coef_df, use_container_width=True)

with colB:
    st.markdown("#### Model summary")
    st.markdown(
        f"""
- **Observations**: {int(results.nobs):,}  
- **R²**: {results.rsquared:.3f}  
- **Adj. R²**: {results.rsquared_adj:.3f}  

Interpretation of **β (brightness_change)**:

- Compares **counties that are relatively brighter vs darker *within the same year-month***  
- If β > 0: brighter counties in a given month tend to have **higher next-month returns**  
- If β < 0: brighter counties in a given month tend to have **lower next-month returns**
"""
    )

# Optional: show a short FE vs no-FE comparison
st.markdown("### 3. How this relates to correlation")

st.markdown(
    """
- The simple correlation above mixes together:
  - cross-sectional differences (some months are crazy, some chill), and  
  - seasonality / market-wide shocks.  

- The FE regression is like a **seasonality-adjusted correlation**:
  - Within each month, it looks at **which HQ counties got brighter/dimmer**  
  - and asks whether those firms had systematically different **next-month returns**.
"""
)

# --- Optional: show a few month fixed effects ---

show_fe = st.checkbox("Show a few month fixed effects (γ_month)", value=False)

if show_fe:
    fe_params = results.params[[c for c in results.params.index if c.startswith("ym_")]]
    fe_df = (
        fe_params.rename_axis("month_dummy")
        .reset_index(name="Coef.")
        .sort_values("Coef.")
    )
    st.markdown("#### Example month fixed effects (relative to base month)")
    st.dataframe(fe_df.head(10), use_container_width=True)
