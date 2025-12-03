import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.load_data import load_model_data

st.markdown("## 5. Regression – Returns on HQ Night-Lights with Month Fixed Effects")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset is missing. Run `python scripts/build_all.py` first.")
    st.stop()

required = {"date", "brightness_change", "ret_fwd_1m"}
if not required.issubset(df.columns):
    st.error(f"`nightlights_model_data` must contain at least: {required}")
    st.stop()

# --- Clean & ensure numeric ---

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Force numeric for regression variables
df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")

# Drop rows with missing anything we need
df = df.dropna(subset=["date", "brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error(
        "No valid rows remaining after cleaning `date`, `brightness_change`, and `ret_fwd_1m`.\n"
        "Check that these columns are present and numeric in `nightlights_model_data.csv`."
    )
    st.stop()

# Explain what "ret" is
st.markdown(
    r"""
We estimate the regression:

\[
\text{ret\_{fwd, i,t}} = \alpha + \beta \cdot \Delta \text{Brightness}\_{i,t}
+ \gamma_{\text{month}(t)} + \varepsilon_{i,t}
\]

Where:

- **Dependent variable**: `ret_fwd_1m` = ticker’s **next-month total return**  
  (we are *not* subtracting the risk-free rate or market return – this is made explicit).  
- **Key regressor**: `brightness_change` = change in HQ county night-lights from month \(t-1\) to \(t\).  
- \(\gamma_{\text{month}(t)}\) = **year–month fixed effects** `C(year-month)`:
  - compares **bright vs dark HQ counties within the *same calendar month***,  
  - controls for **seasonality** and macro effects in that month.

The coefficient **β** answers:

> Holding the calendar month fixed, do firms in HQ counties that brighten more tend to have higher next-month returns?
"""
)

# --- 1. Raw correlation (no controls) ---

corr = df["brightness_change"].corr(df["ret_fwd_1m"])
st.markdown("### 1. Raw correlation (no controls)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Corr(ΔBrightness, next-month return)", f"{corr:.3f}")
with col2:
    st.caption(
        "This is the plain correlation across all ticker–months, with **no controls**.\n"
        "The fixed-effects regression below is the 'seasonality-adjusted' version of this."
    )

# --- 2. Fixed-effects regression: ret_fwd_1m ~ brightness_change + C(year-month) ---

st.markdown("### 2. Fixed-effects regression (C(year-month))")

# Year-month label for fixed effects
df["ym"] = df["date"].dt.to_period("M").astype(str)

# Month dummies (drop_first=True to avoid dummy trap)
month_dummies = pd.get_dummies(df["ym"], prefix="ym", drop_first=True)

# Design matrix: constant + brightness_change + month dummies
X = pd.concat([df[["brightness_change"]], month_dummies], axis=1)
X = sm.add_constant(X)

# Ensure everything is float (statsmodels hates object dtype)
X = X.astype(float)
y = df["ret_fwd_1m"].astype(float)

# Fit OLS
model = sm.OLS(y.values, X.values)
results = model.fit()

# Rebuild param index using our column names
params = pd.Series(results.params, index=X.columns)
bse = pd.Series(results.bse, index=X.columns)
tvals = pd.Series(results.tvalues, index=X.columns)
pvals = pd.Series(results.pvalues, index=X.columns)

# Extract main coefficient (brightness_change)
beta = params.get("brightness_change", np.nan)
se_beta = bse.get("brightness_change", np.nan)
t_beta = tvals.get("brightness_change", np.nan)
p_beta = pvals.get("brightness_change", np.nan)

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

**Interpretation of β (brightness_change):**

- Compares **counties that are relatively brighter vs darker within the *same year–month***.  
- If β > 0: brighter HQ counties in a given month tend to have **higher next-month returns**.  
- If β < 0: brighter HQ counties in a given month tend to have **lower next-month returns**.
"""
    )

# --- 3. How this relates to simple correlation ---

st.markdown("### 3. How this relates to correlation")

st.markdown(
    """
- The simple correlation mixes together:
  - cross-sectional differences across months, and  
  - seasonality / market-wide shocks.  

- The fixed-effects regression is like a **seasonality-adjusted correlation**:
  - Within each calendar month, it looks at **which HQ counties got brighter/dimmer**,  
  - and asks whether those firms’ stocks had systematically different **next-month returns**.
"""
)

# --- Optional: show some month fixed effects ---

show_fe = st.checkbox("Show a few month fixed effects (γ_month)", value=False)

if show_fe:
    fe_mask = params.index.str.startswith("ym_")
    fe_params = params[fe_mask]
    fe_df = (
        fe_params.rename_axis("month_dummy")
        .reset_index(name="Coef.")
        .sort_values("Coef.")
    )
    st.markdown("#### Example month fixed effects (relative to base month)")
    st.dataframe(fe_df.head(10), use_container_width=True)

