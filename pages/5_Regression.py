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

# ---------- Clean & ensure numeric ----------
df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")

df = df.dropna(subset=["date", "brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error(
        "No valid rows remaining after cleaning `date`, `brightness_change`, and `ret_fwd_1m`.\n"
        "Check that these columns are present and numeric in `nightlights_model_data.csv`."
    )
    st.stop()

st.markdown(
    r"""
We estimate the following regression:

\[
\text{ret\_{fwd, i,t}} = \alpha + \beta \cdot \Delta \text{Brightness}\_{i,t}
+ \gamma_{\text{year-month}(t)} + \varepsilon_{i,t}
\]

Where:

- **Dependent variable**: `ret_fwd_1m`  
  - This is the **total next-month return** on ticker \(i\) (we are not subtracting market or risk-free returns).  
- **Key regressor**: `brightness_change`  
  - The change in HQ **county night-lights** from month \(t-1\) to \(t\).  
- **Fixed effects**: \(\gamma_{\text{year-month}(t)}\)  
  - Implemented as **year–month dummies** `C(year-month)`.  
  - They compare **bright vs dark HQ counties *within the same calendar month***, and absorb:
    - seasonal patterns, and  
    - common macro / market shocks.

Our main question:

> “After controlling for the calendar month, do firms in HQ counties that brighten more tend to have higher next-month returns?”
"""
)

# ---------- 1. Raw correlation ----------
st.markdown("### 1. Raw correlation (no controls)")

corr = df["brightness_change"].corr(df["ret_fwd_1m"])

col1, col2 = st.columns([1.1, 2.9])
with col1:
    st.metric("Corr(ΔBrightness, next-month return)", f"{corr:.3f}")
with col2:
    st.caption(
        "This is the simple correlation, with **no fixed effects**. "
        "It mixes together cross-sectional, seasonal, and macro effects."
    )

# ---------- 2. Fixed-effects regression ----------
st.markdown("### 2. Fixed-effects regression: ret_fwd_1m ~ brightness_change + C(year-month)")

# Year-month label for fixed effects
df["ym"] = df["date"].dt.to_period("M").astype(str)

# Month dummies (drop one to avoid dummy trap)
month_dummies = pd.get_dummies(df["ym"], prefix="ym", drop_first=True)

X = pd.concat([df[["brightness_change"]], month_dummies], axis=1)
X = sm.add_constant(X)

X = X.astype(float)
y = df["ret_fwd_1m"].astype(float)

model = sm.OLS(y.values, X.values)
results = model.fit()

params = pd.Series(results.params, index=X.columns)
bse = pd.Series(results.bse, index=X.columns)
tvals = pd.Series(results.tvalues, index=X.columns)
pvals = pd.Series(results.pvalues, index=X.columns)

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
    st.markdown("#### Key coefficient on ΔBrightness")
    st.dataframe(coef_df, use_container_width=True)

with colB:
    st.markdown("#### Model summary and interpretation")
    st.markdown(
        f"""
- **Observations**: {int(results.nobs):,}  
- **R²**: {results.rsquared:.3f}  
- **Adj. R²**: {results.rsquared_adj:.3f}  

**How to interpret β (`brightness_change`):**

- This coefficient compares **HQ counties that are relatively brighter vs relatively darker *within the same year–month***.  
- A **positive** β would mean:
  - In a given calendar month, firms in HQ counties that brighten more tend to have **higher next-month returns**.  
- A **negative** β would mean:
  - In a given calendar month, firms in HQ counties that brighten more tend to have **lower next-month returns**.  

Because we include **year–month fixed effects**, β is **not** driven by things like:

- Christmas or holiday lights,  
- COVID months,  
- Fed events, etc.

Those broad month-level shocks are absorbed by the fixed effects.
"""
    )

# ---------- 3. Connect back to the rest of the dashboard ----------
st.markdown("### 3. How this regression connects to the other tabs")

st.markdown(
    """
- **Overview tab**: shows the **distribution** of `brightness_change` and `ret_fwd_1m`, plus the **raw scatter** between them.  
  - That’s like a **simple correlation view** with no controls.  

- **Ticker Explorer**: zooms in on a **single firm**.  
  - You see the firm’s **HQ brightness changes over time** and its **next-month returns**.  
  - The mini scatter for that firm is like a small, firm-level version of this regression.  

- **County Explorer**: zooms in on a **single county**.  
  - It shows which HQ firms are located there and how their returns behave when the **county’s lights change**.  

- **Regression tab (this one)**:  
  - pools **all firms and all counties**,  
  - uses **`ret_fwd_1m` as the dependent variable**,  
  - uses **`brightness_change` as the main predictor**,  
  - and includes **year–month fixed effects** `C(year-month)` to provide the cleanest estimate of the relationship.
"""
)

# Optional: show some month fixed effects
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
    st.dataframe(fe_df.head(15), use_container_width=True)

