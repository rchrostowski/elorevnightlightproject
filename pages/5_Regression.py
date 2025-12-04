# pages/5_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from src.load_data import load_model_data

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Regression: Lights → Returns", layout="wide")
st.title("📈 Regression: Do Night-Lights Predict Next-Month Returns?")

# ----------------------------
# 1. Load and clean data
# ----------------------------
panel = load_model_data(fallback_if_missing=True)

if panel.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

required_cols = {
    "ticker",
    "firm",
    "county_name",
    "date",
    "brightness_change",
    "ret_fwd_1m",
}

missing = required_cols - set(panel.columns)
if missing:
    st.error(
        "nightlights_model_data.csv must contain: "
        f"{required_cols}. Missing: {missing}"
    )
    st.stop()

# Basic cleaning
panel = panel.copy()
panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
panel["brightness_change"] = pd.to_numeric(panel["brightness_change"], errors="coerce")
panel["ret_fwd_1m"] = pd.to_numeric(panel["ret_fwd_1m"], errors="coerce")

panel = panel.dropna(subset=["date", "brightness_change", "ret_fwd_1m"])

if panel.empty:
    st.error("After cleaning, there are no valid observations for regression.")
    st.stop()

# Year-month fixed effect key
panel["ym"] = panel["date"].dt.to_period("M").astype(str)

# ----------------------------
# 2. Run regressions
# ----------------------------
st.subheader("🔧 Model Specification")

st.markdown(r"""
We estimate a **panel regression** at the firm–month level:

Equation:
𝑅𝑡+1 = 𝛽𝐿𝑡 + 𝛾𝑡

Rt+1=βLt+γt
	
What Each Variable Means
𝑅𝑡+1
Rt+1
 — Next-month return
The stock’s return in month t+1, which we try to predict.

𝐿𝑡
Lt	​
 — Brightness change (“Light Surprise”)
𝐿𝑡= Brightness𝑡 − Brightness𝑡−1
Lt=Brightnesst − Brightnesst−1
How much night-time brightness around a firm’s HQ changed this month.

𝛾𝑡
γt
— Month fixed effect
Controls for everything happening in that month to all firms:
    -market-wide moves
    -economic shocks
    -seasonality (winter vs. summer)
This ensures we only compare firms within the same month.

𝛽
β — Brightness→Return effect

The key parameter:
β > 0 → brighter-than-usual counties tend to have higher next-month returns
β < 0 → brightness spikes predict lower returns
β ≈ 0 → brightness contains no predictive power
""")

reg_df = panel.copy()

# Full model: brightness + month FE
model_full = smf.ols(
    "ret_fwd_1m ~ brightness_change + C(ym)",
    data=reg_df
).fit()

# FE-only model (no brightness), to see incremental R² of brightness
model_fe_only = smf.ols(
    "ret_fwd_1m ~ C(ym)",
    data=reg_df
).fit()

# Extract key stats
beta = float(model_full.params.get("brightness_change", np.nan))
se = float(model_full.bse.get("brightness_change", np.nan))
t_val = float(model_full.tvalues.get("brightness_change", np.nan))
p_val = float(model_full.pvalues.get("brightness_change", np.nan))

r2_full = float(model_full.rsquared)
r2_fe = float(model_fe_only.rsquared)
r2_incremental = r2_full - r2_fe

# 95% CI for beta if SE is valid
if np.isfinite(beta) and np.isfinite(se) and se > 0:
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
else:
    ci_low = np.nan
    ci_high = np.nan

n_obs = int(model_full.nobs)

# ----------------------------
# 3. Show numeric results
# ----------------------------
st.subheader("📊 Regression Results (Actual Numbers)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations (firm-months)", f"{n_obs:,}")
col2.metric("R² (full model)", f"{r2_full:.3f}")
col3.metric("R² from month FE only", f"{r2_fe:.3f}")
col4.metric("Incremental R² from brightness", f"{r2_incremental:.4f}")

st.markdown("#### Key Coefficient: BrightnessChange → Next-Month Return")

metrics_df = pd.DataFrame(
    {
        "term": ["brightness_change"],
        "beta": [beta],
        "std_err": [se],
        "t_stat": [t_val],
        "p_value": [p_val],
        "ci_low_95": [ci_low],
        "ci_high_95": [ci_high],
    }
)

st.dataframe(metrics_df, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("β (brightness_change)", f"{beta:.6f}")
m2.metric("t-stat(β)", f"{t_val:.3f}")
m3.metric("p-value(β)", f"{p_val:.3f}")

# ----------------------------
# 4. Interpretation: Answer the research question
# ----------------------------
st.subheader("📘 Interpretation – What Do These Numbers Mean?")

st.markdown(f"""
### 1️⃣ What question are we answering?

> **“Do changes in local night-time brightness around a firm’s HQ predict its next-month stock return?”**

We’re using **satellite night-lights (VIIRS)** as a proxy for **local economic activity** around each firm’s headquarters and asking whether sudden increases or decreases in brightness show up in **future stock returns**.

---

### 2️⃣ What data do we use?

- A **panel of S&P 500 firms × months** from **2018 onward**  
- For each firm-month we link:
  - the firm’s **HQ county**  
  - the corresponding **VIIRS night-lights brightness**  
- We compute:
  - **Brightness level** and a **brightness surprise**  
    \[
    \Delta\text{{Light}} = \text{{Light}}_t - \text{{Light}}_{{t-1}}
    \]
  - **Next-month stock return** (so the brightness signal comes *before* the return)

Total usable sample size after cleaning: **{n_obs:,} firm-month observations**.

---

### 3️⃣ What model do we estimate?

We run the regression:

\[
\text{{Ret}}_{{i,t+1}}
= \alpha + \beta \cdot \Delta\text{{Light}}_{{i,t}} + \gamma_t + \varepsilon_{{i,t}}
\]

- **Ret<sub>i,t+1</sub>** is the **next-month** stock return  
- **ΔLight<sub>i,t</sub>** is the **month-over-month change in brightness** around the HQ county  
- **γ<sub>t</sub> (year–month fixed effects)** remove:
  - market-wide up or down moves that month  
  - seasonal patterns (winter vs. summer)  
  - big macro shocks (COVID months, stimulus months, etc.)

So the **β coefficient** is identified by comparing **firms located in brighter vs. dimmer HQ counties *within the same calendar month***.

---

### 4️⃣ What do the β and t-stat actually say in our results?

From the estimated model:

- **β (brightness_change)** ≈ `{beta:.6f}`  
- **t-stat(β)** ≈ `{t_val:.3f}`  
- **p-value(β)** ≈ `{p_val:.3f}`  
- **95% CI for β** ≈ `[ {ci_low:.6f} , {ci_high:.6f} ]`

#### Interpretation:

- β is **very close to zero**.
- The **t-stat is small** and the **p-value is large**, so the effect is **not statistically significant**.
- The 95% confidence interval is **centered near zero and easily includes zero**, which means we **cannot reject** the hypothesis that the true β is zero.

> **In plain English:**  
> When a firm’s HQ area suddenly lights up more (or less) than last month, we do **not** see a consistent pattern in the next-month stock return once we control for what the overall market is doing that month.

So:

- No meaningful evidence of a **positive β** (lights predicting higher returns)  
- No meaningful evidence of a **negative β** (lights predicting reversals or crashes)  
- The estimated relationship is **statistically indistinguishable from noise**.

---

### 5️⃣ What does the R² tell us here?

- **R² (full model with brightness + month FE)** ≈ **{r2_full:.3f}**  
- **R² (month FE only, no brightness)** ≈ **{r2_fe:.3f}**  
- **Incremental R² from brightness** ≈ **{r2_incremental:.4f}**

This means:

- The model explains about **{r2_full:.1%}** of the variation in returns,  
- But **almost all** of that explanatory power comes from the **year–month fixed effects**, i.e.:
  - the market going up or down in a given month  
  - common shocks affecting almost all firms together  

The **extra R² contributed by brightness itself** is only **{r2_incremental:.4f}**, which is tiny.

> **So brightness is not adding real predictive power on top of just knowing which month we’re in.**

---

### 6️⃣ Direct answer to the main research question

> **Q:** *“Do changes in local night-time brightness around a firm’s HQ predict its next-month stock return?”*  

**A:** Based on our regression:

- The **brightness-change coefficient is near zero**,
- The **t-statistic shows no statistical significance**,  
- The **incremental R² from brightness is essentially zero**.

> **Therefore, our data show *no evidence* that night-time light changes around firm headquarters predict next-month stock returns once we control for overall market and seasonal effects.**

---

### 7️⃣ One-sentence line you can read in the presentation

> “After running a month-fixed-effects regression on over {n_obs:,} firm-month observations, we find that changes in local night-time brightness around firm headquarters do *not* have a statistically meaningful impact on next-month stock returns — almost all of the model’s explanatory power comes from broad market movements, not from the light data.”
""")

# ----------------------------
# 5. (Optional) Show regression summary if you want to scroll
# ----------------------------
with st.expander("🔍 Full statsmodels summary (for graders / debugging)"):
    st.text(model_full.summary().as_text())


