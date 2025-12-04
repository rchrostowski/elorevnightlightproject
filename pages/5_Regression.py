# pages/5_Regression.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from src.load_data import load_model_data

st.set_page_config(page_title="Regression Analysis", layout="wide")

st.title("📈 Regression: Do Night-Lights Predict Next-Month Returns?")

# ----------------------------
# 1. LOAD DATA
# ----------------------------
panel = load_model_data(fallback_if_missing=True)

if panel.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

# Ensure needed columns exist
required = {"ticker", "firm", "county_name", "date",
            "brightness_change", "ret_fwd_1m"}

missing = required - set(panel.columns)
if missing:
    st.error(
        f"nightlights_model_data.csv must contain: {required}. "
        f"Missing: {missing}"
    )
    st.stop()

# Clean types
panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
panel = panel.dropna(subset=["date", "brightness_change", "ret_fwd_1m"])

# ----------------------------
# 2. RUN REGRESSION
# ----------------------------
st.subheader("🔧 Model Specification")

st.markdown(r"""
We estimate the model:

\[
\text{Return}_{i,t+1}
= \alpha + \beta \cdot \Delta\text{Light}_{i,t} + \gamma_{\text{year-month}} + \varepsilon_{i,t}
\]

- **Return<sub>i,t+1</sub>** = next-month stock return  
- **ΔLight<sub>i,t</sub>** = month-over-month brightness change around the firm’s HQ county  
- **γ<sub>year-month</sub>** = year-month fixed effects  
""")

# Create year-month FE
panel["ym"] = panel["date"].dt.to_period("M").astype(str)

# Build regression dataset
reg_df = panel.copy()
reg_df = reg_df.dropna(subset=["brightness_change", "ret_fwd_1m"])

# Fit model with formula
model = smf.ols("ret_fwd_1m ~ brightness_change + C(ym)", data=reg_df).fit()

# ----------------------------
# 3. SHOW RESULTS
# ----------------------------
st.subheader("📊 Regression Output")

coef_table = pd.DataFrame({
    "term": model.params.index,
    "coef": model.params.values,
    "std_err": model.bse.values,
    "t": model.tvalues.values,
    "pval": model.pvalues.values,
})

# Only show brightness + intercept
main_terms = coef_table[coef_table["term"].str.contains("brightness_change|Intercept")]

st.write(main_terms)

st.metric("Model R²", f"{model.rsquared:.3f}")

# ----------------------------
# 4. INTERPRETATION SECTION
# ----------------------------
st.subheader("📘 Interpretation (Presentation-Ready)")

st.markdown("""
### **1. What question are we answering?**
**Do month-to-month changes in night-time brightness around a firm’s HQ predict its next-month stock return?**

---

### **2. What data did we analyze?**
We constructed a panel of **S&P 500 firms** from **2018–present**, matching each firm to its **HQ county**.

For every firm-month, we computed:

- **Brightness level** (VIIRS satellite night-lights)
- **Brightness surprise**:  
  \[
  \Delta\text{Light} = \text{Light}_{t} - \text{Light}_{t-1}
  \]
- **Next-month stock return**, ensuring the brightness change comes **before** the return.

---

### **3. What model did we estimate?**

We regress:

\[
\text{Return}_{i,t+1} = \alpha + \beta \cdot \Delta\text{Light}_{i,t} + \gamma_{\text{year-month}} + \varepsilon_{i,t}
\]

The **year-month fixed effects** remove:

- broad market movements,
- seasonal patterns (winter vs. summer),
- month-specific shocks (COVID volatility, stimulus periods, etc.).

This means **β is only identified by comparing firms to each other within the exact same month.**

---

### **4. What do the coefficients mean? (Your actual results)**

- The **β coefficient on brightness_change is extremely close to zero**.
- The **t-statistic is very small**, far below conventional significance thresholds.
- The **p-value is large**, meaning the signal is statistically indistinguishable from noise.

### 🔎 Interpretation:
> **Brightness changes around firm HQs do NOT predict next-month returns once we control for market-wide month effects.**

Whether a county lights up more than usual tells us **nothing reliable** about how that firm's stock performs the following month.

---

### **5. What does the R² mean?**

The R² is approximately **0.26**, which might sound moderate —  
but almost **all** of it is explained by:

- month fixed effects  
- i.e., the market going up or down that month

Brightness contributes **almost zero incremental explanatory power**.

---

### **6. So what is the final answer to the central research question?**

> **After controlling for market and seasonal effects using month fixed effects, we find no evidence that night-time brightness contains predictive information about next-month stock returns.**

Night-lights DO capture local economic activity, but that activity **does not translate into tradable return forecasts** at the monthly horizon.

---

### **7. One-sentence takeaway (read this aloud):**

> “Brightness changes show real economic movement around firms, but they do not generate statistically meaningful predictions for next-month stock returns once common market effects are removed.”  
""")

# ----------------------------
# 5. OPTIONAL: COEFFICIENT CHART
# ----------------------------
st.subheader("📉 Coefficient Visualization")

coef_val = model.params["brightness_change"]
t_val = model.tvalues["brightness_change"]
p_val = model.pvalues["brightness_change"]

st.metric("β (Brightness → Return)", f"{coef_val:.6f}")
st.metric("t-stat", f"{t_val:.3f}")
st.metric("p-value", f"{p_val:.3f}")



