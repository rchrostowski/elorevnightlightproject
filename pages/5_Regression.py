# pages/5_Regression.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.set_page_config(
    page_title="Regression Lab – Night Lights Anomalia",
    layout="wide",
)

def _get_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None

df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

brightness_col = _get_col(df, ["brightness_change", "d_light", "delta_light"], required=True)
ret_fwd_col = _get_col(df, ["ret_fwd_1m", "ret_fwd", "ret_forward_1m"], required=True)

st.markdown("## Regression Lab – Does ΔLight Predict Next-Month Returns?")

st.markdown(
    """
    We estimate regressions of the form

    \\[
    r_{i,t+1} = \\alpha + \\beta \\Delta Light_{i,t} + \\gamma_{month} + \\varepsilon_{i,t+1}
    \\]

    where:
    - \\( r_{i,t+1} \\): next-month return (possibly excess return)  
    - \\( \\Delta Light_{i,t} \\): change in HQ county brightness  
    - \\( \\gamma_{month} \\): calendar month fixed effects (Jan-2018, Feb-2018, …)  

    This is exactly what your professor described: comparing **bright vs dim HQ counties
    within the same calendar month**, so all seasonality is absorbed by the month dummies.
    """
)

# ---------- Build panel data with month FE ----------

df_reg = df[[brightness_col, ret_fwd_col, "date", "ticker"]].dropna().copy()

df_reg["ym"] = df_reg["date"].dt.to_period("M").astype(str)

st.write(f"Usable obs after dropping missing: **{len(df_reg):,}**")

# ---------- Run OLS with month fixed effects ----------

try:
    import statsmodels.formula.api as smf
except ImportError:
    st.error(
        "statsmodels is not installed in this environment. "
        "Add `statsmodels` to your requirements and redeploy, "
        "or run the regression locally in a notebook."
    )
    st.stop()

formula = f"{ret_fwd_col} ~ {brightness_col} + C(ym)"

with st.spinner("Fitting OLS with month fixed effects..."):
    model = smf.ols(formula, data=df_reg).fit(cov_type="HC1")

beta = model.params.get(brightness_col, float("nan"))
t_beta = model.tvalues.get(brightness_col, float("nan"))
p_beta = model.pvalues.get(brightness_col, float("nan"))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("β on ΔLight", f"{beta:.4f}")
with c2:
    st.metric("t-stat(β)", f"{t_beta:.2f}")
with c3:
    st.metric("R²", f"{model.rsquared:.3f}")

st.markdown("### Regression output (key coefficients)")

# Show only the ΔLight row + a couple of largest month FE’s to keep it readable
coefs = model.summary2().tables[1].reset_index()
coefs = coefs.rename(columns={"index": "term"})
coefs["is_brightness"] = coefs["term"] == brightness_col
coefs_sorted = pd.concat([
    coefs[coefs["is_brightness"]],
    coefs[~coefs["is_brightness"]].sort_values("Coef.", key=lambda s: s.abs(), ascending=False).head(10),
])

st.dataframe(coefs_sorted[["term", "Coef.", "Std.Err.", "t", "P>|t|"]], use_container_width=True)

st.markdown("### Interpretation notes")

st.markdown(
    f"""
    - **β on ΔLight = {beta:.4f}**:  
      On average, a 1-unit increase in HQ county ΔLight in month *t* is associated with
      an increase of about **{beta:.4f}** in next-month returns, holding calendar month fixed.

    - The month fixed effects \\(\\gamma_{{month}}\\) absorb:
      - Seasonality in night lights (e.g., darker winter months)
      - Market-wide shocks that line up with calendar months

    - What you’d say in the write-up:
      - “We run a panel regression of next-month returns on the change in night-time light
        in the HQ county, including year-month fixed effects. This compares brighter vs dimmer
        HQ counties **within the same month**, so the coefficient on ΔLight is not driven by
        winter vs summer or market-wide events.”

    """
)
