# pages/5_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
from math import erf, sqrt

from src.load_data import load_model_data

st.title("Regression: Returns on Nightlights (with Month Fixed Effects)")

df = load_model_data(fallback_if_missing=True).copy()
if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# We use next-month returns as the dependent variable
df = df.dropna(subset=["brightness_change", "ret_fwd"]).copy()
df["month"] = df["date"].dt.to_period("M").astype(str)

st.sidebar.header("Regression options")
year_min = int(df["date"].dt.year.min())
year_max = int(df["date"].dt.year.max())

year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

df = df[(df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])]

if len(df) < 30:
    st.warning("Not enough observations in this period to run a meaningful regression.")
    st.stop()

st.markdown(
    """
We estimate:

\\[
\\text{Ret}_{i,t+1} = \\alpha + \\beta \\cdot \\Delta \\text{Light}_{i,t} + \\gamma_t + \\varepsilon_{i,t},
\\]

where:

- \\(\\Delta \\text{Light}_{i,t}\\) = change in average nightlights around firm *i*'s HQ county in month *t*
- \\(\\gamma_t\\) = **month fixed effects** (one dummy per calendar month, which removes seasonality)
- Dependent variable is **next-month raw stock return** (not excess return).
"""
)

# ---------- Within (demeaned) regression: Ret_fwd ~ brightness_change + month FE ----------

x = df["brightness_change"].astype(float)
y = df["ret_fwd"].astype(float)

# Demean by month (fixed effects)
g = df.groupby("month")
x_tilde = x - g["brightness_change"].transform("mean")
y_tilde = y - g["ret_fwd"].transform("mean")

valid = x_tilde.notna() & y_tilde.notna()
x_tilde = x_tilde[valid]
y_tilde = y_tilde[valid]

n = len(x_tilde)
if n < 10:
    st.warning("Not enough non-missing observations after demeaning.")
    st.stop()

# OLS on demeaned data (no intercept)
xx = np.sum(x_tilde ** 2)
xy = np.sum(x_tilde * y_tilde)

beta = xy / xx
resid = y_tilde - beta * x_tilde

# sigma^2 and standard error
sigma2 = np.sum(resid ** 2) / (n - 1)  # rough dof; FE exact dof is more complicated
var_beta = sigma2 / xx
se_beta = np.sqrt(var_beta)

t_stat = beta / se_beta
# Normal approximation for p-value
def normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2.0)))

p_value = 2 * (1 - normal_cdf(abs(t_stat)))

st.markdown("### Coefficient on ΔBrightness (within month)")

coef_table = pd.DataFrame(
    {
        "term": ["brightness_change"],
        "Coef.": [beta],
        "Std.Err.": [se_beta],
        "t": [t_stat],
        "P>|t|": [p_value],
    }
)

st.dataframe(coef_table.style.format({"Coef.": "{:.4f}", "Std.Err.": "{:.4f}", "t": "{:.2f}", "P>|t|": "{:.3f}"}),
             use_container_width=True)

st.markdown("### Scatter of demeaned variables (after month fixed effects)")

df_plot = pd.DataFrame({"x_tilde": x_tilde, "y_tilde": y_tilde})

import altair as alt

scatter = (
    alt.Chart(df_plot)
    .mark_circle(opacity=0.4)
    .encode(
        x=alt.X("x_tilde:Q", title="ΔBrightness (demeaned within month)"),
        y=alt.Y("y_tilde:Q", title="Next-month return (demeaned within month)"),
    )
)

line = (
    alt.Chart(df_plot)
    .mark_line(color="red")
    .transform_regression("x_tilde", "y_tilde")
    .encode(x="x_tilde", y="y_tilde")
)

st.altair_chart(alt.layer(scatter, line), use_container_width=True)

st.caption(
    "The coefficient above measures how much *within-month* differences in ΔBrightness "
    "across HQ counties are associated with *within-month* differences in next-month returns. "
    "Because we control for month fixed effects, this removes seasonality."
)
