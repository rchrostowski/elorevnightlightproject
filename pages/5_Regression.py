# pages/5_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from src.load_data import load_model_data

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nightlights × Returns – Regression",
    layout="wide",
)

st.title("Regression: Do Nightlights Predict Stock Returns?")

st.markdown(
    """
This page runs the **core panel regression** for the project.  

We estimate:

\\[
\\text{Return}_{i,t+1}
= \\alpha
+ \\beta \\cdot \\text{BrightnessChange}_{i,t}
+ \\gamma_{\\text{month}(t)}
+ \\varepsilon_{i,t}
\\]

- **Return** is next-month stock return for firm *i* (forward return).
- **BrightnessChange** is the **change in VIIRS night-lights** around the firm’s HQ county from month *t−1* to *t*.
- **Month fixed effects** (\\(\\gamma_{\\text{month}(t)}\\)) control for market-wide and seasonal effects in each calendar month.
- The key question: **Is \\(\\beta\\) significantly different from 0?**  
  If yes, **brightness changes contain predictive information** about next-month returns beyond broad market moves and seasonality.
"""
)

st.markdown("---")

# -----------------------------------------------------------------------------
# 1. Load data and basic checks
# -----------------------------------------------------------------------------
df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "nightlights_model_data.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py`, commit the updated CSV in "
        "`data/final/`, and redeploy."
    )
    st.stop()

required_cols = {
    "ticker",
    "firm",
    "county_name",
    "date",
    "avg_rad_month",
    "brightness_change",
    "ret",
    "ret_fwd",
}
missing = required_cols - set(df.columns)
if missing:
    st.error(
        "nightlights_model_data.csv must contain the following columns for the "
        "regression page to work:\n\n"
        f"`{sorted(required_cols)}`\n\n"
        f"Missing from your file: `{sorted(missing)}`"
    )
    st.stop()

# If `state` is missing, derive a display version from `state_full` if available
if "state" not in df.columns and "state_full" in df.columns:
    df["state"] = df["state_full"]

# -----------------------------------------------------------------------------
# 2. Clean / prep data for regression
# -----------------------------------------------------------------------------
df = df.copy()

# Dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Choose which return to use
# Prefer forward return (predictive), else fall back to same-month return
if "ret_fwd" in df.columns:
    y_col = "ret_fwd"
    y_label = "Next-month return (ret_fwd)"
else:
    y_col = "ret"
    y_label = "Same-month return (ret)"

# Ensure numeric types
for col in ["brightness_change", y_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["brightness_change", y_col])

# Month fixed effects: Year–Month as a string like "2020-05"
df["year_month"] = df["date"].dt.to_period("M").astype(str)

# Basic sanity
if df.empty or df["brightness_change"].nunique() < 2:
    st.error(
        "Not enough variation in brightness_change or returns after cleaning. "
        "Double-check that `brightness_change` and return columns are populated."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar filters – let user slice data
# -----------------------------------------------------------------------------
st.sidebar.header("Regression Filters")

# State filter
state_options = ["All states"]
if "state" in df.columns:
    state_options += sorted(df["state"].dropna().unique().tolist())

state_choice = st.sidebar.selectbox("Filter by state (HQ county)", state_options)

if state_choice != "All states":
    df = df[df["state"] == state_choice]

# Optional ticker filter
all_tickers = sorted(df["ticker"].unique().tolist())
ticker_options = ["All tickers"] + all_tickers
ticker_choice = st.sidebar.selectbox("Filter by ticker", ticker_options)

if ticker_choice != "All tickers":
    df = df[df["ticker"] == ticker_choice]

# Date range filter
min_date = df["date"].min()
max_date = df["date"].max()

start_date, end_date = st.sidebar.date_input(
    "Sample window (based on signal date)",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

if len(df) < 50:
    st.warning(
        "After filters, there are fewer than 50 observations.\n\n"
        "Loosen the filters (state, ticker, or date range) to get a more stable regression."
    )

# -----------------------------------------------------------------------------
# 4. Run regression: Return ~ BrightnessChange + Month FE
# -----------------------------------------------------------------------------
st.subheader("Regression Results")

st.markdown(
    f"""
We estimate the following **fixed-effects panel regression** on the filtered sample:

\\[
{y_label} = \\alpha + \\beta\\cdot \\text{{BrightnessChange}} + \\gamma_{{\\text{{Year–Month}}}} + \\varepsilon
\\]

- **Year–Month fixed effects** absorb:
  - Overall market movements in that month
  - Seasonal patterns in nightlights and returns  
- The **coefficient on BrightnessChange (β)** compares **counties that are unusually bright vs. dim within the same calendar month.**
"""
)

# Build formula
formula = f"{y_col} ~ brightness_change + C(year_month)"

# We use robust (HC1) standard errors
try:
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
except Exception as e:
    st.error(
        "The regression failed to run, likely due to non-numeric data types "
        "or insufficient variation.\n\n"
        f"Internal error: `{e}`"
    )
    st.stop()

# Summary stats
n_obs = int(model.nobs)
r2 = model.rsquared
r2_adj = model.rsquared_adj

col1, col2, col3 = st.columns(3)
col1.metric("Number of observations", f"{n_obs:,}")
col2.metric("R² (overall fit)", f"{r2:.3f}")
col3.metric("Adjusted R²", f"{r2_adj:.3f}")

# -----------------------------------------------------------------------------
# 5. Coefficient table (with BrightnessChange highlighted)
# -----------------------------------------------------------------------------
summary_table = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})

# Rename columns for display
summary_table = summary_table.rename(
    columns={
        "Coef.": "Coefficient",
        "Std.Err.": "Std. Error",
        "P>|t|": "p-value",
    }
)

# Metric card for key coefficient: brightness_change
brightness_row = summary_table[summary_table["term"] == "brightness_change"]

st.markdown("### Key Coefficient: BrightnessChange")

if not brightness_row.empty:
    row = brightness_row.iloc[0]
    b_hat = row["Coefficient"]
    t_stat = row["t"]
    p_val = row["p-value"]

    c1, c2, c3 = st.columns(3)
    c1.metric("β (BrightnessChange)", f"{b_hat:.4f}")
    c2.metric("t-statistic", f"{t_stat:.2f}")
    c3.metric("p-value", f"{p_val:.3f}")

    st.markdown(
        f"""
**How to read this:**

- **Sign of β**:
  - If β > 0: counties with **strong positive brightness surprises** tend to have **higher** next-month returns, controlling for month.
  - If β < 0: unusually bright months are followed by **lower** returns.
- **t-statistic / p-value** measure **statistical significance**:
  - |t| ≈ 2 and p < 0.05 → statistically significant at the 5% level.
  - Here, β = {b_hat:.4f}, t = {t_stat:.2f}, p = {p_val:.3f}.
"""
    )
else:
    st.warning(
        "The regression did not produce a separate coefficient for `brightness_change`.\n\n"
        "This usually means there was no variation or it was collinear with other terms."
    )

st.markdown("### Full Coefficient Table (including Month Fixed Effects)")

st.caption(
    "Month fixed effects (C(year_month)[T.xxx]) capture broad market + seasonal "
    "patterns. We mainly care about the row labelled `brightness_change`."
)
st.dataframe(summary_table, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Visualization: Residualized (FE-adjusted) relationship
# -----------------------------------------------------------------------------
st.markdown("### Visualization: Brightness vs. Month-Adjusted Returns")

st.markdown(
    """
To visualize the regression after controlling for seasonality, we create a **partial regression plot**:

1. **Step 1:** Regress returns on **month fixed effects only** and take the residuals  
   → “Month-adjusted returns” (what’s left after removing average return in each calendar month).
2. **Step 2:** Regress brightness_change on **month fixed effects only** and take the residuals  
   → “Month-adjusted brightness surprises.”
3. **Step 3:** Plot these residuals against each other.  
   The slope of the best-fit line in this plot is the same β as in the full regression.
"""
)

# Month-adjusted returns
fe_returns = smf.ols(f"{y_col} ~ C(year_month)", data=df).fit()
df["ret_fe_resid"] = fe_returns.resid

# Month-adjusted brightness
fe_bright = smf.ols("brightness_change ~ C(year_month)", data=df).fit()
df["bright_fe_resid"] = fe_bright.resid

# Sample a subset if huge
plot_df = df.dropna(subset=["ret_fe_resid", "bright_fe_resid"]).copy()
if len(plot_df) > 5000:
    plot_df = plot_df.sample(5000, random_state=42)

import plotly.express as px

fig = px.scatter(
    plot_df,
    x="bright_fe_resid",
    y="ret_fe_resid",
    opacity=0.4,
    trendline="ols",
    labels={
        "bright_fe_resid": "BrightnessChange (month-adjusted)",
        "ret_fe_resid": f"{y_label} (month-adjusted)",
    },
    title="Partial Regression: Nightlights Surprise vs. Month-Adjusted Returns",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**Interpretation of the plot:**

- Each point is a **firm-month** in your sample.
- The x-axis shows how **unusually bright or dim** the county is, compared to the average in that calendar month.
- The y-axis shows how **unusually high or low** the return is, compared to the average return in that month.
- The **trendline** is the regression of FE-adjusted returns on FE-adjusted brightness.  
  Its slope equals the β from the main regression.

If the cloud of points slopes upward, it visually supports a **positive β** (brighter → higher returns).  
If it slopes downward, it supports a **negative β**.
"""
)

# -----------------------------------------------------------------------------
# 7. “How to explain this slide out loud” section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### How to Explain This Slide in the Presentation")

st.markdown(
    """
**1. What question are we answering?**  
> *“Do changes in local night-time brightness around a firm’s HQ predict its next-month stock return?”*

**2. What data do we use?**

- Panel of **S&P 500 firms × months** from 2018 onward.
- For each firm-month, we link the firm’s **HQ county** to VIIRS **night-lights**.
- We compute:
  - A **brightness level** and a **brightness surprise** (ΔLight = month-over-month change).
  - A **next-month stock return** (so the signal comes before the return).

**3. What model do we estimate?**

- We regress **next-month return** on **brightness surprise** plus **year-month fixed effects**.
- The fixed effects remove:
  - broad **market moves** in that month,
  - **seasonal patterns** (e.g., winter vs. summer, COVID period shocks, etc.).

So the **brightness coefficient** is identified by **comparing counties that are unusually bright vs. dim *within the same month***.

**4. How do we interpret the β coefficient?**

- If β is significantly **positive**:
  - Firms whose local area suddenly lights up more than usual tend to have **higher future returns**.
- If β is significantly **negative**:
  - Brightness spikes are followed by **lower returns**, suggesting a reversal or overreaction.
- If β is **close to zero and not significant**:
  - Night-lights do **not** add much predictive power beyond standard month effects.

**5. What does R² tell us?**

- R² is around the value shown at the top (e.g., ≈0.28 for the full sample).
- That means the model explains about that fraction of the **cross-sectional and time-series variation** in returns once we include month fixed effects.
- Most of that R² comes from **month fixed effects** (market/seasonality);  
  the incremental contribution of brightness is captured by the **β and its t-stat**.

**6. Big picture summarizing line:**  
> *“We control for broad market and seasonal effects using month fixed effects, then ask whether local economic activity—proxied by night-time brightness—helps explain which stocks outperform next month. The coefficient on **BrightnessChange** and its t-stat tell us whether there is a statistically meaningful link between lights and returns.”*
"""
)

