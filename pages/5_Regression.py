# pages/5_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.formula.api as smf

from src.load_data import load_model_data

st.set_page_config(page_title="Regression Results", page_icon="📈")

st.title("📈 Regression: Return ~ Brightness + Month Fixed Effects")

# ----------------- LOAD MODEL DATA -----------------

df = load_model_data(fallback_if_missing=True).copy()
if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]

if "date" not in df.columns:
    st.error("Model data must have a 'date' column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# guess brightness and return variables
brightness_col = None
for cand in ["brightness_change", "d_light", "brightness", "avg_rad_month", "avg_brightness"]:
    if cand in df.columns:
        brightness_col = cand
        break

ret_col = None
for cand in ["ret_excess", "excess_ret", "mkt_excess_ret", "risk_excess", "ret_fwd_1m", "ret"]:
    if cand in df.columns:
        ret_col = cand
        break

if brightness_col is None or ret_col is None:
    st.error(
        "Could not find both a brightness column and a return column.\n\n"
        f"Brightness candidates: brightness_change, d_light, brightness, avg_rad_month, avg_brightness\n"
        f"Return candidates: ret_excess, excess_ret, mkt_excess_ret, risk_excess, ret_fwd_1m, ret\n"
        f"Columns found: {df.columns.tolist()}"
    )
    st.stop()

df["year_month"] = df["date"].dt.to_period("M").astype(str)

# ----------------- SIDEBAR FILTERS -----------------

with st.sidebar:
    st.header("Sample filters")

    # Date window
    min_d, max_d = df["date"].min(), df["date"].max()
    start, end = st.date_input(
        "Date window",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )
    if isinstance(start, tuple):  # streamlit quirk
        start, end = start

    # optional ticker filter
    ticker_col = "ticker" if "ticker" in df.columns else None
    if ticker_col:
        all_tickers = sorted(df[ticker_col].unique().tolist())
        chosen = st.multiselect(
            "Limit to tickers (optional)",
            options=all_tickers,
            default=[],
        )
    else:
        chosen = []

# apply filters
mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
if ticker_col and chosen:
    mask &= df[ticker_col].isin(chosen)

reg_df = df.loc[mask, ["date", "year_month", brightness_col, ret_col]].dropna()
n_obs = len(reg_df)

if n_obs < 30:
    st.error(f"Not enough observations after filters (n = {n_obs}). Loosen the filters.")
    st.stop()

st.markdown(
    f"Using **{n_obs:,} observations** from "
    f"{reg_df['date'].min().strftime('%Y-%m')} → {reg_df['date'].max().strftime('%Y-%m')}."
)

# ----------------- RUN REGRESSION -----------------
# Ret ~ Brightness + C(YearMonth)

formula = f"{ret_col} ~ {brightness_col} + C(year_month)"
model = smf.ols(formula=formula, data=reg_df)
res = model.fit(cov_type="HC1")  # robust SEs

# build tidy coefficient table
coef_df = pd.DataFrame({
    "term": res.params.index,
    "coef": res.params.values,
    "std_err": res.bse.values,
    "t": res.tvalues.values,
    "pval": res.pvalues.values,
})

# separate brightness coefficient and fixed effects
is_bright = coef_df["term"] == brightness_col
is_intercept = coef_df["term"] == "Intercept"

coef_main = coef_df[is_intercept | is_bright].copy()
coef_fe = coef_df[~is_intercept & ~is_bright].copy()

st.subheader("Key coefficient: brightness")

if brightness_col in coef_main["term"].values:
    row = coef_main[coef_main["term"] == brightness_col].iloc[0]
    st.write(
        f"**Regression:** `{ret_col} ~ {brightness_col} + C(year_month)` "
        "(return on brightness, controlling for month fixed effects)."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("β (brightness)", f"{row['coef']:.4f}")
    with c2:
        st.metric("t-stat", f"{row['t']:.2f}")
    with c3:
        st.metric("p-value", f"{row['pval']:.3f}")
else:
    st.warning("Brightness term not found in coefficient table (this should not happen).")

# full table
st.subheader("All coefficients (incl. month fixed effects)")

coef_df_rounded = coef_df.copy()
coef_df_rounded["coef"] = coef_df_rounded["coef"].round(5)
coef_df_rounded["std_err"] = coef_df_rounded["std_err"].round(5)
coef_df_rounded["t"] = coef_df_rounded["t"].round(3)
coef_df_rounded["pval"] = coef_df_rounded["pval"].round(4)

st.dataframe(
    coef_df_rounded.rename(
        columns={"term": "Term", "coef": "Coef", "std_err": "Std.Err", "pval": "P>|t|"}
    ),
    use_container_width=True,
)

# model stats
st.subheader("Model fit")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("R²", f"{res.rsquared:.3f}")
with c2:
    st.metric("Adj. R²", f"{res.rsquared_adj:.3f}")
with c3:
    st.metric("N", f"{n_obs:,}")

# ----------------- SCATTER PLOT -----------------

st.subheader("Scatter: brightness vs return")

scatter_df = reg_df.copy()
scatter_df["year_month"] = scatter_df["year_month"].astype(str)

chart = (
    alt.Chart(scatter_df.sample(min(len(scatter_df), 5000)))
    .mark_circle(size=20, opacity=0.4)
    .encode(
        x=alt.X(f"{brightness_col}:Q", title="Brightness"),
        y=alt.Y(f"{ret_col}:Q", title="Return"),
        color=alt.Color("year_month:N", title="Year-month", legend=None),
        tooltip=["date:T", f"{brightness_col}:Q", f"{ret_col}:Q"],
    )
    .properties(height=350)
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "OLS with month fixed effects: compares **brighter vs darker counties within the same calendar month**, "
    "which removes seasonality in night lights."
)

