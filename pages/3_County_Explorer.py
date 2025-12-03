# pages/3_County_Explorer.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.markdown("## 🏙 County Explorer (HQ Counties)")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset is missing. Run `python scripts/build_all.py` first.")
    st.stop()

# --- Robust detection of county/state columns ---

county_col = None
for c in ["county_name", "county", "county_label"]:
    if c in df.columns:
        county_col = c
        break

state_col = None
for c in ["state", "state_abbr", "state_code", "state_name"]:
    if c in df.columns:
        state_col = c
        break

if county_col is None:
    st.error(
        "`nightlights_model_data` must contain a county column "
        "(one of: county_name, county, county_label)."
    )
    st.stop()

if "date" not in df.columns:
    st.error("`nightlights_model_data` must contain a `date` column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Create a nice label for selection
if state_col:
    df["county_label"] = df[county_col].astype(str) + ", " + df[state_col].astype(str)
else:
    df["county_label"] = df[county_col].astype(str)

# --- Sidebar: pick county and date window ---

st.sidebar.header("County filters")

county_options = sorted(df["county_label"].unique())
default_label = "Santa Clara County, CA"
if default_label not in county_options and county_options:
    default_label = county_options[0]

county_sel = st.sidebar.selectbox(
    "HQ county",
    options=county_options,
    index=county_options.index(default_label) if default_label in county_options else 0,
)

df_c = df[df["county_label"] == county_sel].copy()
if df_c.empty:
    st.warning("No observations for that county.")
    st.stop()

date_min = df_c["date"].min()
date_max = df_c["date"].max()
start, end = st.sidebar.slider(
    "Date window",
    min_value=date_min.to_pydatetime(),
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    format="YYYY-MM",
)

df_c = df_c[df_c["date"].between(start, end)].copy()
if df_c.empty:
    st.warning("No observations in that date window for this county.")
    st.stop()

# --- Summary metrics ---

n_tickers = df_c["ticker"].nunique() if "ticker" in df_c.columns else 0
avg_bright = df_c["avg_rad_month"].mean() if "avg_rad_month" in df_c.columns else float("nan")
avg_dlight = df_c["brightness_change"].mean() if "brightness_change" in df_c.columns else float("nan")
avg_ret = df_c["ret_fwd_1m"].mean() if "ret_fwd_1m" in df_c.columns else float("nan")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tickers HQ'd here", n_tickers)
with col2:
    if pd.notna(avg_dlight):
        st.metric("Avg Δbrightness", f"{avg_dlight:.2f}")
    else:
        st.metric("Avg Δbrightness", "n/a")
with col3:
    if pd.notna(avg_ret):
        st.metric("Avg next-month return", f"{avg_ret:.2%}")
    else:
        st.metric("Avg next-month return", "n/a")

# --- Firms headquartered in this county ---

st.markdown("### Firms headquartered in this county")

if {"ticker", "firm"}.issubset(df_c.columns):
    firms_table = (
        df_c[["ticker", "firm"]]
        .drop_duplicates()
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    st.table(firms_table)
else:
    st.info("Ticker / firm information not available in this dataset.")

# --- County-level time series ---

st.markdown("### County time series: brightness vs returns")

plot_cols = []
labels = {}

if "avg_rad_month" in df_c.columns:
    plot_cols.append("avg_rad_month")
    labels["avg_rad_month"] = "Brightness level (avg_rad_month)"

if "brightness_change" in df_c.columns:
    plot_cols.append("brightness_change")
    labels["brightness_change"] = "ΔBrightness (month-to-month)"

if "ret_fwd_1m" in df_c.columns:
    plot_cols.append("ret_fwd_1m")
    labels["ret_fwd_1m"] = "Next-month stock return"

if plot_cols:
    ts = df_c[["date"] + plot_cols].set_index("date").rename(columns=labels)
    st.line_chart(ts)
    st.caption(
        "Lines show how **county brightness** (level and monthly change) and "
        "**next-month returns** evolve for HQ firms in this county."
    )
else:
    st.info("No brightness / return columns available to plot.")

# --- Optional scatter: ΔBrightness vs returns, per county ---

if {"brightness_change", "ret_fwd_1m"}.issubset(df_c.columns):
    st.markdown("### Scatter: ΔBrightness vs next-month return (county level)")

    scat_c = df_c[["brightness_change", "ret_fwd_1m"]].dropna()
    if not scat_c.empty:
        corr_c = scat_c["brightness_change"].corr(scat_c["ret_fwd_1m"])
        st.write(f"Correlation in **{county_sel}**: **{corr_c:.3f}**")
        st.scatter_chart(scat_c, x="brightness_change", y="ret_fwd_1m")
        st.caption(
            "Each point is a month aggregated for HQ firms in this county. "
            "X-axis: change in night-lights; Y-axis: next-month return."
        )

# --- Raw table ---

st.markdown("### Underlying observations in this county")

show_cols = [c for c in [
    "ticker", "firm", county_col, state_col,
    "date", "avg_rad_month", "brightness_change",
    "ret", "ret_fwd_1m"
] if c and c in df_c.columns]

st.dataframe(
    df_c.sort_values(["date", "ticker"])[show_cols],
    use_container_width=True,
)
