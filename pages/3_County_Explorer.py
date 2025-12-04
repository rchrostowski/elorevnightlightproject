# pages/3_County_Explorer.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.load_data import load_model_data

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="County Explorer – Nightlights × Returns",
    layout="wide",
)

st.title("County Explorer: Nightlights vs. Stock Returns")

st.markdown(
    """
This page drills down to the **county level** and connects three ideas:

1. **Where are S&P 500 headquarters actually located?**  
   Each firm is mapped to a specific **county** (e.g., Apple → Santa Clara County, CA).

2. **How do local night-time lights change over time in that county?**  
   We track **brightness changes** (ΔLight) over time as a proxy for local economic activity.

3. **Do these brightness changes relate to that stock’s future returns?**  
   For each ticker–county pair, we look at how well brightness changes explain **next-month returns** using
   a simple **R² statistic**. Higher R² means brightness changes are more informative for that stock in that
   location.

Use the controls on the left to filter by **state**, **ticker**, and **minimum observations**, then explore the
county-level time series and the **leaderboard of counties with the highest R²**.
"""
)

# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------
df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "nightlights_model_data.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` to rebuild it, commit the CSV in "
        "`data/final/`, and redeploy."
    )
    st.stop()

required_cols = {
    "ticker",
    "firm",
    "county_name",
    "date",
    "brightness_change",
}

missing = required_cols - set(df.columns)
if missing:
    st.error(
        f"nightlights_model_data.csv is missing required columns: {missing}\n\n"
        f"Found columns: {df.columns.tolist()}"
    )
    st.stop()

# Handle state column flexibly
if "state" in df.columns:
    df["state_display"] = df["state"].astype(str)
elif "state_full" in df.columns:
    df["state_display"] = df["state_full"].astype(str)
elif "state_key" in df.columns:
    df["state_display"] = df["state_key"].astype(str)
else:
    df["state_display"] = "(Unknown)"

# Handle forward return column (ret_fwd_1m vs ret_fwd)
if "ret_fwd_1m" in df.columns:
    pass
elif "ret_fwd" in df.columns:
    df["ret_fwd_1m"] = df["ret_fwd"]
else:
    st.error(
        "nightlights_model_data.csv must contain either `ret_fwd_1m` or `ret_fwd` "
        "so we can measure next-month returns."
    )
    st.stop()

# Ensure proper dtypes
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")

df = df.dropna(subset=["brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error("After cleaning, there are no rows with valid brightness_change and ret_fwd_1m.")
    st.stop()

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

# State filter
state_options = ["All states"] + sorted(df["state_display"].unique().tolist())
state_choice = st.sidebar.selectbox("Filter by state:", options=state_options)

# Ticker filter
all_tickers = sorted(df["ticker"].unique().tolist())
ticker_choice = st.sidebar.multiselect(
    "Filter by ticker (optional):",
    options=all_tickers,
    default=[],
)

# Minimum observations per county–ticker for leaderboard
min_obs = st.sidebar.slider(
    "Minimum months per county–ticker for R² leaderboard:",
    min_value=4,
    max_value=48,
    value=12,
    step=1,
)

# Apply filters
df_filt = df.copy()
if state_choice != "All states":
    df_filt = df_filt[df_filt["state_display"] == state_choice]

if ticker_choice:
    df_filt = df_filt[df_filt["ticker"].isin(ticker_choice)]

if df_filt.empty:
    st.warning("No data after applying filters. Try loosening the filters.")
    st.stop()

# ---------------------------------------------------------
# High-level summary
# ---------------------------------------------------------
min_date = df_filt["date"].min()
max_date = df_filt["date"].max()

n_firms = df_filt["firm"].nunique()
n_tickers = df_filt["ticker"].nunique()
n_counties = df_filt[["county_name", "state_display"]].drop_duplicates().shape[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Firms in sample", f"{n_firms}")
with col2:
    st.metric("Tickers", f"{n_tickers}")
with col3:
    st.metric("HQ counties", f"{n_counties}")
with col4:
    st.metric(
        "Time span",
        f"{min_date.strftime('%Y-%m')} → {max_date.strftime('%Y-%m')}",
    )

st.markdown(
    """
**Interpretation:**  
We’re working with a panel of S&P 500 firms where each firm is anchored to a **single HQ county**.  
For every month in the sample, we observe:

- A **change in night-time brightness** in that county (ΔLight), and  
- The firm’s **next-month stock return**.

The rest of this page asks: *“In which counties does ΔLight do the best job at explaining next-month returns?”*
"""
)

# ---------------------------------------------------------
# County summary table
# ---------------------------------------------------------
group_cols = ["state_display", "county_name"]
county_summary = (
    df_filt.groupby(group_cols)
    .agg(
        n_obs=("date", "size"),
        n_firms=("firm", "nunique"),
        n_tickers=("ticker", "nunique"),
        avg_brightness_change=("brightness_change", "mean"),
        avg_next_month_ret=("ret_fwd_1m", "mean"),
    )
    .reset_index()
)

county_summary["county_label"] = (
    county_summary["county_name"] + " (" + county_summary["state_display"] + ")"
)

st.subheader("County-level summary")

st.markdown(
    """
This table aggregates the panel to the **county level**:

- **n_obs** – number of month-firm observations in the county  
- **n_firms / n_tickers** – how many distinct firms/tickers are headquartered there  
- **avg_brightness_change** – average ΔLight across all firm-months in that county  
- **avg_next_month_ret** – average forward return for firms in that county  

You can click on a specific county below to see a detailed **time-series view**.
"""
)

st.dataframe(
    county_summary[
        [
            "county_label",
            "n_obs",
            "n_firms",
            "n_tickers",
            "avg_brightness_change",
            "avg_next_month_ret",
        ]
    ].sort_values("n_obs", ascending=False),
    use_container_width=True,
    height=350,
)

# ---------------------------------------------------------
# Drill-down: county-level time series
# ---------------------------------------------------------
st.subheader("County drill-down: ΔLight and next-month returns over time")

if not county_summary.empty:
    default_county = county_summary.sort_values("n_obs", ascending=False)["county_label"].iloc[0]
    selected_county_label = st.selectbox(
        "Choose a county to visualize:",
        options=county_summary["county_label"].tolist(),
        index=county_summary["county_label"].tolist().index(default_county),
    )

    sel_row = county_summary[county_summary["county_label"] == selected_county_label].iloc[0]
    sel_state = sel_row["state_display"]
    sel_county = sel_row["county_name"]

    ts = (
        df_filt[
            (df_filt["state_display"] == sel_state)
            & (df_filt["county_name"] == sel_county)
        ]
        .sort_values("date")
        .copy()
    )

    if ts.empty:
        st.warning("No time-series data for this county after filters.")
    else:
        # Two side-by-side charts: returns and brightness
        c1, c2 = st.columns(2)

        with c1:
            fig_ret = px.line(
                ts,
                x="date",
                y="ret_fwd_1m",
                title=f"Next-month returns – {selected_county_label}",
                labels={"date": "Month", "ret_fwd_1m": "Next-month return"},
            )
            fig_ret.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with c2:
            fig_light = px.line(
                ts,
                x="date",
                y="brightness_change",
                title=f"Brightness change (ΔLight) – {selected_county_label}",
                labels={"date": "Month", "brightness_change": "Brightness change"},
            )
            fig_light.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_light, use_container_width=True)

        st.markdown(
            """
**How to read these charts:**

- The **left chart** shows the **stock’s next-month return** for each firm-month headquartered in this county.  
- The **right chart** shows the **change in night-time brightness (ΔLight)** in the same county over time.  

When you see periods where **ΔLight spikes or drops** and **returns move in the same direction in the following month**,  
that’s exactly the pattern our regression is trying to quantify.
"""
        )

# ---------------------------------------------------------
# R² leaderboard (county × ticker)
# ---------------------------------------------------------
st.subheader("Leaderboard: Counties where brightness explains returns best")

st.markdown(
    """
Here we ask a more *statistical* question:

> For each **ticker–county** combination, how well do brightness changes explain next-month returns?

For every ticker–county pair, we compute a simple **R²** from a regression of:
\\[
\\text{ret\_fwd\_1m} = \\alpha + \\beta \\cdot \\text{brightness\_change} + \\varepsilon
\\]

- A **higher R²** means brightness changes do a better job of explaining variation in returns.  
- The **sign** (shown as *Signed R²*) indicates whether the relationship is **positive** or **negative**  
  (positive = higher brightness → higher returns, on average).
"""
)

def simple_r2(group: pd.DataFrame) -> float:
    """Signed R² from simple correlation between brightness_change and ret_fwd_1m."""
    g = group.dropna(subset=["brightness_change", "ret_fwd_1m"]).copy()
    if len(g) < min_obs:
        return np.nan
    x = g["brightness_change"]
    y = g["ret_fwd_1m"]
    if x.var() == 0 or y.var() == 0:
        return np.nan
    r = x.corr(y)
    if pd.isna(r):
        return np.nan
    return float(np.sign(r) * (r ** 2))


rows = []
group_cols_r2 = ["ticker", "firm", "county_name", "state_display"]

for keys, sub in df_filt.groupby(group_cols_r2):
    r2_signed = simple_r2(sub)
    if not np.isnan(r2_signed):
        ticker, firm, county_name, state_disp = keys
        rows.append(
            {
                "ticker": ticker,
                "firm": firm,
                "county_name": county_name,
                "state": state_disp,
                "n_obs": len(sub),
                "r2_signed": r2_signed,
                "r2_abs": abs(r2_signed),
            }
        )

if not rows:
    st.warning(
        "Could not compute any R² values with the current filters and minimum observation threshold."
    )
else:
    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values("r2_abs", ascending=False)

    st.markdown("#### Top county–ticker combinations by |R²|")

    st.dataframe(
        leaderboard[
            ["ticker", "firm", "county_name", "state", "n_obs", "r2_signed", "r2_abs"]
        ]
        .head(15)
        .rename(
            columns={
                "county_name": "County",
                "state": "State",
                "n_obs": "# Months",
                "r2_signed": "Signed R²",
                "r2_abs": "|R²|",
            }
        ),
        use_container_width=True,
        height=400,
    )

    st.markdown(
        """
**Interpretation:**

- **|R²|** close to 0 → ΔLight does **not** explain much of the variation in that stock’s returns.  
- **|R²|** closer to 1 → ΔLight explains **a lot** of the return variation for that ticker in that county.  
- A **positive Signed R²** means brighter-than-usual months tend to be followed by **higher** returns.  
- A **negative Signed R²** means brighter-than-usual months tend to be followed by **lower** returns (or the signal is noisy).

This leaderboard is a concrete way to answer:  
> “For which **stocks and HQ locations** does night-time brightness contain the most information about future returns?”
"""
    )

