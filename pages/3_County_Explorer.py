# pages/3_County_Explorer.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.load_data import load_model_data


st.set_page_config(page_title="County Explorer", layout="wide")


# --------------------------------------------------------------------
# Helper: compute per-ticker R² from ret_fwd ~ brightness_change
# (same logic as in Ticker Explorer, re-used here)
# --------------------------------------------------------------------
def compute_ticker_r2_leaderboard(df: pd.DataFrame, min_obs: int = 12) -> pd.DataFrame:
    """
    Compute per-ticker R² from a simple regression:

        ret_fwd ~ brightness_change

    Using R² = corr(ret_fwd, brightness_change)^2 for robustness.
    """
    # Determine return column
    ret_col = None
    if "ret_fwd" in df.columns:
        ret_col = "ret_fwd"
    elif "ret_fwd_1m" in df.columns:
        ret_col = "ret_fwd_1m"
    elif "ret" in df.columns:
        ret_col = "ret"
    else:
        return pd.DataFrame(
            {
                "error": [
                    "No return column found. Expected one of: 'ret_fwd', 'ret_fwd_1m', 'ret'."
                ]
            }
        )

    required = {"ticker", "brightness_change", ret_col}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(
            {"error": [f"Missing columns for R² leaderboard: {missing}"]}
        )

    rows = []
    for tkr, g in df.groupby("ticker"):
        g = g.dropna(subset=["brightness_change", ret_col])
        if len(g) < min_obs:
            continue
        if g["brightness_change"].nunique() <= 1 or g[ret_col].nunique() <= 1:
            continue

        r = g["brightness_change"].corr(g[ret_col])
        if pd.isna(r):
            continue

        r2 = float(r**2)

        row = {
            "ticker": tkr,
            "R² (ret_vs_brightness)": r2,
            "n_obs": int(len(g)),
        }
        if "firm" in g.columns:
            row["firm"] = g["firm"].iloc[0]
        if "county_name" in g.columns:
            row["HQ county"] = g["county_name"].iloc[0]
        if "state_full" in g.columns:
            row["HQ state"] = g["state_full"].iloc[0]
        elif "state" in g.columns:
            row["HQ state"] = g["state"].iloc[0]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("R² (ret_vs_brightness)", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------
# Load and validate data
# --------------------------------------------------------------------
panel = load_model_data(fallback_if_missing=True)

if panel.empty:
    st.error(
        "nightlights_model_data.csv is empty or missing.\n\n"
        "Rebuild it with `python scripts/build_all.py`, commit, and redeploy."
    )
    st.stop()

required_panel_cols = {
    "ticker",
    "firm",
    "county_name",
    "state",          # 2-letter or full; we'll display what we have
    "date",
    "brightness_change",
}
missing_panel = required_panel_cols - set(panel.columns)
if missing_panel:
    st.error(
        f"nightlights_model_data.csv must contain: {required_panel_cols}. "
        f"Missing: {missing_panel}"
    )
    st.stop()

# Date handling
panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
panel = panel.dropna(subset=["date"])

# Decide which return column to use
ret_col = "ret_fwd" if "ret_fwd" in panel.columns else (
    "ret_fwd_1m" if "ret_fwd_1m" in panel.columns else "ret"
)
if ret_col not in panel.columns:
    st.error("No return column found. Expected 'ret_fwd', 'ret_fwd_1m', or 'ret'.")
    st.stop()

# Optional nicer state name
state_display_col = "state_full" if "state_full" in panel.columns else "state"

# --------------------------------------------------------------------
# Title + explanation
# --------------------------------------------------------------------
st.title("County Explorer (HQ-mapped)")

st.markdown(
    """
This page zooms in on the **geography** of our strategy.

We first map each **S&P 500 firm** to a specific **headquarters county**. Then, for that
county, we track:

- **Night-time brightness** (VIIRS) and its **changes over time**, and  
- The **forward stock returns** for firms headquartered there.

This lets us ask two related questions:

1. *Which counties show the biggest swings in night-time economic activity?*  
2. *Which stocks are most tightly linked to those local brightness changes?*
"""
)

# --------------------------------------------------------------------
# Sidebar: county selection
# --------------------------------------------------------------------
st.sidebar.header("County selection")

# Unique counties (county_name + state)
panel["county_key"] = panel["county_name"].astype(str) + " (" + panel[state_display_col].astype(str) + ")"
counties_sorted = sorted(panel["county_key"].dropna().unique().tolist())
default_county = counties_sorted[0] if counties_sorted else None

county_choice = st.sidebar.selectbox(
    "Choose HQ county:",
    options=counties_sorted,
    index=0,
)

# Filter panel to that county
county_name, county_state = county_choice.rsplit(" (", 1)
county_state = county_state.rstrip(")")

county_df = panel[
    (panel["county_name"].astype(str) == county_name)
    & (panel[state_display_col].astype(str) == county_state)
].copy()

county_df = county_df.sort_values("date")

st.markdown(
    f"### County: **{county_name}** ({county_state}) – HQ-mapped firms and brightness"
)

hq_firms = (
    county_df[["ticker", "firm"]]
    .drop_duplicates()
    .sort_values("ticker")
)

st.markdown("#### Firms headquartered in this county")
st.dataframe(hq_firms, use_container_width=True)

st.markdown(
    """
Each firm above is **headquartered in this county**, so its night-lights signal is
coming from **this local economy**. If warehouses, offices, or surrounding commercial
areas get busier (or quieter), VIIRS brightness will typically move with it.
"""
)

# --------------------------------------------------------------------
# County-level time-series: brightness and average forward returns
# --------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Brightness change over time (county-level)")

    # Aggregate brightness by county per date
    bright_ts = (
        county_df.groupby("date", as_index=False)["brightness_change"]
        .mean()
        .rename(columns={"brightness_change": "avg_brightness_change"})
    )

    if not bright_ts.empty:
        fig_b = px.line(
            bright_ts,
            x="date",
            y="avg_brightness_change",
            labels={
                "date": "Date",
                "avg_brightness_change": "Average Δ brightness (HQ county)",
            },
        )
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("No brightness_change data for this county.")

with col2:
    st.markdown("#### Average forward returns over time (county firms)")

    ret_ts = (
        county_df.groupby("date", as_index=False)[ret_col]
        .mean()
        .rename(columns={ret_col: "avg_forward_return"})
    )

    if not ret_ts.empty:
        fig_r = px.line(
            ret_ts,
            x="date",
            y="avg_forward_return",
            labels={
                "date": "Date",
                "avg_forward_return": "Average forward return (firms in county)",
            },
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info(f"No {ret_col} data for this county.")

st.markdown(
    """
These two charts summarize the **county-level story**:

- On the **left**, we see how much **night-time brightness in the county** moves over time.  
- On the **right**, we see the **average forward return** of all HQ firms located there.

If there is a link between **local economic activity** and **stock performance**, we
would expect big moves in brightness to be followed by meaningful changes in returns.
"""
)

# --------------------------------------------------------------------
# County-level scatter: brightness vs forward returns (all firms in county)
# --------------------------------------------------------------------
st.markdown("#### Scatter: Δ brightness vs forward returns (firms in this county)")

df_scatter_c = county_df.dropna(subset=["brightness_change", ret_col])

if len(df_scatter_c) >= 5:
    fig_scatter_c = px.scatter(
        df_scatter_c,
        x="brightness_change",
        y=ret_col,
        color="ticker",
        labels={
            "brightness_change": "Δ brightness (HQ county)",
            ret_col: "Forward return",
            "ticker": "Ticker",
        },
    )
    st.plotly_chart(fig_scatter_c, use_container_width=True)

    r_c = df_scatter_c["brightness_change"].corr(df_scatter_c[ret_col])
    r2_c = r_c**2 if pd.notna(r_c) else np.nan

    st.markdown(
        f"""
If we pool all firms headquartered in **{county_name} ({county_state})** and look at

\\[
\\text{{{ret_col}}} = \\alpha + \\beta \\cdot \\text{{brightness\_change}} + \\varepsilon,
\\]

the squared correlation (approximate R²) is about **{r2_c:.3f}**.

This is a **county-level measure** of how much local brightness helps explain the
subsequent stock performance of firms based there.
"""
    )
else:
    st.info(
        "Not enough non-missing observations in this county to draw a scatter or estimate a meaningful R²."
    )

# --------------------------------------------------------------------
# Global ticker-level R² leaderboard (county / HQ view)
# --------------------------------------------------------------------
st.markdown("---")
st.markdown("## Ticker R² leaderboard – HQ county night-lights vs returns")

leader = compute_ticker_r2_leaderboard(panel)

if leader.empty or "error" in leader.columns:
    if not leader.empty and "error" in leader.columns:
        st.error(leader["error"].iloc[0])
    else:
        st.info(
            "Not enough data to compute the ticker R² leaderboard. "
            "We need valid brightness_change and forward returns."
        )
else:
    st.markdown(
        """
Here we move from **one county at a time** to the **entire S&P 500 universe**.

For each stock, we run a simple regression:

\\[
\\text{forward return} = \\alpha + \\beta \\cdot \\text{brightness\_change} + \\varepsilon
\\]

using that stock’s HQ county brightness, and compute the **R²** from the squared
correlation.

- A **high R²** means: the stock’s forward returns are **strongly related**
  to swings in local brightness around HQ.  
- A **low R²** means: night-lights carry little information for that ticker.

This gives us a **ranking of names** where night-time lights are most promising as a
trading signal.
"""
    )

    st.dataframe(
        leader.head(10)[
            [
                "ticker",
                "firm",
                "HQ county",
                "HQ state",
                "R² (ret_vs_brightness)",
                "n_obs",
            ]
        ],
        use_container_width=True,
    )

    st.caption(
        "In our dataset, the top names (e.g., IDXX, OTIS, MCK, SRE, COIN) show R² "
        "in the ~0.13–0.18 range. That is **unusually high** for return data, and "
        "it suggests that night-time brightness around HQ carries real forward-looking "
        "information for those firms."
    )
