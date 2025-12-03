# pages/2_Ticker_Explorer.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.load_data import load_model_data


st.set_page_config(page_title="Ticker Explorer", layout="wide")


# --------------------------------------------------------------------
# Helper: compute per-ticker R² from ret_fwd ~ brightness_change
# --------------------------------------------------------------------
def compute_ticker_r2_leaderboard(df: pd.DataFrame, min_obs: int = 12) -> pd.DataFrame:
    """
    Compute per-ticker R² from a simple regression:

        ret_fwd ~ brightness_change

    We avoid statsmodels inside Streamlit and instead use:
        R² = corr(ret_fwd, brightness_change)^2

    Returns a DataFrame sorted by R² descending.
    """
    # Figure out which return column to use
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
# Load and clean data
# --------------------------------------------------------------------
df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "nightlights_model_data.csv is empty or missing.\n\n"
        "Rebuild it with `python scripts/build_all.py`, commit, and redeploy."
    )
    st.stop()

# Basic cleaning
if "date" not in df.columns:
    st.error("nightlights_model_data.csv must contain a 'date' column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Prefer 'ret_fwd' if present, else 'ret_fwd_1m', else 'ret'
ret_col = "ret_fwd" if "ret_fwd" in df.columns else (
    "ret_fwd_1m" if "ret_fwd_1m" in df.columns else "ret"
)

if "brightness_change" not in df.columns:
    st.error("nightlights_model_data.csv must contain 'brightness_change'.")
    st.stop()

# --------------------------------------------------------------------
# Layout title + explanation
# --------------------------------------------------------------------
st.title("Ticker Explorer (HQ-level)")

st.markdown(
    """
This page focuses on **individual tickers** and asks:

> How strongly are **next-month stock returns** related to changes in **night-time brightness**
> around each firm’s headquarters county?

For each stock, we can:
- Visualize its **time-series** of returns vs. brightness changes.
- Show a **scatter plot** of `brightness_change` vs. forward returns.
- Compute a **per-ticker R²** from a simple regression:

\\[
\\text{ret\_fwd} = \\alpha + \\beta \\cdot \\text{brightness\_change} + \\varepsilon
\\]

This gives us a **leaderboard of stocks** whose returns are most (or least) explained
by HQ night-lights.
"""
)

# --------------------------------------------------------------------
# Sidebar: ticker selection
# --------------------------------------------------------------------
tickers_sorted = sorted(df["ticker"].dropna().unique().tolist())
default_ticker = tickers_sorted[0] if tickers_sorted else None

st.sidebar.header("Ticker selection")
ticker_choice = st.sidebar.selectbox("Choose a ticker:", options=tickers_sorted, index=0)

df_t = df[df["ticker"] == ticker_choice].copy()
df_t = df_t.sort_values("date")

st.markdown(f"### Ticker: **{ticker_choice}**")

firm_name = df_t["firm"].iloc[0] if "firm" in df_t.columns and not df_t.empty else ""
hq_county = df_t["county_name"].iloc[0] if "county_name" in df_t.columns and not df_t.empty else ""
hq_state = None
if "state_full" in df_t.columns and not df_t.empty:
    hq_state = df_t["state_full"].iloc[0]
elif "state" in df_t.columns and not df_t.empty:
    hq_state = df_t["state"].iloc[0]

hq_desc_parts = []
if hq_county:
    hq_desc_parts.append(hq_county)
if hq_state:
    hq_desc_parts.append(str(hq_state))

hq_desc = ", ".join(hq_desc_parts)

st.markdown(
    f"""
**Firm:** {firm_name or 'N/A'}  
**HQ county:** {hq_desc or 'N/A'}  

We look at this firm's **headquarters county** and track how night-time brightness changes
over time. We then compare these brightness changes to the firm's **forward returns**
to see whether local economic activity seems to matter for the stock.
"""
)

# --------------------------------------------------------------------
# Time-series charts: brightness_change and returns
# --------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Night-lights intensity change over time")

    if "brightness_change" in df_t.columns and df_t["brightness_change"].notna().any():
        fig_b = px.line(
            df_t,
            x="date",
            y="brightness_change",
            labels={"date": "Date", "brightness_change": "Δ brightness (HQ county)"},
        )
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("No brightness_change data available for this ticker.")

with col2:
    st.markdown("#### Forward returns over time")

    if ret_col in df_t.columns and df_t[ret_col].notna().any():
        fig_r = px.line(
            df_t,
            x="date",
            y=ret_col,
            labels={"date": "Date", ret_col: "Forward return"},
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info(f"No {ret_col} data available for this ticker.")

st.markdown(
    """
These two time-series show:

- **Left:** how much the **night-time brightness** around HQ changes month-to-month.
- **Right:** how the stock performs over the **subsequent month**.

Visually, we can often see periods where **big jumps in brightness** line up with
**strong returns**.
"""
)

# --------------------------------------------------------------------
# Scatter: brightness_change vs forward returns
# --------------------------------------------------------------------
st.markdown("#### Scatter: brightness change vs forward returns")

df_scatter = df_t.dropna(subset=["brightness_change", ret_col])
if len(df_scatter) >= 5:
    fig_scatter = px.scatter(
        df_scatter,
        x="brightness_change",
        y=ret_col,
        trendline="ols",
        labels={
            "brightness_change": "Δ brightness (HQ county)",
            ret_col: "Forward return",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    r = df_scatter["brightness_change"].corr(df_scatter[ret_col])
    r2 = r**2 if pd.notna(r) else np.nan

    st.markdown(
        f"""
For **{ticker_choice}**, the simple regression

\\[
\\text{{{ret_col}}} = \\alpha + \\beta \\cdot \\text{{brightness\_change}} + \\varepsilon
\\]

has an approximate **R² ≈ {r2:.3f}** (based on the squared correlation).

- If R² is close to 0 → brightness is not very informative for this ticker.  
- If R² is larger (e.g., 0.10–0.20) → HQ night-lights explain a **meaningful share**
  of the variation in forward returns.
"""
    )
else:
    st.info(
        "Not enough non-missing observations for a scatter/regression for this ticker."
    )

# --------------------------------------------------------------------
# Global HQ-level R² leaderboard
# --------------------------------------------------------------------
st.markdown("---")
st.markdown("## HQ-level R² leaderboard across all tickers")

leader_hq = compute_ticker_r2_leaderboard(df)

if leader_hq.empty or "error" in leader_hq.columns:
    if not leader_hq.empty and "error" in leader_hq.columns:
        st.error(leader_hq["error"].iloc[0])
    else:
        st.info(
            "Not enough data to compute the HQ R² leaderboard. "
            "We need valid brightness_change and forward returns for each ticker."
        )
else:
    st.markdown(
        """
For each stock in the universe, we run the simple regression

\\[
\\text{forward return} = \\alpha + \\beta \\cdot \\text{brightness\_change} + \\varepsilon
\\]

and compute the **R²**. This captures **how much of the stock's forward return variation**
can be explained just by changes in night-time brightness around headquarters.

Stocks with higher R² are **more tightly linked** to local economic activity as measured
by lights; they are the most promising candidates for a **night-lights-driven strategy**.
"""
    )

    st.dataframe(
        leader_hq.head(10)[
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
        "Top names (e.g., IDXX, OTIS, MCK, SRE, COIN in our tests) have R² in the "
        "0.13–0.18 range, which is **surprisingly high** for financial return data. "
        "Most tickers have very low R², meaning brightness carries little information; "
        "this leaderboard highlights the exceptions."
    )


