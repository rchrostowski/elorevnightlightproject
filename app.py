# app.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.set_page_config(
    page_title="Night Lights Anomalia – HQ County Alpha",
    layout="wide",
)

# ---------- Shared helpers ----------

def _get_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


# ---------- Load data ----------

df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "nightlights_model_data.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` locally to rebuild it and push to GitHub."
    )
    st.stop()

# Standardize date
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

brightness_col = _get_col(df, ["brightness_change", "d_light", "delta_light"])
level_col = _get_col(df, ["brightness_hq", "avg_rad_hq", "avg_rad_month", "light_level"])
ret_col = _get_col(df, ["ret_excess", "ret", "return"])
ret_fwd_col = _get_col(df, ["ret_fwd_1m", "ret_fwd", "ret_forward_1m"])

# ---------- Layout ----------

st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Night Lights Anomalia</h1>
    <p style="color:#bbbbcc; font-size:0.95rem; margin-bottom:1.5rem;">
    Testing whether <b>changes in night-time brightness in a firm’s HQ county</b> predict
    <b>future stock returns</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2.5, 1.5])

with left:
    st.markdown("### Project story")

    st.markdown(
        """
        **Research question**  
        Does an increase in night-time light intensity around a firm's headquarters
        signal higher economic activity – and therefore **predict higher future returns**?

        **Panel setup**

        - Unit: firm × month (HQ county level)  
        - Period: roughly 2018+ (post clean-up)  
        - For each firm and month we have:
          - HQ county brightness level and change
          - Stock return this month
          - **Next-month** return (target)
        - We run regressions of the form:
          - \\( r_{i,t+1} = \\alpha + \\beta \\Delta Light_{i,t} + \\gamma_{month} + \\varepsilon_{i,t+1} \\)
            with calendar **month fixed effects** \\( \\gamma_{month} \\)
        """
    )

    st.markdown("### How to use this app")

    st.markdown(
        """
        - **Overview** – global summary, time-series, and distributions  
        - **Ticker Explorer** – zoom in on a single firm’s HQ light vs returns  
        - **County Explorer** – zoom in on a HQ county across multiple firms  
        - **Globe** – HQ hotspots by brightness or predicted alpha  
        - **Regression Lab** – actual panel regression with month fixed effects
        """
    )

with right:
    st.markdown("### Snapshot of the dataset")

    n_obs = len(df)
    n_tickers = df["ticker"].nunique() if "ticker" in df.columns else None
    n_counties = df["county_name"].nunique() if "county_name" in df.columns else None

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Observations", f"{n_obs:,}")
        if n_tickers is not None:
            st.metric("Tickers", f"{n_tickers:,}")
    with c2:
        if n_counties is not None:
            st.metric("HQ counties", f"{n_counties:,}")
        date_min = df["date"].min()
        date_max = df["date"].max()
        st.metric("Date range", f"{date_min:%Y-%m} → {date_max:%Y-%m}")

    if brightness_col and ret_fwd_col:
        tmp = df[[brightness_col, ret_fwd_col]].dropna()
        if not tmp.empty:
            corr = tmp[brightness_col].corr(tmp[ret_fwd_col])
            st.metric("Corr(ΔLight, next-month return)", f"{corr:0.3f}")
        else:
            st.info("Not enough data to compute correlation yet.")

    st.caption(
        "All charts in the other pages are built directly off this dataset."
    )

