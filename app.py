# app.py

import streamlit as st
from src.load_data import load_model_data

st.set_page_config(
    page_title="Nightlights & Stock Returns",
    page_icon="🌃",
    layout="wide",
)

st.title("🌃 Nightlights & Stock Returns Dashboard")
st.markdown(
    """
Welcome to the FIN 377 Nightlights project app.

This dashboard connects **VIIRS nighttime lights** with **S&P 500 stock returns**.

Use the pages in the sidebar to:
- 📊 See an overview of the dataset and basic patterns  
- 🔍 Explore individual tickers  
- 🗺 Look at patterns by state (if you add that page)  
- 🤖 Run model-style regressions of brightness changes vs future returns (if you add that page)

The main dataset driving this app is:

`data/final/nightlights_model_data.csv`

It was built using:
- **State-level VIIRS nightlights** (trimmed to recent years)
- **S&P 500 firm locations** (lat/lon mapped to US states)
- **Monthly returns from Yahoo Finance** (2018+)
    """
)

# Quick sanity check: show a tiny summary so you know the app is wired
try:
    df = load_model_data(fallback_if_missing=False)
    if df.empty:
        st.warning(
            "`nightlights_model_data.csv` is empty. "
            "Double-check that `python scripts/build_all.py` ran successfully."
        )
    else:
        st.subheader("Quick Data Check")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Unique tickers", df["ticker"].nunique() if "ticker" in df.columns else "—")
        with col3:
            if "date" in df.columns:
                st.metric(
                    "Date range",
                    f"{df['date'].min().date()} → {df['date'].max().date()}"
                )
            else:
                st.metric("Date range", "—")

        st.caption("Preview of the final dataset:")
        st.dataframe(df.head())
except FileNotFoundError as e:
    st.error(
        f"Could not find the final model data.\n\n"
        f"{e}\n\n"
        "Run `python scripts/build_all.py` in your repo and redeploy."
    )

