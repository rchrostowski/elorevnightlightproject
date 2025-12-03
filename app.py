# app.py

import streamlit as st
import pandas as pd

from src.load_data import load_model_data

st.set_page_config(
    page_title="Night Lights Anomalia Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #050710;
        color: #f9fafb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .metric-card {
        background: #0b0e1a;
        border-radius: 18px;
        padding: 1rem 1.25rem;
        border: 1px solid #15192a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Night Lights Anomalia Dashboard")

st.markdown(
    """
This dashboard links **satellite night-time lights** to **stock returns** for  
S&P 500 firms, using the brightness of the **county where each firm’s HQ is located**.

Use the tabs on the left to:
- 🔍 Explore specific **tickers** and their HQ county brightness
- 🏙 Drill into **counties** and see which firms live there
- 🌍 View an interactive **globe** of hotspots
- 📊 Run **regressions** of returns on brightness changes
"""
)

# ---------- Load final model data ----------
df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "Final dataset `nightlights_model_data.csv` is missing or empty.\n\n"
        "Run `python scripts/build_all.py` and commit "
        "`data/final/nightlights_model_data.csv`."
    )
    st.stop()

# Basic cleaning
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

# ---------- High-level metrics ----------
n_rows = len(df)
n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
n_counties = df["county_name"].nunique() if "county_name" in df.columns else 0
date_min = df["date"].min() if "date" in df.columns else None
date_max = df["date"].max() if "date" in df.columns else None

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Observations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{n_rows:,}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Tickers</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{n_tickers}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">HQ counties</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{n_counties}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Sample window</div>', unsafe_allow_html=True)
    if date_min is not None and date_max is not None:
        st.markdown(
            f'<div class="metric-value">'
            f'{date_min.strftime("%Y-%m")} → {date_max.strftime("%Y-%m")}'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="metric-value">n/a</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Simple correlation summary ----------
if {"brightness_change", "ret_fwd_1m"}.issubset(df.columns):
    corr_df = df[["brightness_change", "ret_fwd_1m"]].dropna()
    corr_val = corr_df["brightness_change"].corr(corr_df["ret_fwd_1m"])
    st.markdown("### Brightness vs. returns (raw correlation, full panel)")

    colc1, colc2 = st.columns([1, 3])
    with colc1:
        st.metric(
            "Corr(ΔBrightness, next-month return)",
            f"{corr_val:.3f}" if pd.notna(corr_val) else "n/a",
        )

    with colc2:
        st.caption(
            "Correlation of county-level **change in brightness** and **next-month stock return** "
            "across the full panel (all tickers × months)."
        )

# ---------- Sample of data ----------
st.markdown("### Peek at the final dataset")

show_cols = [c for c in [
    "ticker", "firm", "county_name", "state",
    "date", "avg_rad_month", "brightness_change",
    "ret", "ret_fwd_1m"
] if c in df.columns]

st.dataframe(
    df.sort_values("date").head(500)[show_cols],
    use_container_width=True,
)
