# pages/4_Globe.py

import streamlit as st
import pandas as pd
from pathlib import Path

from src.load_data import load_lights_monthly_by_coord, load_model_data

# ---------- Page config & styling ----------
st.set_page_config(
    page_title="Night Lights Anomalia Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #050710;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .anomaly-card {
        background: #0b0e1a;
        border-radius: 18px;
        padding: 1rem 1.25rem;
        border: 1px solid #15192a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #c8c8d8;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
    }
    .panel-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='margin-bottom: 1rem;'>Night Lights Anomalia Dashboard</h1>",
    unsafe_allow_html=True,
)

# ---------- 1. Load data for metrics ----------

lights = load_lights_monthly_by_coord(fallback_if_missing=True)
model = load_model_data(fallback_if_missing=True)

if not lights.empty and "date" in lights.columns:
    lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
    lights = lights.dropna(subset=["date"])

if not model.empty and "date" in model.columns:
    model["date"] = pd.to_datetime(model["date"], errors="coerce")
    model = model.dropna(subset=["date"])

# ---------- 2. Sidebar month selector (drives metrics) ----------

if not lights.empty:
    unique_dates = sorted(lights["date"].unique())
elif not model.empty:
    unique_dates = sorted(model["date"].unique())
else:
    unique_dates = []

if unique_dates:
    default_idx = len(unique_dates) - 1
    selected_date = st.sidebar.selectbox(
        "Select month:",
        options=unique_dates,
        index=default_idx,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
    )
else:
    selected_date = None

# ---------- 3. Layout: big globe + right metrics panel ----------

left, right = st.columns([3.2, 1])

with left:
    img_path = Path("assets/night_globe.png")
    if img_path.exists():
        st.image(str(img_path), use_column_width=True)
    else:
        st.info(
            "Add a night-lights globe image at `assets/night_globe.png` "
            "to match the design."
        )

with right:
    # Brightness panel (static description, like the small map card)
    st.markdown("<div class='anomaly-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Brightness Intensity</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem; color:#b0b0c5;'>"
        "Nighttime radiance patterns aggregated from VIIRS data over your sample period. "
        "Use the month selector on the left to drive the metrics below."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics card
    st.markdown("<div class='anomaly-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Metrics</div>", unsafe_allow_html=True)

    delta_light_display = "—"
    pred_ret_display = "—"

    if selected_date is not None and not model.empty:
        if {"brightness_change", "ret_fwd_1m", "date"}.issubset(model.columns):
            m_month = model[model["date"] == pd.to_datetime(selected_date)].copy()
            if not m_month.empty:
                delta_light = m_month["brightness_change"].mean()
                pred_ret = m_month["ret_fwd_1m"].mean()
                delta_light_display = f"{delta_light:.2f}"
                pred_ret_display = f"{pred_ret:.2f}"

    st.markdown(
        f"<div class='metric-label'>ΔLight</div>"
        f"<div class='metric-value'>{delta_light_display}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='metric-label'>Predicted Return</div>"
        f"<div class='metric-value'>{pred_ret_display}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

if selected_date is not None:
    st.caption(
        f"Metrics computed from your merged nightlights–returns panel for "
        f"{pd.Timestamp(selected_date).strftime('%Y-%m')}."
    )



