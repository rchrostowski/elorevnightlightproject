# pages/4_Globe.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---- Safe import of data loader ----
try:
    from src.load_data import load_model_data
except Exception as e:
    st.set_page_config(page_title="Night Lights Anomalia Dashboard", layout="wide")
    st.error(
        "Could not import data-loading functions.\n\n"
        "Make sure src/ is a Python package (has __init__.py) "
        "and that src/load_data.py defines load_model_data.\n\n"
        f"Original error: {e}"
    )
    st.stop()

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

# ---------- 1. Load final modeling data ----------
model = load_model_data(fallback_if_missing=True)

if model.empty:
    st.error(
        "Final model data is missing or empty.\n\n"
        "Run `python scripts/build_all.py` to rebuild "
        "`data/final/nightlights_model_data.csv` and redeploy."
    )
    st.stop()

# Make sure date is datetime
if "date" not in model.columns:
    st.error("`nightlights_model_data.csv` must have a `date` column.")
    st.stop()

model["date"] = pd.to_datetime(model["date"], errors="coerce")
model = model.dropna(subset=["date"])

# We’ll work at *monthly* frequency (floor to first of month)
model["month"] = model["date"].dt.to_period("M").dt.to_timestamp()

# ---------- 2. State mapping / coords ----------

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District Of Columbia": "DC", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR",
}
STATE_NAME_FROM_ABBR = {v: k for k, v in STATE_ABBR.items()}

STATE_COORDS = {
    "AL": (32.7, -86.7),   "AK": (64.8, -147.7), "AZ": (34.0, -111.7),
    "AR": (34.9, -92.3),   "CA": (37.3, -119.7), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7),   "DE": (39.0, -75.5),  "DC": (38.9, -77.0),
    "FL": (28.4, -82.5),   "GA": (32.7, -83.3),  "HI": (20.8, -157.0),
    "ID": (44.4, -114.4),  "IL": (40.0, -89.2),  "IN": (39.9, -86.3),
    "IA": (42.1, -93.5),   "KS": (38.5, -98.0),  "KY": (37.5, -85.3),
    "LA": (31.0, -92.0),   "ME": (45.3, -69.2),  "MD": (39.0, -76.7),
    "MA": (42.4, -71.8),   "MI": (44.3, -85.4),  "MN": (46.3, -94.2),
    "MS": (32.7, -89.7),   "MO": (38.5, -92.3),  "MT": (46.9, -110.4),
    "NE": (41.5, -99.8),   "NV": (39.5, -116.6), "NH": (43.7, -71.6),
    "NJ": (40.1, -74.7),   "NM": (34.2, -106.0), "NY": (42.9, -75.0),
    "NC": (35.5, -79.4),   "ND": (47.5, -100.5), "OH": (40.3, -82.8),
    "OK": (35.6, -97.5),   "OR": (44.0, -120.5), "PA": (41.0, -77.6),
    "RI": (41.7, -71.6),   "SC": (33.8, -80.9),  "SD": (44.4, -100.2),
    "TN": (35.9, -86.4),   "TX": (31.0, -99.0),  "UT": (39.3, -111.7),
    "VT": (44.0, -72.7),   "VA": (37.5, -78.8),  "WA": (47.4, -120.5),
    "WV": (38.6, -80.6),   "WI": (44.6, -89.5),  "WY": (43.1, -107.6),
    "PR": (18.2, -66.4),
}

cols = set(model.columns)

# Figure out how to get a 2-letter state code
if "state" in cols:
    model["state_code"] = model["state"].astype(str).str.upper()
elif "state_full" in cols:
    model["state_code"] = (
        model["state_full"]
        .astype(str)
        .str.strip()
        .str.title()
        .map(STATE_ABBR)
    )
elif "state_name" in cols:
    model["state_code"] = (
        model["state_name"]
        .astype(str)
        .str.strip()
        .str.title()
        .map(STATE_ABBR)
    )
else:
    st.error(
        "Could not find a state column. Expected one of: "
        "`state`, `state_full`, or `state_name` in nightlights_model_data.csv."
    )
    st.stop()

model = model.dropna(subset=["state_code"])
model = model[model["state_code"].isin(STATE_COORDS.keys())].copy()

if model.empty:
    st.error("No US state rows found after mapping state codes.")
    st.stop()

# ---------- 3. Sidebar controls ----------
st.sidebar.header("Globe controls")

available_months = sorted(model["month"].unique())
default_idx = len(available_months) - 1  # latest month

selected_month = st.sidebar.selectbox(
    "Month",
    options=available_months,
    index=default_idx,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
)

# Filter to selected month
month_df = model[model["month"] == selected_month].copy()
if month_df.empty:
    st.warning(
        f"No observations in model data for "
        f"{pd.Timestamp(selected_month).strftime('%Y-%m')}."
    )
    st.stop()

# ---------- 4. Aggregate to state level ----------
if "avg_rad_month" not in month_df.columns:
    st.error("`nightlights_model_data.csv` must have an `avg_rad_month` column.")
    st.stop()

# Some builds may only have `ret_fwd`; make sure we have `ret_fwd_1m`
if "ret_fwd_1m" not in month_df.columns and "ret_fwd" in month_df.columns:
    month_df["ret_fwd_1m"] = month_df["ret_fwd"]

agg_cols = {
    "avg_rad_month": ("avg_rad_month", "mean"),
}
if "ret_fwd_1m" in month_df.columns:
    agg_cols["ret_fwd_1m"] = ("ret_fwd_1m", "mean")

state_df = (
    month_df.groupby("state_code", as_index=False)
    .agg(**agg_cols)
)

# Attach nice state name + coords
state_df["state_name"] = state_df["state_code"].map(STATE_NAME_FROM_ABBR)
state_df["lat"] = state_df["state_code"].map(lambda s: STATE_COORDS[s][0])
state_df["lon"] = state_df["state_code"].map(lambda s: STATE_COORDS[s][1])

state_df = state_df.dropna(subset=["lat", "lon"])

if state_df.empty:
    st.error("No states had coordinates mapped for this month.")
    st.stop()

# ---------- 5. Marker encoding: use expected return if available ----------
use_return = "ret_fwd_1m" in state_df.columns and state_df["ret_fwd_1m"].notna().any()

if use_return:
    r = state_df["ret_fwd_1m"].fillna(0.0)
    max_abs = r.abs().max()
    if max_abs == 0:
        r_norm = pd.Series(0.5, index=r.index)
    else:
        r_norm = (r / max_abs + 1.0) / 2.0  # map [-max, max] -> [0, 1]
    marker_sizes = 8 + 22 * r_norm         # size 8–30
    marker_intensity = r_norm              # color 0–1
else:
    b = state_df["avg_rad_month"]
    if b.nunique() > 1:
        b_norm = (b - b.min()) / (b.max() - b.min())
    else:
        b_norm = pd.Series(0.5, index=b.index)
    marker_sizes = 8 + 22 * b_norm
    marker_intensity = b_norm

blue_scale = [
    [0.0, "rgb(2, 6, 23)"],
    [0.3, "rgb(13, 37, 88)"],
    [0.6, "rgb(37, 99, 235)"],
    [1.0, "rgb(191, 219, 254)"],
]

# ---------- 6. Build globe figure ----------
def _fmt_hover(row):
    if use_return and pd.notna(row.get("ret_fwd_1m", None)):
        return (
            f"{row['state_name']}<br>"
            f"Brightness: {row['avg_rad_month']:.4f}<br>"
            f"Next-month return: {row['ret_fwd_1m']:.3f}"
        )
    else:
        return (
            f"{row['state_name']}<br>"
            f"Brightness: {row['avg_rad_month']:.4f}"
        )

hover_text = state_df.apply(_fmt_hover, axis=1)

fig = go.Figure()
fig.add_trace(
    go.Scattergeo(
        lon=state_df["lon"],
        lat=state_df["lat"],
        text=hover_text,
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_intensity,
            colorscale=blue_scale,
            cmin=0,
            cmax=1,
            opacity=0.95,
        ),
    )
)

fig.update_geos(
    projection_type="orthographic",
    projection_rotation=dict(lon=-95, lat=35, roll=0),
    showcountries=True,
    showcoastlines=False,
    showland=True,
    landcolor="rgb(0,0,0)",
    showocean=True,
    oceancolor="rgb(0,0,0)",
    bgcolor="rgba(0,0,0,0)",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
)

# ---------- 7. Layout: globe + right-hand metrics panel ----------
left, right = st.columns([3.2, 1])

with left:
    st.plotly_chart(fig, use_container_width=True, height=650)

with right:
    st.markdown("<div class='anomaly-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>How to read the globe</div>", unsafe_allow_html=True)

    if use_return:
        desc = (
            "Each dot is a US state. Marker SIZE and BLUE intensity are scaled by "
            "the state's **expected next-month return** (ret_fwd_1m). "
            "Hover to see brightness and return."
        )
    else:
        desc = (
            "Each dot is a US state. Marker SIZE and BLUE intensity are scaled by "
            "average VIIRS nighttime radiance for the month. "
            "No forward returns are available in the dataset."
        )

    st.markdown(
        f"<p style='font-size:0.85rem; color:#b0b0c5;'>{desc}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics card
    st.markdown("<div class='anomaly-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Month metrics</div>", unsafe_allow_html=True)

    avg_brightness = state_df["avg_rad_month"].mean()
    st.markdown(
        f"<div class='metric-label'>Avg brightness</div>"
        f"<div class='metric-value'>{avg_brightness:.2f}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if use_return:
        avg_ret = state_df["ret_fwd_1m"].mean()
        st.markdown(
            f"<div class='metric-label'>Avg next-month return</div>"
            f"<div class='metric-value'>{avg_ret:.2%}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='metric-label'>Avg next-month return</div>"
            "<div class='metric-value'>—</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    f"Globe shows state-level hotspots for "
    f"{pd.Timestamp(selected_month).strftime('%Y-%m')}. "
    "Marker size and color are scaled by expected next-month return when available, "
    "otherwise by average VIIRS nighttime radiance."
)
