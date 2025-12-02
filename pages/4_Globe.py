# pages/4_Globe.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

# ---------- 1. Load data ----------

lights = load_lights_monthly_by_coord(fallback_if_missing=True)
model = load_model_data(fallback_if_missing=True)

if lights.empty:
    st.error(
        "lights_monthly_by_coord.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` to rebuild it, commit the CSV in "
        "`data/intermediate/`, and redeploy."
    )
    st.stop()

required_light_cols = {"iso", "name_1", "date", "avg_rad_month"}
missing = required_light_cols - set(lights.columns)
if missing:
    st.error(
        f"Missing columns in lights_monthly_by_coord.csv: {missing}\n\n"
        f"Found columns: {lights.columns.tolist()}"
    )
    st.stop()

# Clean/standardize
lights = lights[lights["iso"] == "USA"].copy()
lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
lights = lights.dropna(subset=["date"])

if not model.empty and "date" in model.columns:
    model["date"] = pd.to_datetime(model["date"], errors="coerce")
    model = model.dropna(subset=["date"])

if lights.empty:
    st.error("No valid USA rows in lights panel after cleaning.")
    st.stop()

# ---------- 2. State mapping: names -> postal & lat/lon ----------

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
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

# Rough state centroids (lat, lon) – good enough for visualization
STATE_COORDS = {
    "AL": (32.7, -86.7),
    "AK": (64.8, -147.7),
    "AZ": (34.0, -111.7),
    "AR": (34.9, -92.3),
    "CA": (37.3, -119.7),
    "CO": (39.0, -105.5),
    "CT": (41.6, -72.7),
    "DE": (39.0, -75.5),
    "DC": (38.9, -77.0),
    "FL": (28.4, -82.5),
    "GA": (32.7, -83.3),
    "HI": (20.8, -157.0),
    "ID": (44.4, -114.4),
    "IL": (40.0, -89.2),
    "IN": (39.9, -86.3),
    "IA": (42.1, -93.5),
    "KS": (38.5, -98.0),
    "KY": (37.5, -85.3),
    "LA": (31.0, -92.0),
    "ME": (45.3, -69.2),
    "MD": (39.0, -76.7),
    "MA": (42.4, -71.8),
    "MI": (44.3, -85.4),
    "MN": (46.3, -94.2),
    "MS": (32.7, -89.7),
    "MO": (38.5, -92.3),
    "MT": (46.9, -110.4),
    "NE": (41.5, -99.8),
    "NV": (39.5, -116.6),
    "NH": (43.7, -71.6),
    "NJ": (40.1, -74.7),
    "NM": (34.2, -106.0),
    "NY": (42.9, -75.0),
    "NC": (35.5, -79.4),
    "ND": (47.5, -100.5),
    "OH": (40.3, -82.8),
    "OK": (35.6, -97.5),
    "OR": (44.0, -120.5),
    "PA": (41.0, -77.6),
    "RI": (41.7, -71.6),
    "SC": (33.8, -80.9),
    "SD": (44.4, -100.2),
    "TN": (35.9, -86.4),
    "TX": (31.0, -99.0),
    "UT": (39.3, -111.7),
    "VT": (44.0, -72.7),
    "VA": (37.5, -78.8),
    "WA": (47.4, -120.5),
    "WV": (38.6, -80.6),
    "WI": (44.6, -89.5),
    "WY": (43.1, -107.6),
    "PR": (18.2, -66.4),
}

lights["state_name"] = lights["name_1"].astype(str)
lights["state"] = lights["state_name"].map(STATE_ABBR)
lights = lights.dropna(subset=["state"])

if lights.empty:
    st.error(
        "Could not map any state names in `name_1` to US postal codes. "
        "Check that `name_1` contains US state names like 'California', etc."
    )
    st.stop()

# ---------- 3. Sidebar controls ----------

st.sidebar.header("Map Controls")

unique_dates = sorted(lights["date"].unique())
default_date = unique_dates[-1]

selected_date = st.sidebar.selectbox(
    "Select month:",
    options=unique_dates,
    index=len(unique_dates) - 1,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
)

# ---------- 4. Filter data & prepare hotspot points ----------

lights_month = lights[lights["date"] == selected_date].copy()

if lights_month.empty:
    st.warning(
        f"No data for selected month {pd.Timestamp(selected_date).strftime('%Y-%m')}."
    )
    st.stop()

state_df = (
    lights_month.groupby(["state", "state_name"], as_index=False)["avg_rad_month"]
    .mean()
)

# Attach coordinates
state_df["lat"] = state_df["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
state_df["lon"] = state_df["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
state_df = state_df.dropna(subset=["lat", "lon"])

if state_df.empty:
    st.error("No states had coordinates mapped for this month.")
    st.stop()

# Normalize brightness for marker sizing
b = state_df["avg_rad_month"]
if b.nunique() > 1:
    b_norm = (b - b.min()) / (b.max() - b.min())
else:
    b_norm = pd.Series(0.5, index=b.index)

marker_sizes = 6 + 18 * b_norm  # size between 6 and 24

# ---------- 5. Build interactive spinning globe ----------

fig = go.Figure()

fig.add_trace(
    go.Scattergeo(
        lon=state_df["lon"],
        lat=state_df["lat"],
        text=state_df.apply(
            lambda row: f"{row['state_name']}<br>Brightness: {row['avg_rad_month']:.4f}",
            axis=1,
        ),
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=state_df["avg_rad_month"],
            colorscale="Blues",
            cmin=b.min(),
            cmax=b.max(),
            opacity=0.95,
        ),
    )
)

fig.update_geos(
    projection_type="orthographic",      # <-- globe
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

# ---------- 6. Layout: globe + metrics panel ----------

left, right = st.columns([3.2, 1])

with left:
    st.plotly_chart(fig, use_container_width=True, height=650)

with right:
    # Brightness card
    st.markdown("<div class='anomaly-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Brightness Intensity</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem; color:#b0b0c5;'>"
        "Each hotspot represents a US state. Marker size and color track "
        "average VIIRS nighttime radiance for the selected month."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics card (ΔLight + predicted return)
    st.markdown("<div class='anomaly-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Metrics</div>", unsafe_allow_html=True)

    delta_light_display = "—"
    pred_ret_display = "—"

    if not model.empty and {"brightness_change", "ret_fwd_1m", "date"}.issubset(model.columns):
        model_month = model[model["date"] == selected_date].copy()
        if not model_month.empty:
            delta_light = model_month["brightness_change"].mean()
            pred_ret = model_month["ret_fwd_1m"].mean()
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

st.caption(
    f"Globe shows state-level nightlights hotspots for "
    f"{pd.Timestamp(selected_date).strftime('%Y-%m')}. "
    "Drag to rotate and explore different regions."
)



