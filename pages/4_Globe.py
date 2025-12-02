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

# Rough state centroids (lat, lon) – visualization only
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

# ---------- 4. Filter data & aggregate by state ----------

lights_month = lights[lights["date"] == selected_date].copy()
if lights_month.empty:
    st.warning(
        f"No nightlights data for {pd.Timestamp(selected_date).strftime('%Y-%m')}."
    )
    st.stop()

# Average brightness per state
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

# ---------- 5. Attach state-level returns and use them for brightness/size ----------

# Default: no returns yet
state_df["ret_fwd_1m"] = pd.NA

if not model.empty and {"ret_fwd_1m", "date"}.issubset(model.columns):
    model_month = model[model["date"] == selected_date].copy()
    if not model_month.empty:
        # Try to aggregate by 'state' if present, otherwise by 'state_name'
        if "state" in model_month.columns:
            ret_state = (
                model_month.groupby("state", as_index=False)["ret_fwd_1m"]
                .mean()
                .rename(columns={"ret_fwd_1m": "ret_state"})
            )
            state_df = state_df.merge(ret_state, on="state", how="left")
        elif "state_name" in model_month.columns:
            ret_state = (
                model_month.groupby("state_name", as_index=False)["ret_fwd_1m"]
                .mean()
                .rename(columns={"ret_fwd_1m": "ret_state"})
            )
            state_df = state_df.merge(ret_state, on="state_name", how="left")
        else:
            state_df["ret_state"] = pd.NA

        state_df["ret_fwd_1m"] = state_df["ret_state"]
else:
    state_df["ret_state"] = pd.NA

# Use returns for visual encoding if available; otherwise fallback to brightness
use_return = state_df["ret_fwd_1m"].notna().any()

if use_return:
    r = state_df["ret_fwd_1m"].fillna(0)
    max_abs = r.abs().max()
    if max_abs == 0:
        r_norm = pd.Series(0.5, index=r.index)
    else:
        # map [-max, max] to [0, 1]
        r_norm = (r / max_abs + 1) / 2.0
    # marker size and brightness from returns
    marker_sizes = 8 + 22 * r_norm           # 8 → 30
    marker_intensity = r_norm                # 0 → dark, 1 → bright blue
    color_title = "State avg next-month return"
else:
    b = state_df["avg_rad_month"]
    if b.nunique() > 1:
        b_norm = (b - b.min()) / (b.max() - b.min())
    else:
        b_norm = pd.Series(0.5, index=b.index)
    marker_sizes = 8 + 22 * b_norm
    marker_intensity = b_norm
    color_title = "Avg brightness"

# Custom deep-blue colorscale
blue_scale = [
    [0.0, "rgb(2, 6, 23)"],
    [0.3, "rgb(13, 37, 88)"],
    [0.6, "rgb(37, 99, 235)"],
    [1.0, "rgb(191, 219, 254)"],
]

# ---------- 6. Build interactive spinning globe ----------

hover_col = "ret_fwd_1m" if use_return else "avg_rad_month"


def _fmt_hover(row):
    if use_return and pd.notna(row["ret_fwd_1m"]):
        return (
            f"{row['state_name']}<br>"
            f"ΔLight (avg): {row['avg_rad_month']:.4f}<br>"
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
    projection_type="orthographic",      # globe
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

# ---------- 7. Layout: globe + summary panel ----------

left, right = st.columns([3.2, 1.2])

with left:
    st.plotly_chart(fig, use_container_width=True, height=650)

with right:
    st.markdown("### Month summary")

    # Selected month
    st.markdown(
        f"**Selected month:** {pd.Timestamp(selected_date).strftime('%Y-%m')}"
    )

    # Average brightness (across all states for that month)
    avg_brightness = state_df["avg_rad_month"].mean()
    st.markdown("#### Averages")
    st.metric("Avg brightness", f"{avg_brightness:.2f}")

    # Average next-month return if available
    if use_return and state_df["ret_fwd_1m"].notna().any():
        avg_ret = state_df["ret_fwd_1m"].mean()
        st.metric("Avg next-month return", f"{avg_ret:.2%}")

    # Model-based metrics for this month (ΔLight and predicted return)
    if not model.empty and {"brightness_change", "ret_fwd_1m", "date"}.issubset(model.columns):
        model_month = model[model["date"] == selected_date].copy()
        if not model_month.empty:
            st.markdown("#### Model metrics")
            delta_light = model_month["brightness_change"].mean()
            pred_ret = model_month["ret_fwd_1m"].mean()
            st.metric("Avg ΔLight", f"{delta_light:.2f}")
            st.metric("Avg predicted return", f"{pred_ret:.2%}")

    # Top 5 brightest states
    st.markdown("#### Brightest states")
    top_states = (
        state_df.sort_values("avg_rad_month", ascending=False)
                .head(5)[["state", "state_name", "avg_rad_month"]]
                .rename(
                    columns={
                        "state": "Code",
                        "state_name": "State",
                        "avg_rad_month": "Brightness",
                    }
                )
    )
    st.table(top_states)

st.caption(
    f"Globe shows state-level hotspots for "
    f"{pd.Timestamp(selected_date).strftime('%Y-%m')}. "
    "Drag to rotate and explore different regions."
)




