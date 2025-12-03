# pages/4_Globe.py

import streamlit as st
import pandas as pd
import numpy as np
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

# Clean / standardize
if not lights.empty:
    lights = lights.copy()
    lights.columns = [c.strip().lower() for c in lights.columns]
    if "date" in lights.columns:
        lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
        lights = lights.dropna(subset=["date"])

if not model.empty:
    model = model.copy()
    model.columns = [c.strip().lower() for c in model.columns]
    if "date" in model.columns:
        model["date"] = pd.to_datetime(model["date"], errors="coerce")
        model = model.dropna(subset=["date"])

# ---------- 2. Try to build a point-level globe from MODEL (preferred) ----------

lat_col = None
lon_col = None
for cand in ["lat_round", "lat"]:
    if not model.empty and cand in model.columns:
        lat_col = cand
        break

for cand in ["lon_round", "lon"]:
    if not model.empty and cand in model.columns:
        lon_col = cand
        break

# pick brightness column
brightness_col = None
if not model.empty:
    for cand in ["brightness", "avg_rad_month", "avg_brightness"]:
        if cand in model.columns:
            brightness_col = cand
            break

# pick return column (market excess / risk excess / plain)
ret_col = None
if not model.empty:
    for cand in [
        "ret_excess",
        "excess_ret",
        "mkt_excess_ret",
        "risk_excess",
        "ret_fwd_1m",
        "ret",
    ]:
        if cand in model.columns:
            ret_col = cand
            break

use_model_points = (
    (not model.empty)
    and lat_col is not None
    and lon_col is not None
    and ret_col is not None
)

# ---------- 3A. MODEL-BASED GLOBE (HQ / county-level points) ----------
if use_model_points:
    df = model.copy()

    # Year-month factor
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # sidebar controls
    st.sidebar.header("Map Controls")

    ym_options = sorted(df["year_month"].dropna().unique())
    default_ym = ym_options[-1] if ym_options else None

    selected_ym = st.sidebar.selectbox(
        "Select month:",
        options=ym_options,
        index=len(ym_options) - 1 if ym_options else 0,
    )

    # percentile trimming sliders
    st.sidebar.caption("Filter brightness percentiles")
    bright_low, bright_high = st.sidebar.slider(
        "Brightness Percentile",
        0.0, 100.0, (5.0, 95.0),
        step=1.0,
    )

    st.sidebar.caption("Filter return percentiles")
    ret_low, ret_high = st.sidebar.slider(
        "Return Percentile",
        0.0, 100.0, (5.0, 95.0),
        step=1.0,
    )

    # filter to selected month
    df_m = df[df["year_month"] == selected_ym].copy()

    if df_m.empty:
        st.warning(f"No model data for {selected_ym}.")
        st.stop()

    # aggregate at lat/lon to avoid a thousand identical stacked points
    group_cols = [lat_col, lon_col]
    agg_cols = {ret_col: "mean"}
    if brightness_col is not None:
        agg_cols[brightness_col] = "mean"

    df_points = (
        df_m.groupby(group_cols, as_index=False)
        .agg(agg_cols)
        .dropna(subset=[lat_col, lon_col])
    )

    # rename for convenience
    df_points = df_points.rename(
        columns={
            lat_col: "lat",
            lon_col: "lon",
        }
    )

    # percentile trims
    r_series = df_points[ret_col].dropna()
    if r_series.empty:
        st.error(f"No valid {ret_col} values in this month.")
        st.stop()

    rl, rh = np.percentile(r_series, [ret_low, ret_high])
    df_points = df_points[df_points[ret_col].between(rl, rh)]

    if brightness_col is not None:
        b_series = df_points[brightness_col].dropna()
        if not b_series.empty:
            bl, bh = np.percentile(b_series, [bright_low, bright_high])
            df_points = df_points[df_points[brightness_col].between(bl, bh)]

    if df_points.empty:
        st.warning("No data after applying percentile filters.")
        st.stop()

    # normalize return for size
    r = df_points[ret_col].fillna(0.0)
    max_abs_ret = r.abs().max()
    if max_abs_ret == 0:
        ret_norm = pd.Series(0.5, index=r.index)
    else:
        ret_norm = r.abs() / max_abs_ret  # 0 → small, 1 → big

    marker_sizes = 8 + 22 * ret_norm  # 8 → 30

    # normalize brightness for color (or use return if brightness missing)
    if brightness_col is not None:
        b = df_points[brightness_col].fillna(b_series.mean())
        if b.nunique() > 1:
            b_norm = (b - b.min()) / (b.max() - b.min())
        else:
            b_norm = pd.Series(0.5, index=b.index)
        marker_intensity = b_norm
        color_title = f"Brightness ({brightness_col})"
    else:
        # no brightness: color based on signed return
        if max_abs_ret == 0:
            color_norm = pd.Series(0.5, index=r.index)
        else:
            color_norm = (r / max_abs_ret + 1) / 2.0
        marker_intensity = color_norm
        color_title = f"Return ({ret_col})"

    # blue-ish colorscale
    blue_scale = [
        [0.0, "rgb(2, 6, 23)"],
        [0.3, "rgb(13, 37, 88)"],
        [0.6, "rgb(37, 99, 235)"],
        [1.0, "rgb(191, 219, 254)"],
    ]

    # hover text
    def _fmt_hover(row):
        base = f"Lat {row['lat']:.2f}, Lon {row['lon']:.2f}<br>"
        btxt = ""
        if brightness_col is not None and brightness_col in row:
            btxt = f"Brightness: {row[brightness_col]:.4f}<br>"
        rtxt = f"Return ({ret_col}): {row[ret_col]:.3f}"
        return base + btxt + rtxt

    hover_text = df_points.apply(_fmt_hover, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=df_points["lon"],
            lat=df_points["lat"],
            text=hover_text,
            hoverinfo="text",
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=marker_intensity,
                colorscale=blue_scale,
                cmin=0,
                cmax=1,
                opacity=0.97,
            ),
        )
    )

    fig.update_geos(
        projection_type="orthographic",   # spinning globe-style
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

        st.markdown(f"**Selected month:** {selected_ym}")

        st.markdown("#### Averages")
        if brightness_col is not None:
            avg_brightness = df_points[brightness_col].mean()
            st.metric("Avg brightness", f"{avg_brightness:.3f}")
        avg_ret = df_points[ret_col].mean()
        st.metric(
            "Avg return",
            f"{avg_ret:.2%}" if abs(avg_ret) < 1 else f"{avg_ret:.3f}",
        )

        # Top 10 brightest points (if brightness exists)
        if brightness_col is not None:
            st.markdown("#### Brightest locations")
            top = (
                df_points.sort_values(brightness_col, ascending=False)
                .head(10)[["lat", "lon", brightness_col, ret_col]]
                .rename(
                    columns={
                        "lat": "Lat",
                        "lon": "Lon",
                        brightness_col: "Brightness",
                        ret_col: "Return",
                    }
                )
            )
            st.table(top)

    # Caption clarifying what return is
    if "excess" in (ret_col or "") or "mkt" in (ret_col or ""):
        ret_label = "Monthly **excess** return"
    else:
        ret_label = "Monthly return"

    st.caption(
        f"Each point is a location (county or HQ) for {selected_ym}. "
        f"Color ≈ brightness (nightlights), size ≈ |{ret_label}| ({ret_col})."
    )

# ---------- 3B. FALLBACK: use the old state-level aggregation if model points unavailable ----------
else:
    # If model data doesn't have lat/lon + returns, fall back to your original
    # state-level brightness globe using lights + rough state centroids.

    if lights.empty:
        st.error(
            "Neither model-level globe data nor valid lights data is available.\n\n"
            "Run `python scripts/build_all.py` to rebuild intermediate and final data."
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

    lights = lights[lights["iso"] == "usa"].copy()
    if lights.empty:
        st.error("No valid USA rows in lights panel after cleaning.")
        st.stop()

    # ---------- Rough state mapping (your original logic) ----------
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

    st.sidebar.header("Map Controls")
    unique_dates = sorted(lights["date"].unique())
    default_date = unique_dates[-1]
    selected_date = st.sidebar.selectbox(
        "Select month:",
        options=unique_dates,
        index=len(unique_dates) - 1,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
    )

    lights_month = lights[lights["date"] == selected_date].copy()
    if lights_month.empty:
        st.warning(
            f"No nightlights data for {pd.Timestamp(selected_date).strftime('%Y-%m')}."
        )
        st.stop()

    state_df = (
        lights_month.groupby(["state", "state_name"], as_index=False)["avg_rad_month"]
        .mean()
    )
    state_df["lat"] = state_df["state"].map(
        lambda s: STATE_COORDS.get(s, (None, None))[0]
    )
    state_df["lon"] = state_df["state"].map(
        lambda s: STATE_COORDS.get(s, (None, None))[1]
    )
    state_df = state_df.dropna(subset=["lat", "lon"])
    if state_df.empty:
        st.error("No states had coordinates mapped for this month.")
        st.stop()

    # brightness → color and size
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

    def _fmt_hover_state(row):
        return (
            f"{row['state_name']}<br>"
            f"Brightness: {row['avg_rad_month']:.4f}"
        )

    hover_text = state_df.apply(_fmt_hover_state, axis=1)

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

    left, right = st.columns([3.2, 1.2])

    with left:
        st.plotly_chart(fig, use_container_width=True, height=650)

    with right:
        st.markdown("### Month summary")
        st.markdown(
            f"**Selected month:** {pd.Timestamp(selected_date).strftime('%Y-%m')}"
        )

        avg_brightness = state_df["avg_rad_month"].mean()
        st.markdown("#### Averages")
        st.metric("Avg brightness", f"{avg_brightness:.2f}")

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
        f"{pd.Timestamp(selected_date).strftime('%Y-%m')}."
    )





