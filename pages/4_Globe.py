# pages/4_Globe.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.load_data import load_model_data

st.set_page_config(
    page_title="HQ Globe – Night Lights Anomalia",
    layout="wide",
)

def _get_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None

df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

lat_col = _get_col(df, ["hq_lat", "lat"])
lon_col = _get_col(df, ["hq_lon", "lon"])
if not lat_col or not lon_col:
    st.error("Expected HQ latitude/longitude columns (e.g., 'hq_lat', 'hq_lon').")
    st.stop()

brightness_col = _get_col(df, ["brightness_change", "d_light", "delta_light"], required=True)
ret_fwd_col = _get_col(df, ["ret_fwd_1m", "ret_fwd", "ret_forward_1m"])
ret_col = _get_col(df, ["ret_excess", "ret", "return"])

st.markdown("## HQ Globe – Hotspots by Night Lights and Returns")

st.sidebar.header("Globe controls")

dates = sorted(df["date"].unique())
default_idx = len(dates) - 1
selected_date = st.sidebar.selectbox(
    "Select month",
    options=dates,
    index=default_idx,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
)

encode_option = st.sidebar.radio(
    "Color/size encodes:",
    options=["Next-month return", "ΔLight (HQ county)"],
    index=0 if ret_fwd_col else 1,
)

df_m = df[df["date"] == selected_date].copy()

if df_m.empty:
    st.warning("No observations for the selected month.")
    st.stop()

# Collapse to one point per ticker (HQ)
group_cols = ["ticker", "firm", "state", "county_name", lat_col, lon_col]
group_cols = [c for c in group_cols if c in df_m.columns]

df_hq = (
    df_m.groupby(group_cols, as_index=False)
    .agg(
        d_light=(brightness_col, "mean"),
        ret_fwd=(ret_fwd_col, "mean") if ret_fwd_col and ret_fwd_col in df_m.columns else ("ticker", "size"),
    )
)

if ret_fwd_col and ret_fwd_col in df_m.columns:
    df_hq["ret_fwd"] = df_hq["ret_fwd"].astype(float)

if encode_option == "Next-month return" and ret_fwd_col and ret_fwd_col in df_m.columns:
    val = df_hq["ret_fwd"].fillna(0)
    label_title = "Next-month return"
else:
    val = df_hq["d_light"].fillna(0)
    label_title = "ΔLight (HQ county)"

max_abs = val.abs().max()
if max_abs == 0:
    v_norm = pd.Series(0.5, index=val.index)
else:
    v_norm = (val / max_abs + 1) / 2.0  # [-max,max] → [0,1]

marker_sizes = 5 + 25 * v_norm
marker_intensity = v_norm

def _hover(row):
    ticker = row.get("ticker", "")
    firm = row.get("firm", "")
    county = row.get("county_name", "")
    state = row.get("state", "")
    txt = f"{ticker} – {firm}<br>{county}, {state}<br>"
    txt += f"ΔLight: {row['d_light']:.3f}<br>"
    if "ret_fwd" in row and pd.notna(row["ret_fwd"]):
        txt += f"Next-month ret: {row['ret_fwd']:.2%}"
    return txt

hover_text = df_hq.apply(_hover, axis=1)

blue_scale = [
    [0.0, "rgb(2, 6, 23)"],
    [0.30, "rgb(13, 37, 88)"],
    [0.65, "rgb(37, 99, 235)"],
    [1.0, "rgb(191, 219, 254)"],
]

fig = go.Figure()

fig.add_trace(
    go.Scattergeo(
        lon=df_hq[lon_col],
        lat=df_hq[lat_col],
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
    projection_rotation=dict(lon=-95, lat=30, roll=0),
    showcountries=True,
    showcoastlines=False,
    showland=True,
    landcolor="rgb(0,0,0)",
    showocean=True,
    oceancolor="rgb(0,0,0)",
    bgcolor="rgba(0,0,0,0)",
)

fig.update_layout(
    title=f"HQ hotspots – {pd.Timestamp(selected_date):%Y-%m} "
          f"({label_title} encoded by color/size)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=40, b=0),
)

left, right = st.columns([3.2, 1.4])

with left:
    st.plotly_chart(fig, use_container_width=True, height=650)

with right:
    st.markdown("### Month summary")
    st.markdown(f"**Month:** {pd.Timestamp(selected_date):%Y-%m}")

    avg_dlight = df_hq["d_light"].mean()
    st.metric("Avg ΔLight (HQ)", f"{avg_dlight:.3f}")

    if "ret_fwd" in df_hq.columns and df_hq["ret_fwd"].notna().any():
        avg_ret = df_hq["ret_fwd"].mean()
        st.metric("Avg next-month return", f"{avg_ret:.2%}")

    st.markdown("#### Top brightening HQs")
    top = (
        df_hq.sort_values("d_light", ascending=False)
        .head(10)[["ticker", "firm", "county_name", "state", "d_light"]]
        .rename(
            columns={
                "ticker": "Ticker",
                "firm": "Firm",
                "county_name": "County",
                "state": "State",
                "d_light": "ΔLight",
            }
        )
    )
    st.dataframe(top, use_container_width=True)

st.caption("Drag to spin the globe. This uses HQ coordinates, not broad state averages.")




