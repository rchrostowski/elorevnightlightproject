# pages/4_Globe.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.load_data import load_model_data

# ---------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="HQ Globe – Nightlights Anomalía",
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
    "<h1 style='margin-bottom: 0.25rem;'>HQ Globe: Nightlights Anomalía</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
This globe view places **each S&P 500 headquarters on an interactive Earth** and lets you
visualize different **signal layers**:

- **Brightness level** – average night-time light in the HQ county  
- **Brightness change (ΔLight)** – month-over-month change in brightness  
- **Next-month return** – forward stock return associated with each HQ  
- **R² by ticker** – how well ΔLight explains next-month returns for that ticker across time  

Use the controls on the left to pick a **month** and a **metric**, then drag the globe to explore
where the strongest signals are concentrated.
"""
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = load_model_data(fallback_if_missing=True)

if df.empty:
    st.error(
        "nightlights_model_data.csv is missing or empty.\n\n"
        "Run `python scripts/build_all.py` to rebuild it, commit the CSV in "
        "`data/final/`, and redeploy."
    )
    st.stop()

required_cols = {
    "ticker",
    "firm",
    "lat",
    "lon",
    "date",
    "brightness_change",
}

missing = required_cols - set(df.columns)
if missing:
    st.error(
        f"nightlights_model_data.csv must contain: {required_cols}. "
        f"Missing: {missing}\n\nFound columns: {df.columns.tolist()}"
    )
    st.stop()

# Forward returns column
if "ret_fwd_1m" in df.columns:
    pass
elif "ret_fwd" in df.columns:
    df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd"], errors="coerce")
else:
    df["ret_fwd_1m"] = np.nan  # we'll just disable return-based metrics if missing

# Brightness level (if available)
if "avg_rad_month" not in df.columns:
    df["avg_rad_month"] = np.nan

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

df = df.dropna(subset=["date", "lat", "lon"])

if df.empty:
    st.error("After cleaning, there are no rows with valid date/lat/lon.")
    st.stop()

# Optional state/county info
if "state_full" in df.columns:
    df["state_display"] = df["state_full"].astype(str)
elif "state" in df.columns:
    df["state_display"] = df["state"].astype(str)
else:
    df["state_display"] = ""

if "county_name" in df.columns:
    df["county_display"] = df["county_name"].astype(str)
else:
    df["county_display"] = ""

# ---------------------------------------------------------
# Precompute R² by ticker (brightness_change → next-month return)
# ---------------------------------------------------------
def ticker_r2(group: pd.DataFrame) -> float:
    g = group.dropna(subset=["brightness_change", "ret_fwd_1m"]).copy()
    if len(g) < 8:
        return np.nan
    x = g["brightness_change"]
    y = g["ret_fwd_1m"]
    if x.var() == 0 or y.var() == 0:
        return np.nan
    r = x.corr(y)
    if pd.isna(r):
        return np.nan
    return float(np.sign(r) * (r ** 2))

r2_rows = []
for tkr, sub in df.groupby("ticker"):
    r2 = ticker_r2(sub)
    if not np.isnan(r2):
        r2_rows.append({"ticker": tkr, "r2_signed": r2, "r2_abs": abs(r2)})

r2_df = pd.DataFrame(r2_rows)
if not r2_df.empty:
    df = df.merge(r2_df, on="ticker", how="left")
else:
    df["r2_signed"] = np.nan
    df["r2_abs"] = np.nan

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Globe Controls")

unique_dates = sorted(df["date"].dt.to_period("M").unique())
# Convert Period back to Timestamp for selectbox
unique_months = [p.to_timestamp() for p in unique_dates]

if not unique_months:
    st.error("No valid dates found in the dataset.")
    st.stop()

default_date = unique_months[-1]

selected_month = st.sidebar.selectbox(
    "Select month:",
    options=unique_months,
    index=len(unique_months) - 1,
    format_func=lambda d: d.strftime("%Y-%m"),
)

metric_options = {
    "Brightness level (avg_rad_month)": "avg_rad_month",
    "Brightness change (ΔLight)": "brightness_change",
    "Next-month return": "ret_fwd_1m",
    "R² by ticker (ΔLight → next-month return)": "r2_signed",
}

metric_label = st.sidebar.selectbox(
    "Marker metric:",
    options=list(metric_options.keys()),
    index=1,  # default to ΔLight
)

metric_col = metric_options[metric_label]

st.sidebar.markdown(
    """
**Tip:**  
- Use ΔLight to see where local economies are **accelerating or slowing**.  
- Use Next-month return to see where markets **rewarded or punished** firms.  
- Use R² to see where brightness has been the most **informative signal** historically.
"""
)

# ---------------------------------------------------------
# Filter for selected month
# ---------------------------------------------------------
df_month = df[df["date"].dt.to_period("M") == selected_month.to_period("M")].copy()

if df_month.empty:
    st.warning(f"No HQ observations for {selected_month.strftime('%Y-%m')}.")
    st.stop()

# If metric is static (R²), it doesn't actually vary by month, but we still display points for that month
values = df_month[metric_col].copy()

# If chosen metric is entirely NaN, fall back to ΔLight
if values.isna().all():
    st.warning(
        f"Metric `{metric_label}` is not available for this month. "
        "Falling back to brightness change (ΔLight)."
    )
    metric_label = "Brightness change (ΔLight)"
    metric_col = "brightness_change"
    values = df_month[metric_col].copy()

# ---------------------------------------------------------
# Normalize metric for marker size/color
# ---------------------------------------------------------
vals = values.fillna(0)

# Robust normalization: winsorize at 5th/95th percentiles
low, high = np.percentile(vals, [5, 95]) if len(vals) > 10 else (vals.min(), vals.max())
if high == low:
    norm = pd.Series(0.5, index=vals.index)
else:
    clipped = vals.clip(low, high)
    norm = (clipped - clipped.min()) / (clipped.max() - clipped.min())

marker_sizes = 8 + 22 * norm  # between 8 and 30
marker_intensity = norm       # 0 to 1

# Deep-blue colorscale
blue_scale = [
    [0.0, "rgb(2, 6, 23)"],
    [0.3, "rgb(13, 37, 88)"],
    [0.6, "rgb(37, 99, 235)"],
    [1.0, "rgb(191, 219, 254)"],
]

# ---------------------------------------------------------
# Hover text
# ---------------------------------------------------------
def fmt_val(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    # For returns / R² we want ~percentage style; for brightness leave raw
    return f"{v:.4f}"

hover_text = []
for _, row in df_month.iterrows():
    base = f"<b>{row['ticker']}</b> – {row['firm']}"
    loc = ""
    if row.get("county_display"):
        loc += row["county_display"]
    if row.get("state_display"):
        if loc:
            loc += f", {row['state_display']}"
        else:
            loc = row["state_display"]
    if loc:
        base += f"<br>{loc}"
    metric_value = row.get(metric_col, np.nan)
    if metric_label.startswith("R²"):
        metric_str = f"Signed R²: {fmt_val(metric_value)}"
    elif "return" in metric_label.lower():
        metric_str = f"Next-month return: {fmt_val(metric_value)}"
    elif "ΔLight" in metric_label:
        metric_str = f"ΔLight: {fmt_val(metric_value)}"
    else:
        metric_str = f"Brightness: {fmt_val(metric_value)}"

    hover_text.append(base + "<br>" + metric_str)

# ---------------------------------------------------------
# Build globe figure
# ---------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scattergeo(
        lon=df_month["lon"],
        lat=df_month["lat"],
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

# ---------------------------------------------------------
# Layout: globe + summary panel
# ---------------------------------------------------------
left, right = st.columns([3.2, 1.3])

with left:
    st.plotly_chart(fig, use_container_width=True, height=650)

with right:
    st.markdown("### Month summary")

    st.markdown(
        f"**Selected month:** {selected_month.strftime('%Y-%m')}"
    )

    st.markdown(f"**Marker metric:** {metric_label}")

    # Basic stats for the chosen metric
    metric_series = values.dropna()
    if not metric_series.empty:
        st.markdown("#### Distribution")
        st.write(
            f"Mean: `{metric_series.mean():.4f}`  •  "
            f"Median: `{metric_series.median():.4f}`  •  "
            f"Std: `{metric_series.std():.4f}`"
        )

    # Top 5 firms by metric
    st.markdown("#### Top 5 HQs by metric")

    top_df = (
        df_month[["ticker", "firm", "county_display", "state_display", metric_col]]
        .dropna(subset=[metric_col])
        .copy()
    )

    if not top_df.empty:
        top_df = top_df.sort_values(metric_col, ascending=False).head(5)
        top_df = top_df.rename(
            columns={
                "ticker": "Ticker",
                "firm": "Firm",
                "county_display": "County",
                "state_display": "State",
                metric_col: metric_label,
            }
        )
        st.table(top_df)
    else:
        st.caption("No non-missing metric values for this month.")

st.caption(
    f"The globe shows **firm HQs** for {selected_month.strftime('%Y-%m')} with marker size/color "
    f"based on **{metric_label}**. Drag to rotate the Earth and explore clusters of strong signals."
)

