# pages/3_County_Explorer.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.load_data import load_lights_monthly_by_coord

st.set_page_config(page_title="County Explorer", page_icon="🗺️")

st.title("🗺️ County Night-Lights Explorer")

lights = load_lights_monthly_by_coord(fallback_if_missing=True)
if lights.empty:
    st.error(
        "lights_monthly_by_coord.csv is missing or empty.\n"
        "Run `python scripts/build_all.py` and commit "
        "`data/intermediate/lights_monthly_by_coord.csv`."
    )
    st.stop()

lights = lights.copy()
lights.columns = [c.strip().lower() for c in lights.columns]

required = {"iso", "date", "avg_rad_month"}
if not required.issubset(lights.columns):
    st.error(
        f"lights_monthly_by_coord.csv must have columns {required}.\n"
        f"Found: {lights.columns.tolist()}"
    )
    st.stop()

lights = lights[lights["iso"].str.upper() == "USA"].copy()
lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
lights = lights.dropna(subset=["date"])

lights["year_month"] = lights["date"].dt.to_period("M").astype(str)

area_col = "name_2" if "name_2" in lights.columns else None

with st.sidebar:
    st.header("Filters")

    ym_options = sorted(lights["year_month"].unique())
    ym = st.selectbox("Year-month", ym_options, index=len(ym_options) - 1)

    st.caption("Trim brightness outliers")
    p_low, p_high = st.slider(
        "Brightness percentile",
        0.0,
        100.0,
        (1.0, 99.0),
        step=1.0,
    )

df_m = lights[lights["year_month"] == ym].copy()
if df_m.empty:
    st.warning(f"No county data for {ym}.")
    st.stop()

b = df_m["avg_rad_month"]
low, high = np.percentile(b, [p_low, p_high])
df_m = df_m[df_m["avg_rad_month"].between(low, high)]

st.subheader(f"Brightness distribution across counties – {ym}")

hist = (
    alt.Chart(df_m)
    .mark_bar()
    .encode(
        x=alt.X("avg_rad_month:Q", bin=alt.Bin(maxbins=50), title="Brightness"),
        y=alt.Y("count():Q", title="Number of counties"),
    )
    .properties(height=250)
)
st.altair_chart(hist, use_container_width=True)

st.subheader("Top 20 brightest counties")

cols = ["avg_rad_month"]
if area_col:
    cols = [area_col] + cols

top = (
    df_m.sort_values("avg_rad_month", ascending=False)
    .head(20)[cols]
    .rename(columns={"name_2": "County", "avg_rad_month": "Brightness"})
)
st.dataframe(top, use_container_width=True)
