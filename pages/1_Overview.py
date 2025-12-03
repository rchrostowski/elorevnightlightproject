# pages/1_Overview.py

import streamlit as st
import pandas as pd
import altair as alt

from src.load_data import load_model_data

st.set_page_config(page_title="Overview", page_icon="🏠")

st.title("🏠 Night-Lights Overview")

df = load_model_data(fallback_if_missing=True).copy()
if df.empty:
    st.error("nightlights_model_data.csv is missing or empty.")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]

if "date" not in df.columns:
    st.error("Model data must have a 'date' column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# choose a return column
ret_col = None
for cand in ["ret_excess", "excess_ret", "mkt_excess_ret", "ret"]:
    if cand in df.columns:
        ret_col = cand
        break

brightness_col = None
for cand in ["brightness", "avg_rad_month", "avg_brightness"]:
    if cand in df.columns:
        brightness_col = cand
        break

min_date, max_date = df["date"].min(), df["date"].max()
n_obs = len(df)
n_tickers = df["ticker"].nunique() if "ticker" in df.columns else None

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Observations", f"{n_obs:,}")
with c2:
    st.metric("Tickers", n_tickers if n_tickers is not None else "n/a")
with c3:
    st.metric(
        "Date range",
        f"{min_date.strftime('%Y-%m')} → {max_date.strftime('%Y-%m')}",
    )

st.markdown("### Average over time")

df_m = (
    df.groupby(pd.Grouper(key="date", freq="M"))
    .agg(
        avg_ret=(ret_col, "mean") if ret_col else ("date", "size"),
        avg_brightness=(brightness_col, "mean") if brightness_col else ("date", "size"),
    )
    .reset_index()
)

charts = []

if ret_col:
    charts.append(
        alt.Chart(df_m)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("avg_ret:Q", title=f"Average {ret_col}"),
        )
        .properties(height=250)
    )

if brightness_col:
    charts.append(
        alt.Chart(df_m)
        .mark_line(color="lightblue")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("avg_brightness:Q", title="Average brightness"),
        )
        .properties(height=250)
    )

if charts:
    st.altair_chart(charts[0] if len(charts) == 1 else alt.layer(*charts), use_container_width=True)
else:
    st.info("No return or brightness columns found to plot.")


