# pages/2_Ticker_Explorer.py
import streamlit as st
import altair as alt
from src.load_data import load_model_data

st.title("Ticker Explorer")

df = load_model_data(fallback_if_missing=False).copy()
df["date"] = df["date"].astype("datetime64[ns]")

required = {"ticker", "date", "brightness_change", "ret_fwd", "avg_rad_month"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

tickers = sorted(df["ticker"].unique().tolist())
chosen = st.multiselect(
    "Select tickers to view",
    options=tickers,
    default=tickers[:5] if len(tickers) > 5 else tickers,
)

if not chosen:
    st.info("Select at least one ticker.")
    st.stop()

f = df[df["ticker"].isin(chosen)].copy()

st.subheader("Brightness level over time")
bright_chart = (
    alt.Chart(f)
    .mark_line()
    .encode(
        x="date:T",
        y=alt.Y("avg_rad_month:Q", title="Brightness (avg_rad_month)"),
        color="ticker:N",
        tooltip=["ticker", "date:T", "avg_rad_month"],
    )
    .interactive()
)

st.altair_chart(bright_chart, use_container_width=True)

st.subheader("Brightness change vs next-month return (per ticker)")

scatter = (
    alt.Chart(f)
    .mark_circle(size=50, opacity=0.5)
    .encode(
        x=alt.X("brightness_change:Q", title="Brightness change"),
        y=alt.Y("ret_fwd:Q", title="Next-month return"),
        color="ticker:N",
        tooltip=["ticker", "date:T", "brightness_change", "ret_fwd"],
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

st.subheader("Per-ticker time-series (brightness change & returns)")
ticker_choice = st.selectbox("Single ticker detail", options=chosen)

tf = f[f["ticker"] == ticker_choice].sort_values("date").copy()

col1, col2 = st.columns(2)
with col1:
    st.line_chart(
        tf.set_index("date")["brightness_change"],
        height=250,
    )
    st.caption(f"{ticker_choice}: brightness change over time")

with col2:
    st.line_chart(
        tf.set_index("date")["ret_fwd"],
        height=250,
    )
    st.caption(f"{ticker_choice}: next-month returns over time")

