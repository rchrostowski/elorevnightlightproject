# pages/3_Raw_Data.py
import streamlit as st
from src.load_data import load_model_data

st.title("Raw Data View")

df = load_model_data(fallback_if_missing=True).copy()
df["date"] = df["date"].astype("datetime64[ns]")

st.write("Filtered view of the final modeling dataset.")

min_date, max_date = df["date"].min(), df["date"].max()
start, end = st.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM",
)

mask_date = (df["date"] >= start) & (df["date"] <= end)

tickers = sorted(df["ticker"].unique().tolist())
selected = st.multiselect(
    "Tickers",
    options=tickers,
    default=tickers[:20] if len(tickers) > 20 else tickers,
)
mask_ticker = df["ticker"].isin(selected) if selected else True

f = df[mask_date & mask_ticker].copy().sort_values(["ticker", "date"])

st.dataframe(f, use_container_width=True, height=500)

st.download_button(
    "Download filtered data as CSV",
    data=f.to_csv(index=False).encode("utf-8"),
    file_name="nightlights_filtered.csv",
    mime="text/csv",
)

