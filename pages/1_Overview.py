# pages/1_Overview.py
import streamlit as st
import altair as alt
import numpy as np
from src.load_data import load_model_data
from src.utils import compute_deciles

st.title("Overview: Brightness vs Future Returns")

df = load_model_data(fallback_if_missing=True).copy()
df["date"] = df["date"].astype("datetime64[ns]")

required = {"ticker", "date", "brightness_change", "ret_fwd"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters (Overview)")
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM",
)
start, end = date_range
mask_date = (df["date"] >= start) & (df["date"] <= end)

all_tickers = sorted(df["ticker"].unique().tolist())
selected_tickers = st.sidebar.multiselect(
    "Tickers",
    options=all_tickers,
    default=all_tickers if len(all_tickers) <= 50 else all_tickers[:50],
)
mask_ticker = df["ticker"].isin(selected_tickers) if selected_tickers else True

q_low, q_high = df["brightness_change"].quantile(0.01), df["brightness_change"].quantile(0.99)
bmin, bmax = st.sidebar.slider(
    "Brightness change range",
    min_value=float(np.round(q_low, 2)),
    max_value=float(np.round(q_high, 2)),
    value=(float(np.round(q_low, 2)), float(np.round(q_high, 2))),
)
mask_bright = df["brightness_change"].between(bmin, bmax)

f = df[mask_date & mask_ticker & mask_bright].copy()
if f.empty:
    st.warning("No data for selected filters.")
    st.stop()

# KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Obs", f"{len(f):,}")
with c2:
    st.metric("Avg brightness change", f"{f['brightness_change'].mean():.3f}")
with c3:
    st.metric("Avg next-month return", f"{f['ret_fwd'].mean()*100:.2f}%")
with c4:
    corr = f["brightness_change"].corr(f["ret_fwd"])
    st.metric("Corr(bright, ret)", f"{corr:.3f}" if not np.isnan(corr) else "N/A")

st.markdown("---")

# Time-series chart
st.subheader("Time-series: average brightness change & future returns")

ts = (
    f.groupby("date", as_index=False)[["brightness_change", "ret_fwd"]]
    .mean()
    .rename(columns={"ret_fwd": "Next-month return", "brightness_change": "Brightness change"})
)

base = alt.Chart(ts).encode(x="date:T")

line_bright = base.mark_line().encode(
    y=alt.Y("Brightness change:Q", axis=alt.Axis(title="Brightness change (avg)")),
)
line_ret = base.mark_line(strokeDash=[4, 4]).encode(
    y=alt.Y("Next-month return:Q", axis=alt.Axis(title="Next-month return (avg)")),
    color=alt.value("#FF7F0E"),
)

st.altair_chart(alt.layer(line_bright, line_ret).resolve_scale(y="independent").interactive(),
                use_container_width=True)

# Scatter
st.subheader("Scatter: brightness change vs next-month return")

scatter = (
    alt.Chart(f)
    .mark_circle(size=40, opacity=0.5)
    .encode(
        x=alt.X("brightness_change:Q", title="Brightness change"),
        y=alt.Y("ret_fwd:Q", title="Next-month return"),
        tooltip=["ticker", "date:T", "brightness_change", "ret_fwd"],
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

# Decile chart
st.subheader("Decile portfolios by brightness change")

dec = compute_deciles(f, "brightness_change", q=10, label_col="brightness_decile")
dec_ret = (
    dec.groupby("brightness_decile", as_index=False)["ret_fwd"]
    .mean()
    .rename(columns={"ret_fwd": "Avg next-month return"})
)

bar = (
    alt.Chart(dec_ret)
    .mark_bar()
    .encode(
        x=alt.X("brightness_decile:O", title="Brightness decile (0=lowest, 9=highest)"),
        y=alt.Y("Avg next-month return:Q", title="Avg next-month return"),
        tooltip=["brightness_decile", "Avg next-month return"],
    )
)

st.altair_chart(bar, use_container_width=True)

