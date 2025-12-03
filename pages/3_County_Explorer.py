import streamlit as st
import pandas as pd
import plotly.express as px

from src.load_data import load_model_data

st.markdown("## 3. County Explorer – which HQ counties and how they behave")

df = load_model_data(fallback_if_missing=True)
if df.empty:
    st.error("Final dataset `nightlights_model_data.csv` is missing or empty.")
    st.stop()

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

needed = {"ticker", "firm", "county_name", "brightness_change", "ret_fwd_1m"}
missing = needed - set(df.columns)
if missing:
    st.error(f"`nightlights_model_data` must contain: {needed}. Missing: {missing}")
    st.stop()

# Try to find a state-like column, but don't require it
state_col = None
for cand in ["state", "hq_state", "state_name"]:
    if cand in df.columns:
        state_col = cand
        break

df["county_name"] = df["county_name"].astype(str)
df = df[df["county_name"].str.lower() != "n/a"]

df["brightness_change"] = pd.to_numeric(df["brightness_change"], errors="coerce")
df["ret_fwd_1m"] = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")
df = df.dropna(subset=["brightness_change", "ret_fwd_1m"])

if df.empty:
    st.error("No usable rows after cleaning `brightness_change` and `ret_fwd_1m`.")
    st.stop()

# Build a display label for counties
if state_col:
    df["state_display"] = df[state_col].astype(str)
    df["county_key"] = df["county_name"] + ", " + df["state_display"]
else:
    df["state_display"] = ""
    df["county_key"] = df["county_name"]

county_keys = sorted(df["county_key"].unique())
default_key = "Santa Clara County, CA" if "Santa Clara County, CA" in county_keys else county_keys[0]

county_key = st.selectbox("Select HQ county:", options=county_keys, index=county_keys.index(default_key))

if state_col:
    # Split "County, ST"
    if ", " in county_key:
        county_name, state_disp = county_key.split(", ", 1)
    else:
        county_name, state_disp = county_key, ""
    df_c = df[(df["county_name"] == county_name) & (df["state_display"] == state_disp)].copy()
else:
    county_name = county_key
    df_c = df[df["county_name"] == county_name].copy()

df_c = df_c.sort_values("date")

st.markdown(
    f"""
**Selected HQ county:** {county_key}  

This tab answers:

- Which **firms** are headquartered in this county?  
- How do their **next-month returns** behave when **local night-time lights change**?  
- How the county-level patterns connect back to the global regression.
"""
)

st.markdown("---")

# ----- A. Firms headquartered in this county -----
st.markdown("### A. Firms headquartered in this county")

firm_summary = (
    df_c.groupby("ticker", as_index=False)
        .agg(
            firm=("firm", "first"),
            n_obs=("date", "size"),
            avg_ret=("ret_fwd_1m", "mean"),
            avg_brightness_change=("brightness_change", "mean"),
        )
        .sort_values("n_obs", ascending=False)
)

st.dataframe(
    firm_summary.rename(
        columns={
            "ticker": "Ticker",
            "firm": "Firm",
            "n_obs": "# months",
            "avg_ret": "Avg next-month return",
            "avg_brightness_change": "Avg Δ brightness",
        }
    ),
    use_container_width=True,
)

st.markdown(
    """
**How to explain this table:**  

- Each row is a **firm whose HQ county is the one selected above**.  
- `# months` is how many ticker-months we see for that firm in the sample.  
- `Avg Δ brightness` is the **average change in the county’s night-lights** over the sample window.  
- `Avg next-month return` is the **average of `ret_fwd_1m`** for that firm.

This helps connect the **geography (county)** to **actual names of firms** people recognize.
"""
)

st.markdown("---")

# ----- B. County-level time series: average return and brightness -----
st.markdown("### B. County time series – average return vs average Δ brightness")

df_c_month = (
    df_c.groupby("date", as_index=False)
        .agg(
            avg_ret=("ret_fwd_1m", "mean"),
            avg_brightness_change=("brightness_change", "mean"),
        )
        .sort_values("date")
)

if len(df_c_month) < 3:
    st.warning("Not enough data for a meaningful time-series in this county.")
else:
    colL, colR = st.columns(2)

    with colL:
        fig_ret = px.line(
            df_c_month,
            x="date",
            y="avg_ret",
            title=f"{county_key}: average next-month total return over time",
            labels={"date": "Month", "avg_ret": "Avg next-month total return"},
        )
        fig_ret.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ret, use_container_width=True)

    with colR:
        fig_bright = px.line(
            df_c_month,
            x="date",
            y="avg_brightness_change",
            title=f"{county_key}: average Δ brightness over time",
            labels={"date": "Month", "avg_brightness_change": "Avg Δ brightness"},
        )
        fig_bright.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bright, use_container_width=True)

    st.markdown(
        """
**Narrative:**  

- The **left plot** shows how the **average next-month return** of HQ firms in this county evolves over time.  
- The **right plot** shows how the **average change in night-time brightness** of the county evolves.

The fixed-effects regression is essentially using this kind of information but **across all counties at once**,  
asking whether **unusually bright months in a county** are followed by **unusually high returns** for firms headquartered there.
"""
    )

st.markdown("---")

# ----- C. County scatter: ΔBrightness vs next-month returns (all ticker-months) -----
st.markdown("### C. County scatter – ΔBrightness vs next-month returns across firms")

if len(df_c) >= 10:
    fig_sc = px.scatter(
        df_c,
        x="brightness_change",
        y="ret_fwd_1m",
        color="ticker",
        labels={
            "brightness_change": "Δ brightness (HQ county)",
            "ret_fwd_1m": "Next-month total return",
            "ticker": "Ticker",
        },
        title=f"{county_key}: ΔBrightness vs next-month total returns for all HQ firms",
    )
    fig_sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown(
        """
**How this ties into the regression:**  

- This chart is a **micro version** of the main regression, focusing on a single county.  
- Each dot is a **firm–month**:
  - x-axis: the change in brightness around the HQ that month,  
  - y-axis: the firm’s next-month total return.

In the **Regression** tab, we:

- pool all counties and firms together,  
- add **year–month fixed effects** `C(year-month)`,  
- and estimate a global **β on `brightness_change`** that tells us whether brighter HQ months
  are systematically followed by higher or lower returns.
"""
    )
else:
    st.info("Too few observations in this county to show a meaningful scatter.")

