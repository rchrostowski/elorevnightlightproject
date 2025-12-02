# app.py

import streamlit as st

from src.load_data import load_model_data
from src.modeling import run_basic_regression

st.set_page_config(
    page_title="Nightlights x Stock Returns",
    page_icon="🌃",
    layout="wide",
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🌃 Nightlights × Stock Returns")
st.caption("FIN 377 • VIIRS night-time lights and S&P 500 performance")

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = load_model_data(fallback_if_missing=False)


required_cols = {"ticker", "date", "brightness_change", "ret_fwd"}
missing = required_cols - set(df.columns)

if missing:
    st.error(
        f"Final dataset is missing required columns: {missing}. "
        "Make sure you ran the pipeline (scripts/build_all.py) "
        "and that nightlights_model_data.csv has those columns."
    )
else:
    st.success(f"Loaded {len(df):,} firm–month observations.")

# ---------------------------------------------------------
# Project description
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("What this project does")
    st.markdown(
        """
        This project links **NASA/NOAA VIIRS night-time lights** to **S&P 500 stocks**:

        - Each firm is geocoded to a latitude/longitude.
        - We aggregate VIIRS brightness by location and month.
        - We compute **brightness changes** (growth in light intensity).
        - We test whether those changes predict **next-month stock returns**.

        Mathematically, we estimate regressions of the form:

        \n
        \\[
        r_{i, t+1} = \\alpha + \\beta \\cdot \\Delta \\text{Brightness}_{i, t} + \\varepsilon_{i, t+1}
        \\]

        where:
        - \( r_{i, t+1} \) is the stock's next-month return
        - \( \\Delta \\text{Brightness}_{i, t} \) is the change in VIIRS brightness
        """
    )

with col2:
    st.subheader("Regression snapshot")
    try:
        results = run_basic_regression()
        st.metric("β (brightness_change)", f"{results['beta_brightness']:.4f}")
        st.metric("t-stat (brightness_change)", f"{results['t_brightness']:.2f}")
        st.metric("R²", f"{results['r2']:.3f}")
        st.caption(f"Observations used: {results['n_obs']:,}")
        with st.expander("Show full regression summary"):
            st.text(results["summary"])
    except Exception as e:
        st.warning(f"Regression could not run yet: {e}")

st.markdown("---")

st.subheader("How to explore the data")

st.markdown(
    """
    Use the **pages in the left sidebar** to dig into the results:

    1. **Overview** – time-series plots, scatter of brightness vs returns, decile portfolios  
    2. **Ticker Explorer** – drill into specific firms (brightness over time, returns over time)  
    3. **Raw Data** – table + CSV download of the final modeling dataset  
    4. **Globe** – interactive VIIRS brightness globe by month  
    """
)

st.markdown(
    """
    Once your pipeline is fully connected to the real data, this app becomes
    a live explorer for your FIN 377 nightlight trading idea.
    """
)

