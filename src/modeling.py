# src/modeling.py
import statsmodels.api as sm
import pandas as pd
from .load_data import load_model_data

def run_basic_regression() -> dict:
    """
    ret_fwd ~ brightness_change with firm-clustered SE.
    Returns a small dict with key stats for the app / report.
    """
    df = load_model_data(fallback_if_missing=False).copy()

    reg = df.dropna(subset=["brightness_change", "ret_fwd"]).copy()

    X = reg[["brightness_change"]]
    X = sm.add_constant(X)
    y = reg["ret_fwd"]

    model = sm.OLS(y, X)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg["ticker"]}
    )

    return {
        "alpha": results.params.get("const", float("nan")),
        "beta_brightness": results.params.get("brightness_change", float("nan")),
        "t_brightness": results.tvalues.get("brightness_change", float("nan")),
        "p_brightness": results.pvalues.get("brightness_change", float("nan")),
        "r2": results.rsquared,
        "n_obs": int(results.nobs),
        "summary": results.summary().as_text(),
    }

