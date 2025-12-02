# src/utils.py
import pandas as pd

def compute_deciles(df: pd.DataFrame, col: str, q: int = 10, label_col: str = "decile") -> pd.DataFrame:
    out = df.copy()
    if out[col].nunique() > 1:
        out[label_col] = pd.qcut(out[col], q, labels=False, duplicates="drop")
    else:
        out[label_col] = 0
    return out

