# scripts/build_all.py

"""
Minimal build_all script for the nightlight project.

We now treat data/final/nightlights_model_data.csv as the canonical dataset
(you've already built it), so this script just sanity-checks and lightly
re-cleans it so the professor can run a single command.
"""

from pathlib import Path
import sys
import os

# --- Make sure the project root is on sys.path so `import src...` works ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load_data import load_model_data


def main() -> None:
    print("📥 Loading final nightlights × returns dataset...")
    df = load_model_data(fallback_if_missing=False)

    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Unique tickers: {df['ticker'].nunique():,}")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")

    out_path = Path("data/final/nightlights_model_data.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write back a clean version (dates parsed, columns standardized)
    df.to_csv(out_path, index=False)
    print(f"💾 Wrote cleaned model data to {out_path}")


if __name__ == "__main__":
    main()
