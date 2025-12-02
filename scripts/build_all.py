# scripts/build_all.py

import sys
from pathlib import Path

# Make sure the project root is on sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features_and_model_data


if __name__ == "__main__":
    print("🚧 Building nightlights panel and model data...")
    df = build_features_and_model_data()
    print(f"✅ Done. Rows in final model data: {len(df):,}")

