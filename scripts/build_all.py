# scripts/build_all.py
from src.features import build_features_and_model_data

if __name__ == "__main__":
    print("Building full nightlights pipeline...")
    df = build_features_and_model_data()
    print("Done.")
    print(f"Rows in final model data: {len(df):,}")

