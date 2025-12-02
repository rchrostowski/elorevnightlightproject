# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTER = DATA_DIR / "intermediate"
DATA_FINAL = DATA_DIR / "final"

DATA_INTER.mkdir(parents=True, exist_ok=True)
DATA_FINAL.mkdir(parents=True, exist_ok=True)

# Filenames
LIGHTS_RAW_FILE = "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"
SP500_RAW_FILE = "sp500_clean.csv"
RETURNS_RAW_FILE = "sp500_monthly_returns.csv"  # adjust if your file is named differently
MODEL_DATA_FILE = "nightlights_model_data.csv"

