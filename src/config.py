# src/config.py
from pathlib import Path

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTER = DATA_DIR / "intermediate"
DATA_FINAL = DATA_DIR / "final"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_INTER.mkdir(parents=True, exist_ok=True)
DATA_FINAL.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Filenames (local files)
# ---------------------------------------------------------

# You DO NOT need the full nightlights CSV locally because it is huge.
# But this is the default filename in case you ever want a local copy.
LIGHTS_RAW_FILE = "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"

# These MUST exist locally in data/raw/
SP500_RAW_FILE = "sp500_clean.csv"             # your firm → lat/long file
RETURNS_RAW_FILE = "sp500_monthly_returns.csv"  # your monthly returns file

# Final model output from pipeline
MODEL_DATA_FILE = "nightlights_model_data.csv"

# ---------------------------------------------------------
# Direct link to VIIRS nightlights data (Dropbox)
# ---------------------------------------------------------
# IMPORTANT: Must use dl=1 to force direct CSV download for pandas
LIGHTS_URL = (
    "https://www.dropbox.com/scl/fi/dxmu3q12hf7ovs0cdmnuz/"
    "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"
    "?rlkey=803izc59yiow71sgscawc1q6v&st=c0fgh0qq&dl=1"
)
