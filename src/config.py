# src/config.py
from pathlib import Path
import os
import shutil

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTER = DATA_DIR / "intermediate"
DATA_FINAL = DATA_DIR / "final"

# ---------------------------------------------------------
# Helper: ensure a directory exists, even if a file is blocking it
# ---------------------------------------------------------

def ensure_directory(path: Path):
    """
    Ensures that `path` is a directory.
    If a FILE exists with this name, renames it to <name>_backup
    and then creates the directory.
    """
    if path.exists():
        if not path.is_dir():
            # A FILE exists where a folder should be → rename it safely
            backup_path = path.with_name(path.name + "_backup")
            shutil.move(str(path), str(backup_path))
            print(f"⚠️ Renamed file blocking folder: {path} → {backup_path}")
            path.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


# Ensure all required directories exist
ensure_directory(DATA_DIR)
ensure_directory(DATA_RAW)
ensure_directory(DATA_INTER)
ensure_directory(DATA_FINAL)

# ---------------------------------------------------------
# Filenames (local files)
# ---------------------------------------------------------

LIGHTS_RAW_FILE = "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"
SP500_RAW_FILE = "sp500_clean.csv"
RETURNS_RAW_FILE = "sp500_monthly_returns.csv"

# Final model output file
MODEL_DATA_FILE = "nightlights_model_data.csv"

# ---------------------------------------------------------
# Dropbox URL for VIIRS nightlights
# ---------------------------------------------------------

LIGHTS_URL = (
    "https://www.dropbox.com/scl/fi/dxmu3q12hf7ovs0cdmnuz/"
    "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"
    "?rlkey=803izc59yiow71sgscawc1q6v&st=c0fgh0qq&dl=1"
)

