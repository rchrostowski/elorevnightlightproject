import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"

# VIIRS nightlights Dropbox URL (dl=1 = direct CSV)
LIGHTS_URL = (
    "https://www.dropbox.com/scl/fi/dxmu3q12hf7ovs0cdmnuz/"
    "VIIRS-nighttime-lights-2013m1to2024m5-level2.csv"
    "?rlkey=803izc59yiow71sgscawc1q6v&st=c0fgh0qq&dl=1"
)


# ------------------------------------------------------------
# 1. Raw nightlights (NO FILTERING HERE)
# ------------------------------------------------------------

def load_raw_lights() -> pd.DataFrame:
    """
    Load raw VIIRS level-2 nightlights from Dropbox.

    - Does NOT trim to any year/month.
    - Only fixes encoding / column names.
    - Date trimming (e.g. 2018+) is done in preprocess_lights.py
    """
    print("📥 Loading raw nightlights data...")
    try:
        df = pd.read_csv(LIGHTS_URL, low_memory=False)
    except UnicodeDecodeError:
        print("⚠️ UTF-8 failed, retrying with latin1...")
        try:
            df = pd.read_csv(LIGHTS_URL, encoding="latin1", low_memory=False)
        except UnicodeDecodeError:
            print("⚠️ latin1 failed, retrying with ISO-8859-1 + skip bad lines...")
            df = pd.read_csv(
                LIGHTS_URL,
                encoding="ISO-8859-1",
                low_memory=False,
                on_bad_lines="skip",
            )

    # normalize colnames
    df.columns = [c.strip().lower() for c in df.columns]

    expected = {"iso", "id_1", "name_1", "id_2", "name_2",
                "year", "month", "nlsum", "area"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw lights data missing {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    print(
        f"📅 Raw VIIRS years: {int(df['year'].min())} → {int(df['year'].max())}; "
        f"months: {int(df['month'].min())} → {int(df['month'].max())}"
    )
    print(f"📏 Raw lights rows: {len(df):,}")

    return df


# ------------------------------------------------------------
# 2. S&P500 firm info (raw & clean)
# ------------------------------------------------------------

def load_raw_sp500() -> pd.DataFrame:
    """
    Raw SP500 firm file used by older code.
    We point this at sp500_clean.csv.
    """
    path = RAW_DIR / "sp500_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Upload sp500_clean.csv to data/raw/.")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_sp500_clean() -> pd.DataFrame:
    """
    Clean SP500 firm locations with at least:
        ['ticker','company','lat','lon']
    'state' is optional.
    """
    df = load_raw_sp500()
    expected = {"ticker", "company", "lat", "lon"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"sp500_clean.csv missing {missing}. "
            f"Found: {df.columns.tolist()}"
        )
    return df


# ------------------------------------------------------------
# 3. Raw monthly returns (from fetch_monthly_returns.py)
# ------------------------------------------------------------

def load_raw_returns() -> pd.DataFrame:
    """
    Load raw monthly returns CSV produced by scripts/fetch_monthly_returns.py.

    Expected columns (lowercased): ['ticker', 'date', 'ret']
    """
    path = RAW_DIR / "sp500_monthly_returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run scripts/fetch_monthly_returns.py first."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# ------------------------------------------------------------
# 4. Processed nightlights panel (county-level with coords)
# ------------------------------------------------------------

def save_lights_monthly_by_coord(df: pd.DataFrame) -> None:
    """
    Helper for preprocess_lights to write the intermediate CSV.
    """
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERMEDIATE_DIR / "lights_monthly_by_coord.csv"
    df.to_csv(out_path, index=False)
    print(f"💾 Saved lights_monthly_by_coord.csv to {out_path}")


def load_lights_monthly_by_coord(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the processed lights panel (one row per region×month).

    If fallback_if_missing=True and the file doesn't exist, try to
    build it once using preprocess_lights.build_lights_monthly_by_coord().
    """
    path = INTERMEDIATE_DIR / "lights_monthly_by_coord.csv"
    if not path.exists():
        if not fallback_if_missing:
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/build_all.py to create it."
            )
        # lazy build to support Streamlit first-run
        print("⚙️ lights_monthly_by_coord.csv missing — building via preprocess_lights...")
        from .preprocess_lights import build_lights_monthly_by_coord

        df = build_lights_monthly_by_coord()
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# ------------------------------------------------------------
# 5. Final merged model dataset (nightlights + returns)
# ------------------------------------------------------------

def load_model_data(fallback_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the final merged dataset used by the dashboard.

    If fallback_if_missing=True and the file doesn't exist, build it once
    using features.build_features_and_model_data().
    """
    path = FINAL_DIR / "nightlights_model_data.csv"
    if not path.exists():
        if not fallback_if_missing:
            raise FileNotFoundError(
                f"Missing final dataset at {path}. "
                "Run scripts/build_all.py to generate it."
            )
        print("⚙️ nightlights_model_data.csv missing — building via features...")
        from .features import build_features_and_model_data

        df = build_features_and_model_data()
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df
