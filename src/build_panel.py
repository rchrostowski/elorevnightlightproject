# src/build_panel.py

import pandas as pd
from pathlib import Path

ROOT = Path("data")


def _load_hq_mapping():
    """
    Load firm HQ → county mapping.

    Prefers:
        data/intermediate/hq_with_county.csv   (professor's correct mapping)

    Falls back to:
        data/intermediate/firm_hq_county.csv   (our derived mapping, if needed)
    """
    path_hq1 = ROOT / "intermediate" / "hq_with_county.csv"
    path_hq2 = ROOT / "intermediate" / "firm_hq_county.csv"

    if path_hq1.exists():
        hq = pd.read_csv(path_hq1)
        print(f"📁 Using HQ mapping from {path_hq1}")
        cols = list(hq.columns)
        # From your sample:
        # ticker, firm, state_abbrev, lat, lon, state_full, state_abbrev2, county_name, county_fips
        if len(cols) >= 9:
            hq = hq.rename(
                columns={
                    cols[0]: "ticker",
                    cols[1]: "firm",
                    cols[5]: "state_full",   # e.g., "California"
                    cols[7]: "county_name",  # e.g., "Santa Clara County"
                    cols[8]: "county_fips",
                    cols[3]: "lat",
                    cols[4]: "lon",
                }
            )
        else:
            raise ValueError(
                f"hq_with_county.csv format unexpected. Columns: {cols}"
            )
    elif path_hq2.exists():
        hq = pd.read_csv(path_hq2)
        print(f"📁 Using HQ mapping from {path_hq2}")
        hq.columns = [c.strip().lower() for c in hq.columns]
        # Expect: ticker, firm, county_fips, county_name, state, lat, lon
        missing = {"ticker", "firm", "county_name", "state"} - set(hq.columns)
        if missing:
            raise ValueError(
                f"firm_hq_county.csv missing {missing}. Columns: {list(hq.columns)}"
            )
        # Rename into a compatible schema
        hq = hq.rename(columns={"state": "state_full"})
    else:
        raise FileNotFoundError(
            "Could not find HQ mapping. Expect either:\n"
            "  data/intermediate/hq_with_county.csv OR\n"
            "  data/intermediate/firm_hq_county.csv"
        )

    # Clean keys for merging: lowercase, strip, remove ' county'
    hq["ticker"] = hq["ticker"].astype(str)

    hq["state_key"] = (
        hq["state_full"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    hq["county_key"] = (
        hq["county_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" county", "", regex=False)
    )

    return hq[
        [
            "ticker",
            "firm",
            "state_full",
            "county_name",
            "county_fips",
            "lat",
            "lon",
            "state_key",
            "county_key",
        ]
    ]


def build_panel_firms_with_brightness():
    hq = _load_hq_mapping()

    lights_path = ROOT / "intermediate" / "lights_monthly_by_coord.csv"
    rets_path = ROOT / "raw" / "sp500_monthly_returns.csv"

    lights = pd.read_csv(lights_path)
    rets = pd.read_csv(rets_path)

    # ---------------- Lights: normalize + build merge keys ----------------
    lights.columns = [c.strip().lower() for c in lights.columns]

    # Expect: name_1 (state), name_2 (county), date, avg_rad_month
    if "date" not in lights.columns:
        raise ValueError(f"{lights_path} must contain a 'date' column.")
    if "avg_rad_month" not in lights.columns:
        raise ValueError(f"{lights_path} must contain 'avg_rad_month'.")
    if "name_1" not in lights.columns or "name_2" not in lights.columns:
        raise ValueError(
            f"{lights_path} must contain 'name_1' (state) and 'name_2' (county). "
            f"Found: {list(lights.columns)}"
        )

    lights["date"] = pd.to_datetime(lights["date"], errors="coerce")
    lights = lights.dropna(subset=["date"])

    lights["state_key"] = (
        lights["name_1"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    lights["county_key"] = (
        lights["name_2"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" county", "", regex=False)
    )

    # ---------------- Returns: normalize ----------------
    rets.columns = [c.strip().lower() for c in rets.columns]

    if "date" not in rets.columns:
        raise ValueError(f"{rets_path} must contain a 'date' column.")
    if "ticker" not in rets.columns:
        raise ValueError(f"{rets_path} must contain a 'ticker' column.")

    if "ret" not in rets.columns:
        if "return" in rets.columns:
            rets = rets.rename(columns={"return": "ret"})
        else:
            raise ValueError(
                f"{rets_path} must contain either 'ret' or 'return'. "
                f"Found: {list(rets.columns)}"
            )

    rets["date"] = pd.to_datetime(rets["date"], errors="coerce")
    rets = rets.dropna(subset=["date"])
    rets["ticker"] = rets["ticker"].astype(str)

    # ---------------- Merge HQ with lights by state+county ----------------
    print("🔗 Merging HQ mapping with lights on (state_key, county_key)...")
    panel = hq.merge(
        lights[["state_key", "county_key", "date", "avg_rad_month"]],
        on=["state_key", "county_key"],
        how="left",
    )

    # At this point we have one row per (ticker, county, date)
    # but some firms may not match if naming is off.
    merged_count = panel["avg_rad_month"].notna().sum()
    print(f"  → Non-missing brightness rows: {merged_count} / {len(panel)}")

    # ---------------- Merge in returns on (ticker, date) ----------------
    panel = panel.merge(
        rets[["ticker", "date", "ret"]],
        on=["ticker", "date"],
        how="left",
    )

    # ---------------- Sort + compute forward returns & brightness change ----------------
    panel = panel.sort_values(["ticker", "date"])

    # Next-month return per ticker
    panel["ret_fwd"] = panel.groupby("ticker")["ret"].shift(-1)

    # Change in brightness at the county level over time
    # (using county_key as the panel dimension)
    panel["brightness_change"] = panel.groupby(
        ["state_key", "county_key"]
    )["avg_rad_month"].diff()

    # ---------------- Save final panel ----------------
    out_path = ROOT / "final" / "nightlights_model_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"✅ Saved final dataset to {out_path}")
    print("Preview:")
    print(
        panel[
            [
                "ticker",
                "firm",
                "state_full",
                "county_name",
                "date",
                "avg_rad_month",
                "ret",
                "ret_fwd",
                "brightness_change",
            ]
        ]
        .head(10)
    )

    return panel


if __name__ == "__main__":
    build_panel_firms_with_brightness()
