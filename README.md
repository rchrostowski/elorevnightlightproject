# FIN 377 Nightlights Project 🌃

This repo links **VIIRS night-time lights** to **S&P 500 stocks** and tests whether
brightness changes predict next-month returns.

## Data

1. Download VIIRS lights CSV:

   - Source: your Dropbox link  
   - Save as: `data/raw/VIIRS-nighttime-lights-2013m1to2024m5-level2.csv`

2. Put your S&P 500 firm file (with lat/long) in:

   - `data/raw/sp500_clean.csv`

3. Put your monthly returns file (ticker / date / ret) in:

   - `data/raw/sp500_monthly_returns.csv`

## Build pipeline

```bash
python scripts/build_all.py

