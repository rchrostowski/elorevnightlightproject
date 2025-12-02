# scripts/fetch_monthly_returns.py

from pathlib import Path
import time

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SP500_PATH = PROJECT_ROOT / "data" / "raw" / "sp500_clean.csv"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "sp500_monthly_returns.csv"

START_DATE = "2018-01-01"  # match your nightlights window


def yahoo_symbol_from_ticker(ticker: str) -> str:
    """
    Convert your ticker to Yahoo's format.
    Example: 'BRK.B' -> 'BRK-B', 'BF.B' -> 'BF-B'
    """
    return ticker.replace(".", "-")


def fetch_monthly_returns_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Download monthly adjusted prices for one ticker from Yahoo,
    compute simple monthly returns, and return a DataFrame:
        ticker, date, return
    """
    y_symbol = yahoo_symbol_from_ticker(ticker)

    try:
        data = yf.download(
            y_symbol,
            start=START_DATE,
            interval="1mo",
            auto_adjust=True,   # use adjusted prices directly
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"⚠️ Error downloading {ticker} ({y_symbol}): {e}")
        return pd.DataFrame(columns=["ticker", "date", "return"])

    if data is None or data.empty:
        print(f"⚠️ No data for {ticker} ({y_symbol})")
        return pd.DataFrame(columns=["ticker", "date", "return"])

    # Some versions of yfinance/pandas can return multi-index or weird shapes.
    # We robustly extract a 1-D price series.
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        # Last resort: take the first numeric column
        numeric_cols = data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            print(f"⚠️ No suitable price column for {ticker} ({y_symbol})")
            return pd.DataFrame(columns=["ticker", "date", "return"])
        prices = data[numeric_cols[0]]

    # If prices is a DataFrame, squeeze to Series
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            # Take the first column as a fallback
            prices = prices.iloc[:, 0]

    # Drop any missing prices
    prices = prices.dropna()

    if prices.empty or len(prices) < 2:
        print(f"⚠️ Not enough price data for {ticker} ({y_symbol})")
        return pd.DataFrame(columns=["ticker", "date", "return"])

    # Monthly simple returns
    rets = prices.pct_change().dropna()

    if rets.empty:
        print(f"⚠️ No returns computed for {ticker} ({y_symbol})")
        return pd.DataFrame(columns=["ticker", "date", "return"])

    # Make sure index (dates) is 1-D and values are 1-D
    dates = pd.to_datetime(rets.index)
    values = rets.values.reshape(-1)  # force 1-D

    df = pd.DataFrame(
        {
            "ticker": [ticker] * len(values),
            "date": dates,
            "return": values,
        }
    )

    return df


def main():
    print(f"📥 Loading tickers from {SP500_PATH} ...")
    sp500 = pd.read_csv(SP500_PATH)

    if "ticker" not in sp500.columns:
        raise ValueError(
            f"Expected 'ticker' column in {SP500_PATH}. "
            f"Found columns: {sp500.columns.tolist()}"
        )

    tickers = sorted(sp500["ticker"].unique().tolist())
    print(f"✅ Found {len(tickers)} tickers.")

    all_rows = []
    for i, tkr in enumerate(tickers, start=1):
        print(f"📡 [{i}/{len(tickers)}] Fetching monthly returns for {tkr} ...")
        df_t = fetch_monthly_returns_for_ticker(tkr)

        if not df_t.empty:
            all_rows.append(df_t)

        # Small sleep to be gentle with Yahoo (you can tweak if needed)
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError("No returns downloaded. Something went wrong.")

    result = pd.concat(all_rows, ignore_index=True)

    # Ensure date is clean datetime and sorted
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"💾 Writing monthly returns to {OUT_PATH} ...")
    result.to_csv(OUT_PATH, index=False)
    print("✅ Done. Example rows:")
    print(result.head())


if __name__ == "__main__":
    main()
