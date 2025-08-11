"""
data_fetcher.py
Fetches historical stock data from yfinance and saves to CSV (optional).
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker.
    Args:
        ticker: e.g. 'AAPL'
        period: e.g. '2y', '1mo', '5y'
        interval: '1d', '1wk', '1h', etc.
    Returns:
        DataFrame with Date index and columns ['Open','High','Low','Close','Adj Close','Volume']
    """
    print(f"[fetch_history] Downloading {ticker} {period} {interval} ...")
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}.")
    df = df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index.name = 'Date'
    df = df.sort_index()  # chronological
    print(f"[fetch_history] Downloaded {len(df)} rows.")
    return df

if __name__ == "__main__":
    # quick manual test
    df = fetch_history("AAPL", period="1y")
    print(df.tail())
