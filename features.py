"""
features.py
Create features for ML model from OHLCV timeseries.
Common features: lag close prices, returns, moving averages, volatility.
"""

import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, nlags: int = 5) -> pd.DataFrame:
    """
    Given df with ['Open','High','Low','Close','Volume'] indexed by Date,
    create lag features and rolling statistics, and target column 'Target' = next-day Close.
    Args:
        df: raw price data
        nlags: number of lag days to include (lag1 ... lagn)
    Returns:
        DataFrame ready for ML with NaNs dropped
    """
    df = df.copy()
    # basic features
    df['Close_pct_change'] = df['Close'].pct_change()
    df['Return'] = df['Close'].pct_change()  # same as above, kept for clarity

    # lag features of close
    for lag in range(1, nlags+1):
        df[f'lag_close_{lag}'] = df['Close'].shift(lag)

    # rolling windows
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()

    df['std_5'] = df['Close'].rolling(window=5).std()
    df['std_10'] = df['Close'].rolling(window=10).std()

    # volume-based features
    df['vol_mean_5'] = df['Volume'].rolling(window=5).mean()
    df['vol_mean_10'] = df['Volume'].rolling(window=10).mean()

    # target: next day close (shift -1)
    df['Target'] = df['Close'].shift(-1)

    # drop rows with NaN (from shifts/rolling) and keep only features + target
    feature_cols = [c for c in df.columns if c not in ['Target']]
    out = df[feature_cols + ['Target']].dropna().copy()
    return out

if __name__ == "__main__":
    import yfinance as yf
    df = yf.Ticker("AAPL").history(period="1y")[['Open','High','Low','Close','Volume']]
    X = create_features(df, nlags=5)
    print(X.tail())
