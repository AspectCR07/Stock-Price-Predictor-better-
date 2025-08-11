"""
utils.py
Helper functions: train/test split (time-series aware), save/load model, evaluation.
"""

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def time_series_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Time-series split: first (1-test_size) fraction as train, last test_size as test.
    """
    n = len(df)
    split_idx = int(np.floor((1 - test_size) * n))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Model saved to {path}")

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def evaluate_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae}
