"""
model_train.py
Full training script:
- fetch data
- build features
- train RandomForestRegressor in a pipeline
- evaluate on hold-out test set
- save trained model to disk
- plot predictions vs actual
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from features import create_features
from data_fetcher import fetch_history
from utils import time_series_split, save_model, evaluate_regression

plt.rcParams["figure.figsize"] = (10, 5)

def train(args):
    # 1) get data
    df = fetch_history(args.ticker, period=args.period)
    df = df[['Open','High','Low','Close','Volume']]

    # 2) features
    data = create_features(df, nlags=args.nlags)

    # 3) train/test split (time-aware)
    train_df, test_df = time_series_split(data, test_size=args.test_size)
    X_train = train_df.drop(columns=['Target'])
    y_train = train_df['Target']
    X_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']

    print(f"[train] Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    # 4) pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rfr', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # 5) hyperparameter search (small)
    param_grid = {
        'rfr__n_estimators': [100, 200],
        'rfr__max_depth': [6, 10, None],
    }

    search = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"[train] Best params: {search.best_params_}")
    best = search.best_estimator_

    # 6) evaluate
    preds_test = best.predict(X_test)
    evals = evaluate_regression(y_test, preds_test)
    print(f"[train] Test RMSE: {evals['rmse']:.4f}   MAE: {evals['mae']:.4f}")

    # 7) save model
    model_path = args.output if args.output else f"models/{args.ticker}_rfr.joblib"
    save_model(best, model_path)

    # 8) plot
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test.values, label='Actual Close', linewidth=1)
    ax.plot(y_test.index, preds_test, label='Predicted Close', linewidth=1)
    ax.set_title(f"{args.ticker} Actual vs Predicted Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plot_path = args.plot if args.plot else f"plots/{args.ticker}_pred_vs_actual.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    print(f"[train] Plot saved to {plot_path}")
    print("[train] Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock price predictor")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol (AAPL)')
    parser.add_argument('--period', type=str, default='2y', help='yfinance period (2y, 5y, 1y)')
    parser.add_argument('--nlags', type=int, default=5, help='Number of lag days to use as features')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for test')
    parser.add_argument('--output', type=str, default=None, help='Path to save trained model')
    parser.add_argument('--plot', type=str, default=None, help='Path to save prediction plot')
    args = parser.parse_args()
    train(args)
