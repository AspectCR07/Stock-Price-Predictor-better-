# Stock Price Predictor

A small end-to-end project to fetch historical stock prices, engineer features, train a regressor (RandomForest), and make next-day closing price predictions.

This project is intended for educational/demo purposes — it is **not** financial advice.

---

## Features
- Download historical OHLCV data via `yfinance`.
- Create lag and rolling-window features.
- Train a RandomForestRegressor with a sklearn Pipeline and GridSearch.
- Save trained model and produce an Actual vs Predicted plot.
- Simple prediction script to predict the next day close using the latest data.

---

## Folder structure
