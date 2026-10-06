"""
ARIMA utilities for TSLA forecasting.
Handles data fetch/save, stationarity testing, model fitting, and forecasting.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"


def fetch_and_save(ticker, start, end):
    """Fetch OHLCV data via yfinance, save raw CSV, return Close price Series."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_path = f"{RAW_DIR}/{ticker}_raw.csv"

    if os.path.exists(raw_path):
        print(f"  Found existing raw file: {raw_path}")
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    else:
        print(f"  Downloading {ticker} from yfinance...")
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True)

        # Flatten multi-index columns (yfinance returns e.g. ('Close','TSLA'))
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.to_csv(raw_path)
        print(f"  Saved raw data -> {raw_path}  ({len(raw)} rows)")

    assert os.path.exists(raw_path), f"Save failed: {raw_path} does not exist"

    # Force numeric — guards against any stray string contamination on reload
    price = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    price.name = "Close"
    return price


def save_series(series, filename, index_label="Date"):
    """Save a Series to data/processed/ and verify it landed on disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = f"{PROCESSED_DIR}/{filename}"
    series.to_csv(path, index_label=index_label)
    assert os.path.exists(path), f"Save failed: {path} does not exist"
    print(f"  Saved -> {path}  ({len(series)} rows)")
    return path


def adf_report(series, name):
    """Run ADF test, print verdict, return p-value."""
    res = adfuller(series.dropna())
    p = res[1]
    verdict = "STATIONARY" if p < 0.05 else "NON-STATIONARY"
    print(f"  {name:<20} ADF stat={res[0]:8.4f}  p={p:.6f}  ->  {verdict}")
    return p


def find_best_order(train_returns, seasonal=False, m=1):
    """Use auto_arima to find best (p,d,q). Falls back to (1,0,1) if pmdarima unavailable."""
    try:
        from pmdarima import auto_arima
        model = auto_arima(
            train_returns,
            start_p=0, start_q=0, max_p=5, max_q=5,
            d=0,                      # returns already stationary
            seasonal=seasonal, m=m,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", trace=False,
        )
        order = model.order
        print(f"  auto_arima selected order: {order}  (AIC={model.aic():.2f})")
        return order
    except ImportError:
        print("  pmdarima not installed — defaulting to ARIMA(1,0,1)")
        return (1, 0, 1)


def fit_arima(train_returns, order):
    model = ARIMA(train_returns, order=order)
    return model.fit()


def forecast_with_confidence(model_fit, train_returns, last_train_price, n_steps, z=1.96):
    """Forecast returns, convert to compounded prices, build sqrt(t) confidence cone."""
    fc_result = model_fit.get_forecast(steps=n_steps)
    fc_returns = fc_result.predicted_mean.values

    fc_prices_central = last_train_price * np.cumprod(1 + fc_returns)

    daily_std = float(train_returns.std())
    days_ahead = np.arange(1, n_steps + 1)
    band = last_train_price * daily_std * np.sqrt(days_ahead)

    fc_upper = fc_prices_central + z * band
    fc_lower = fc_prices_central - z * band

    return fc_returns, fc_prices_central, fc_lower, fc_upper


def evaluate_forecast(actual, forecast):
    """MAE, RMSE, MAPE between aligned actual and forecast price series."""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}