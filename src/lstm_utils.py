"""
LSTM utilities for TSLA forecasting.
Handles scaling, sequence creation, model building/training, and forecasting.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

PROCESSED_DIR = "../data/processed"


def fit_scaler(train_price):
    """Fit MinMaxScaler on TRAINING prices only (avoids data leakage)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_price.values.reshape(-1, 1))
    return scaler, train_scaled


def create_sequences(data, window_size):
    """Convert a 1-D scaled array into (X, y) supervised pairs."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_test_sequences(train_scaled, test_scaled, window_size):
    """Use the tail of training data as context for the first test predictions."""
    last_train_window = train_scaled[-window_size:]
    combined = np.concatenate([last_train_window, test_scaled], axis=0)
    return create_sequences(combined, window_size)


def build_lstm_model(window_size, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, activation="tanh", input_shape=(window_size, 1)),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=32, patience=10):
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.1, callbacks=[early_stop], verbose=1,
    )
    return history


def forecast_with_confidence(model, X_test, scaler, last_train_price, daily_std, z=1.96):
    """Predict scaled prices, inverse-transform, build sqrt(t) confidence cone."""
    y_pred_scaled = model.predict(X_test, verbose=0)
    fc_prices = scaler.inverse_transform(y_pred_scaled).flatten()

    days_ahead = np.arange(1, len(fc_prices) + 1)
    band = last_train_price * daily_std * np.sqrt(days_ahead)
    fc_upper = fc_prices + z * band
    fc_lower = fc_prices - z * band

    return fc_prices, fc_lower, fc_upper


def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def save_results(fc_series, lower, upper, metrics, filename="tsla_lstm_forecast.csv"):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = f"{PROCESSED_DIR}/{filename}"
    out = pd.DataFrame({
        "Date": fc_series.index,
        "LSTM_Forecast_Price": fc_series.values,
        "Lower_CI": lower,
        "Upper_CI": upper,
    })
    out.to_csv(path, index=False)
    assert os.path.exists(path), f"Save failed: {path}"
    print(f"  Saved -> {path}  ({len(out)} rows)")

    metrics_path = f"{PROCESSED_DIR}/tsla_lstm_metrics.csv"
    pd.Series(metrics).to_csv(metrics_path, header=["value"])
    print(f"  Saved -> {metrics_path}")
    return path