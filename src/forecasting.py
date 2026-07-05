import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def train_arima(series: pd.Series, seasonal: bool = False, m: int = 1):
    model = auto_arima(series.dropna(), seasonal=seasonal, m=m, suppress_warnings=True, stepwise=True)
    return model


def fit_sarimax(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def arima_forecast(model, periods: int):
    return model.predict(n_periods=periods)


def train_lstm(series: pd.Series, window_size: int = 60, epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001):
    scaler = MinMaxScaler()
    values = series.dropna().values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    x, y = [], []
    for i in range(window_size, len(scaled)):
        x.append(scaled[i - window_size : i, 0])
        y.append(scaled[i, 0])
    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation="tanh", input_shape=(x.shape[1], 1)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler


def prepare_lstm_sequences(series: pd.Series, scaler: MinMaxScaler, window_size: int = 60):
    values = series.dropna().values.reshape(-1, 1)
    scaled = scaler.transform(values)
    x = []
    for i in range(window_size, len(scaled)):
        x.append(scaled[i - window_size : i, 0])
    x = np.array(x)
    return x.reshape((x.shape[0], x.shape[1], 1))


def lstm_predict(model, scaler: MinMaxScaler, series: pd.Series, window_size: int = 60):
    x = prepare_lstm_sequences(series, scaler, window_size)
    preds = model.predict(x, verbose=0)
    return scaler.inverse_transform(preds).flatten()


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
