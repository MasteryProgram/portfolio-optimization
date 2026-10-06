"""
Task 3 — Future forecasting utilities for TSLA.

The best Task-2 model was the LSTM (lowest test MAPE). For genuine multi-step
*future* forecasting we make one important change versus Task 2: the LSTM is
trained on **log-returns** rather than absolute prices.

Why: recursively feeding an absolute-price LSTM's own outputs back into itself
diverges — a small directional bias compounds until the MinMax-scaled path runs
off the trained range and reconstructs to negative (impossible) prices. Log
returns are stationary and bounded, so the recursion is stable, and prices are
rebuilt as  P_t = P_last * exp(cumsum(returns))  which is positive by
construction. This is still an LSTM, still recursive multi-step — just in the
representation where the recursion behaves.
"""
import os
import numpy as np
import pandas as pd

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"


def load_price(ticker):
    """Load cached adjusted Close price series from data/raw/{ticker}_raw.csv."""
    path = f"{RAW_DIR}/{ticker}_raw.csv"
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    price = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    price.name = ticker
    return price


def log_returns(price):
    """Daily log-returns: log(P_t / P_{t-1}). Stationary, additive over time."""
    r = np.log(price / price.shift(1)).dropna()
    r.name = "log_return"
    return r


def future_business_index(last_date, n_steps):
    """Generate n_steps future business-day timestamps starting after last_date."""
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)


def recursive_return_forecast(model, scaler, return_series, window_size, n_steps):
    """Multi-step recursive forecast in LOG-RETURN space.

    Each predicted return is appended to the input window and reused for the
    next step — the genuine multi-step scheme the task requires. Because returns
    are stationary and bounded, the recursion does not diverge.
    """
    scaled = scaler.transform(return_series.values.reshape(-1, 1)).flatten()
    window = list(scaled[-window_size:])
    preds_scaled = []
    for _ in range(n_steps):
        x = np.array(window[-window_size:]).reshape(1, window_size, 1)
        yhat = float(model.predict(x, verbose=0)[0, 0])
        preds_scaled.append(yhat)
        window.append(yhat)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds  # forecasted daily log-returns


def returns_to_prices(forecast_log_returns, last_price):
    """Reconstruct a positive price path: P_t = P_last * exp(cumsum(r))."""
    return last_price * np.exp(np.cumsum(forecast_log_returns))


def lognormal_cone(forecast_prices, daily_std, z=1.96):
    """Positivity-preserving sqrt(t) cone (lognormal / geometric random walk).

    band(t) = exp(± z * daily_std * sqrt(t)) multiplies the central path, so the
    lower bound approaches zero but never crosses it — the correct shape for a
    price, unlike an additive cone which can go negative. `daily_std` is the
    std of daily log-returns. The cone still widens as sqrt(t).
    """
    days_ahead = np.arange(1, len(forecast_prices) + 1)
    sigma_t = daily_std * np.sqrt(days_ahead)
    lower = forecast_prices * np.exp(-z * sigma_t)
    upper = forecast_prices * np.exp(z * sigma_t)
    return lower, upper


def summarize_trend(price, forecast_series, lower, upper):
    """Compute the numbers behind the trend / risk narrative."""
    last_price = float(price.iloc[-1])
    end_price = float(forecast_series.iloc[-1])
    n = len(forecast_series)
    total_return = end_price / last_price - 1.0
    ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0
    direction = "UPWARD" if total_return > 0.02 else "DOWNWARD" if total_return < -0.02 else "STABLE"
    cone_start = float(upper[0] - lower[0])
    cone_end = float(upper[-1] - lower[-1])
    return {
        "last_price": last_price,
        "end_price": end_price,
        "horizon_days": n,
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "direction": direction,
        "cone_width_start": cone_start,
        "cone_width_end": cone_end,
        "cone_width_ratio": cone_end / cone_start if cone_start else np.nan,
    }


def save_future_forecast(forecast_series, forecast_log_returns, lower, upper,
                         filename="tsla_future_forecast.csv"):
    """Persist the future price path + cone AND the forecast returns.

    Task 4 reads `Forecast_Return` as TSLA's expected daily return (its analyst
    "view"). We store simple returns (exp(log_return) - 1) for that purpose.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    simple_returns = np.exp(forecast_log_returns) - 1.0
    out = pd.DataFrame({
        "Date": forecast_series.index,
        "Forecast_Price": forecast_series.values,
        "Lower_CI": lower,
        "Upper_CI": upper,
        "Forecast_Return": simple_returns,
    })
    path = f"{PROCESSED_DIR}/{filename}"
    out.to_csv(path, index=False)
    assert os.path.exists(path), f"Save failed: {path}"
    print(f"  Saved -> {path}  ({len(out)} rows)")
    return path
