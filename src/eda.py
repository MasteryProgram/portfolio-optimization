import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def compute_daily_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    df = df.copy()
    df["Daily Return"] = df.groupby("Ticker")[price_col].pct_change()
    return df


def compute_rolling_metrics(df: pd.DataFrame, window: int = 30, price_col: str = "Adj Close") -> pd.DataFrame:
    df = df.copy()
    df[f"Rolling Mean {window}"] = df.groupby("Ticker")[price_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df[f"Rolling Std {window}"] = df.groupby("Ticker")[price_col].transform(lambda x: x.rolling(window, min_periods=1).std())
    return df


def stationarity_test(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test and return results."""
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Used Lag": result[2],
        "Number of Observations": result[3],
        "Critical Values": result[4],
    }


def calculate_var(series: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate historical Value at Risk for a series of returns."""
    return np.percentile(series.dropna(), 100 * confidence_level)


def calculate_sharpe_ratio(series: pd.Series, risk_free_rate: float = 0.0) -> float:
    mean_ret = series.mean()
    std_ret = series.std(ddof=0)
    if std_ret == 0:
        return np.nan
    return (mean_ret - risk_free_rate) / std_ret


def plot_time_series(df: pd.DataFrame, ticker: str, price_col: str = "Adj Close") -> None:
    subset = df[df["Ticker"] == ticker]
    plt.figure(figsize=(12, 5))
    plt.plot(subset["Date"], subset[price_col], label=f"{ticker} {price_col}")
    plt.title(f"{ticker} {price_col} over time")
    plt.xlabel("Date")
    plt.ylabel(price_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(df: pd.DataFrame, ticker: str) -> None:
    subset = df[df["Ticker"] == ticker]
    plt.figure(figsize=(10, 4))
    sns.histplot(subset["Daily Return"].dropna(), bins=60, kde=True)
    plt.title(f"Daily Return Distribution for {ticker}")
    plt.xlabel("Daily Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, price_col: str = "Adj Close") -> None:
    pivot = df.pivot(index="Date", columns="Ticker", values=price_col).dropna()
    corr = pivot.pct_change().corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation of Daily Returns")
    plt.tight_layout()
    plt.show()
