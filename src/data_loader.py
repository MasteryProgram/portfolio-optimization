import pandas as pd
import yfinance as yf

ASSET_TICKERS = ["TSLA", "BND", "SPY"]


def fetch_asset_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical asset data from yfinance for a single ticker."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")
    df = df.reset_index()
    df["Ticker"] = ticker
    return df


def fetch_assets_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical data for multiple tickers and return a combined DataFrame."""
    frames = []
    for ticker in tickers:
        frame = fetch_asset_data(ticker, start_date, end_date)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"])
    return result


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a combined DataFrame for time series analysis."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize numeric columns to zero mean and unit variance."""
    result = df.copy()
    for col in columns:
        result[col] = (result[col] - result[col].mean()) / result[col].std(ddof=0)
    return result
