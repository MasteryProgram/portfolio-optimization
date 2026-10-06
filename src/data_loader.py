import os
from pathlib import Path
import pandas as pd
import yfinance as yf

ASSET_TICKERS = ["TSLA", "BND", "SPY"]
PROCESSED_DIR = Path("..") / "data" / "processed"


def _ensure_processed_dir():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_asset_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical asset data from yfinance for a single ticker."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")
    df = df.reset_index()
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to `data/processed/` and return the path."""
    _ensure_processed_dir()
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    return path


def load_processed_data(filename: str) -> pd.DataFrame | None:
    """Load a processed CSV from `data/processed/` if it exists, else return None."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def load_or_fetch_assets(tickers: list[str], start_date: str, end_date: str, per_ticker: bool = True, force_fetch: bool = False) -> pd.DataFrame:
    """Load processed data from disk if present, otherwise fetch from yfinance and save.

    - If `per_ticker` is True, files are saved/loaded as `{TICKER}.csv` under `data/processed/`.
    - Otherwise a combined `assets_data.csv` file is used.
    """
    _ensure_processed_dir()
    frames = []
    if per_ticker:
        for ticker in tickers:
            filename = f"{ticker}.csv"
            df = None if force_fetch else load_processed_data(filename)
            if df is None:
                df = fetch_asset_data(ticker, start_date, end_date)
                save_processed_data(df, filename)
            else:
                # ensure dtypes and add missing Ticker column if needed
                if "Ticker" not in df.columns:
                    df["Ticker"] = ticker
                df["Date"] = pd.to_datetime(df["Date"])
            frames.append(df)
        result = pd.concat(frames, ignore_index=True)
        result = prepare_time_series(result)
        return result
    else:
        filename = "assets_data.csv"
        df = None if force_fetch else load_processed_data(filename)
        if df is None:
            frames = [fetch_asset_data(t, start_date, end_date) for t in tickers]
            result = pd.concat(frames, ignore_index=True)
            save_processed_data(result, filename)
        else:
            result = df
        result = prepare_time_series(result)
        return result


def fetch_assets_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical data for multiple tickers and return a combined DataFrame.

    This function always fetches from the API (no caching). Prefer `load_or_fetch_assets`.
    """
    frames = []
    for ticker in tickers:
        frame = fetch_asset_data(ticker, start_date, end_date)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"])
    return result


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a combined DataFrame for time series analysis.

    - Ensures `Date` is datetime
    - Sorts by `Ticker` and `Date`
    - Sets a datetime index named `Date` (keeps the `Date` column)
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df = df.set_index("Date", drop=False)
    return df


def clean_data(df: pd.DataFrame, interpolate: bool = True) -> pd.DataFrame:
    """Simple cleaning routine:

    - Ensures Date is datetime
    - Forward-fills then back-fills non-numeric gaps
    - Optionally interpolates numeric columns
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # fill small gaps
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # forward/backfill first
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    if interpolate:
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    return df


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize numeric columns to zero mean and unit variance."""
    result = df.copy()
    for col in columns:
        result[col] = (result[col] - result[col].mean()) / result[col].std(ddof=0)
    return result
