"""
Portfolio optimization utilities — Task 4.
Expected returns (TSLA forecasted via Task 3 LSTM, BND/SPY historical),
covariance matrix, Monte Carlo cloud for visualization, and PyPortfolioOpt
for exact optimal weights (unconstrained and capped).
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

PROCESSED_DIR = "data/processed"


def load_portfolio_data(tickers, start, end):
    """
    Load price history for all portfolio assets, reusing cached raw CSVs
    from data/raw/ (same cache used by fetch_and_save in arima_utils.py).
    Only downloads a ticker if its raw file doesn't already exist.
    """
    from src.arima_utils import fetch_and_save
    print('hi')
    price_dict = {}
    for ticker in tickers:
        price_dict[ticker] = fetch_and_save(ticker, start, end)

    prices = pd.DataFrame(price_dict).dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def load_tsla_forecast_return(path="../data/processed/tsla_future_forecast.csv"):
    """
    Load the Task 3 LSTM 12-month forecast and compute the implied
    annualized return from the log-return-based forecast.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run the Task 3 forecast notebook first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    first_price = df["Forecast_Price"].iloc[0]
    last_price = df["Forecast_Price"].iloc[-1]
    n_days = len(df)

    total_return = last_price / first_price - 1
    annualized_return = (1 + total_return) ** (252 / n_days) - 1

    return annualized_return, first_price, last_price


def build_expected_returns(hist_returns, tickers, tsla_annual_return):
    """Build mu_vec: TSLA = forecasted, others = historical annualized average."""
    hist_mu = hist_returns.mean() * 252
    mu_vec = hist_mu.copy()
    mu_vec["TSLA"] = tsla_annual_return
    return mu_vec[tickers]


def build_covariance_matrix(hist_returns):
    return hist_returns.cov() * 252


def monte_carlo_cloud(mu_vec, cov_matrix, tickers, n=5000, seed=42):
    """Generate random portfolios for visualization only (not the exact optimum)."""
    rng = np.random.default_rng(seed=seed)
    rets, vols, sharpes, weights = [], [], [], []

    for _ in range(n):
        w = rng.dirichlet(np.ones(len(tickers)))
        r = np.dot(w, mu_vec[tickers])
        v = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        s = r / v
        rets.append(r); vols.append(v); sharpes.append(s); weights.append(w)

    return {
        "returns": np.array(rets),
        "vols": np.array(vols),
        "sharpes": np.array(sharpes),
        "weights": weights,
    }


def solve_portfolio(mu_vec, cov_matrix, risk_free_rate=0.045, weight_bounds=(0, 1)):
    """
    Use PyPortfolioOpt to solve for exact Max Sharpe and Min Volatility
    portfolios, given a weight bound (default unconstrained: 0-100% per asset).
    """
    from pypfopt import EfficientFrontier

    ef_sharpe = EfficientFrontier(mu_vec, cov_matrix, weight_bounds=weight_bounds)
    ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    w_sharpe = ef_sharpe.clean_weights()
    perf_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)

    ef_minvol = EfficientFrontier(mu_vec, cov_matrix, weight_bounds=weight_bounds)
    ef_minvol.min_volatility()
    w_minvol = ef_minvol.clean_weights()
    perf_minvol = ef_minvol.portfolio_performance(risk_free_rate=risk_free_rate)

    return {
        "max_sharpe": {"weights": w_sharpe, "return": perf_sharpe[0],
                        "volatility": perf_sharpe[1], "sharpe": perf_sharpe[2]},
        "min_vol": {"weights": w_minvol, "return": perf_minvol[0],
                     "volatility": perf_minvol[1], "sharpe": perf_minvol[2]},
    }


def save_portfolio_results(results_dict, filename="portfolio_optimization_results.csv"):
    """results_dict: {'unconstrained': {...}, 'capped': {...}}, each with max_sharpe/min_vol."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = f"{PROCESSED_DIR}/{filename}"
    rows = []
    for scenario, results in results_dict.items():
        for label, data in results.items():
            row = {"scenario": scenario, "portfolio": label, "return": data["return"],
                   "volatility": data["volatility"], "sharpe": data["sharpe"]}
            row.update(data["weights"])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    assert os.path.exists(path)
    print(f"  Saved -> {path}")