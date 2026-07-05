"""Data loading and return computation.

Design notes
------------
- We always model LOG returns r_t = ln(P_t / P_{t-1}): prices are
  non-stationary (near random walks); returns are approximately stationary.
  Every model downstream (Kalman, GARCH, bootstrap) assumes stationarity.
- Adjusted close is used so splits/dividends don't create fake jumps.
- Prices are cached to CSV so repeated runs don't hammer the API and so
  results are reproducible even if the vendor revises data.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_prices(
    tickers: list[str],
    start: str = "2021-06-01",
    end: str | None = None,
    cache_dir: str = "data_cache",
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance, with a local CSV cache.

    Run this on your own machine (the sandbox has no market-data access).
    Returns a DataFrame indexed by date, one column per ticker.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = "_".join(sorted(tickers)) + f"_{start}_{end}"
    cache_path = os.path.join(cache_dir, f"prices_{key}.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily: not needed for synthetic runs

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all")
    prices.to_csv(cache_path)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; drops the first (undefined) row.

    Note: we drop rows where ANY asset is missing so all series stay aligned
    on common trading days (needed for beta regressions and the portfolio).
    For QBTS this trims history to its listing date -- expected.
    """
    return np.log(prices / prices.shift(1)).dropna(how="any")


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Fixed-weight (daily-rebalanced) portfolio log-return approximation.

    For daily horizons, the cross-product terms are negligible, so the
    weighted sum of log returns is a standard, defensible approximation.
    """
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {w.sum():.4f}")
    return returns.mul(w, axis=1).sum(axis=1).rename("portfolio")
