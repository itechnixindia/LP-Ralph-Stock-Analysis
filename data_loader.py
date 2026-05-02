"""
Shared data loading utilities — single source of truth for price and factor data.
Eliminates the _load_prices duplication between agent1 and agent3.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = "cache/"


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance, caching to disk on first call.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_ticker = ticker.replace("^", "IDX_")
    cache_file = os.path.join(CACHE_DIR, f"prices_{safe_ticker}_{start}_{end}.csv")

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns (yfinance >= 0.2.x may return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.to_csv(cache_file)
    return df


def load_ff_factors(start: str, end: str) -> pd.DataFrame:
    """
    Download Fama-French 3-factor daily data from Kenneth French's website.
    Caches to disk. Returns empty DataFrame on failure.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"ff_factors_{start}_{end}.csv")

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        import pandas_datareader.data as web

        ff = web.DataReader(
            "F-F_Research_Data_Factors_daily", "famafrench", start=start, end=end
        )[0]
        ff = ff / 100.0
        ff.index = pd.to_datetime(ff.index)
        ff.to_csv(cache_file)
        return ff
    except Exception as e:
        logger.warning(f"Could not download Fama-French factors: {e}. Falling back.")
        return pd.DataFrame()
