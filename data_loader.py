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
    Uses direct zip download instead of pandas_datareader (which has bugs).
    Caches to disk. Returns empty DataFrame on failure.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"ff_factors_{start}_{end}.csv")

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        import io
        import zipfile
        from urllib.request import urlopen

        url = (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        )
        response = urlopen(url, timeout=30)
        zip_data = zipfile.ZipFile(io.BytesIO(response.read()))

        # Find the CSV file inside the zip
        csv_name = [n for n in zip_data.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with zip_data.open(csv_name) as f:
            raw = f.read().decode("utf-8")

        # Skip header lines (varies, find the first numeric date line)
        lines = raw.strip().split("\n")
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and len(stripped.split(",")[0].strip()) == 8:
                data_start = i
                break

        # Parse the data from that point
        data_lines = []
        for line in lines[data_start:]:
            parts = line.strip().split(",")
            if len(parts) >= 4 and parts[0].strip().isdigit():
                data_lines.append(line)
            elif len(parts) < 4 or not parts[0].strip():
                break  # End of daily data section

        csv_str = "Date,Mkt-RF,SMB,HML,RF\n" + "\n".join(data_lines)
        ff = pd.read_csv(io.StringIO(csv_str))
        ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
        ff = ff.set_index("Date")

        # Convert from percentage to decimal
        for col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0

        # Filter date range
        ff = ff.loc[start:end]
        ff.to_csv(cache_file)
        logger.info(f"Fama-French factors loaded: {len(ff)} days")
        return ff

    except Exception as e:
        logger.warning(f"Could not download Fama-French factors: {e}. Falling back.")
        return pd.DataFrame()
