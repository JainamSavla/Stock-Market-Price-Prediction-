"""
Data loading module for the stock prediction dashboard.
Handles fetching stock data from yfinance, ticker validation, and incremental updates.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from utils import (
    load_training_data,
    save_training_data,
    training_data_exists,
    get_last_data_date
)


# Default start date for historical data
DEFAULT_START_DATE = "2015-01-01"


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate a ticker symbol by attempting a test download.

    Args:
        ticker: Stock ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to download last 5 days of data
        test_data = yf.download(
            ticker,
            period="5d",
            progress=False,
            timeout=10
        )

        if test_data.empty:
            return False, "Ticker not found. Please check the symbol and selected exchange."

        return True, ""

    except Exception as e:
        error_msg = str(e)
        if "No data found" in error_msg or "no price data" in error_msg.lower():
            return False, "Ticker not found. Please check the symbol and selected exchange."
        return False, f"Error validating ticker: {error_msg}"


def download_full_history(ticker: str, start_date: str = DEFAULT_START_DATE) -> Optional[pd.DataFrame]:
    """
    Download full historical data for a ticker from start_date to present.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for historical data (YYYY-MM-DD format)

    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
            timeout=30
        )

        if df.empty:
            return None

        # Ensure column names are flat (not MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone info if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None


def download_incremental_data(ticker: str, start_date: datetime) -> Optional[pd.DataFrame]:
    """
    Download data from a specific date to present (for incremental updates).

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for incremental download

    Returns:
        DataFrame with new data or None if no new data available
    """
    try:
        # Add one day to start_date to avoid overlap
        adjusted_start = start_date + timedelta(days=1)

        df = yf.download(
            ticker,
            start=adjusted_start.strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
            timeout=30
        )

        if df.empty:
            return None

        # Ensure column names are flat
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone info if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"Error downloading incremental data for {ticker}: {e}")
        return None


def merge_dataframes(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge old and new data, removing duplicates based on index (date).

    Args:
        old_df: Existing historical data
        new_df: New data to merge

    Returns:
        Combined DataFrame with duplicates removed
    """
    # Combine the dataframes
    combined = pd.concat([old_df, new_df])

    # Remove duplicates, keeping the last occurrence (newer data)
    combined = combined[~combined.index.duplicated(keep='last')]

    # Sort by date
    combined = combined.sort_index()

    return combined


def load_or_download_data(ticker: str) -> Tuple[Optional[pd.DataFrame], bool, str]:
    """
    Load data from cache or download new data as needed.

    Implements incremental learning: if cached data exists, downloads only new data
    and merges it with cached data.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Tuple of (DataFrame, is_new_data, status_message)
        - DataFrame: The loaded/downloaded data
        - is_new_data: True if new data was downloaded and needs retraining
        - status_message: Human-readable status message
    """
    if training_data_exists(ticker):
        # Load cached data
        cached_df = load_training_data(ticker)
        last_date = get_last_data_date(ticker)

        if cached_df is None or last_date is None:
            # Cache corrupted, download fresh
            df = download_full_history(ticker)
            if df is None:
                return None, False, "Failed to download data"
            return df, True, "Downloaded fresh data (cache was corrupted)"

        # Check if we need new data
        today = datetime.now().date()
        if last_date.date() >= today - timedelta(days=1):
            # Data is up to date
            return cached_df, False, "Loaded from cache (data is current)"

        # Download incremental data
        new_df = download_incremental_data(ticker, last_date)

        if new_df is None or new_df.empty:
            # No new data available
            return cached_df, False, "Loaded from cache (no new data available)"

        # Merge old and new data
        # We need to handle the raw OHLCV data, not the feature-engineered data
        # The cached_df might have features added, so we need to get just the raw columns
        raw_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_raw_cols = [c for c in raw_columns if c in cached_df.columns]

        if available_raw_cols:
            cached_raw = cached_df[available_raw_cols]
            merged_df = merge_dataframes(cached_raw, new_df)
        else:
            merged_df = merge_dataframes(cached_df, new_df)

        new_rows = len(merged_df) - len(cached_df)
        return merged_df, True, f"Merged {new_rows} new rows with cached data"

    else:
        # No cached data, download full history
        df = download_full_history(ticker)
        if df is None:
            return None, False, "Failed to download data"
        return df, True, "Downloaded full historical data"


def get_latest_price(ticker: str) -> Optional[dict]:
    """
    Get the latest price information for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with current price info or None if fails
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d", timeout=10)

        if hist.empty:
            return None

        # Flatten columns if MultiIndex
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        volume = hist['Volume'].iloc[-1]

        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

        return {
            "current_price": float(current_price),
            "previous_close": float(prev_close),
            "change": float(change),
            "change_pct": float(change_pct),
            "volume": int(volume)
        }

    except Exception as e:
        print(f"Error getting latest price for {ticker}: {e}")
        return None


def get_stock_info(ticker: str) -> Optional[dict]:
    """
    Get basic stock information.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with stock info or None if fails
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None)
        }

    except Exception as e:
        print(f"Error getting stock info for {ticker}: {e}")
        return None


def get_data_for_date_range(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Download data for a specific date range.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            timeout=30
        )

        if df.empty:
            return None

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone info if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None
