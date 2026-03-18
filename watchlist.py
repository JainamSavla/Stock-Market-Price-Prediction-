"""
Watchlist management module for the stock prediction dashboard.
Handles saving, loading, and managing the user's watchlist.
"""

from typing import List, Dict, Any, Optional

import pandas as pd

from utils import WATCHLIST_FILE, read_json_file, write_json_file, ensure_cache_dirs
from data_loader import get_latest_price, validate_ticker
from features import get_current_rsi, engineer_all_features
from model import check_models_trained
from predictor import get_quick_prediction
from utils import load_training_data, get_signal_from_probability


def get_watchlist() -> List[str]:
    """
    Load the watchlist from disk.

    Returns:
        List of ticker symbols in the watchlist
    """
    data = read_json_file(WATCHLIST_FILE)
    return data.get("tickers", [])


def save_watchlist(tickers: List[str]) -> bool:
    """
    Save the watchlist to disk.

    Args:
        tickers: List of ticker symbols

    Returns:
        True if successful, False otherwise
    """
    ensure_cache_dirs()
    return write_json_file(WATCHLIST_FILE, {"tickers": tickers})


def add_to_watchlist(ticker: str) -> tuple[bool, str]:
    """
    Add a ticker to the watchlist.

    Args:
        ticker: Stock ticker symbol to add

    Returns:
        Tuple of (success, message)
    """
    ticker = ticker.strip().upper()

    if not ticker:
        return False, "Ticker symbol cannot be empty"

    # Validate the ticker first
    is_valid, error = validate_ticker(ticker)
    if not is_valid:
        return False, error

    # Load current watchlist
    watchlist = get_watchlist()

    # Check if already in watchlist
    if ticker in watchlist:
        return False, f"{ticker} is already in your watchlist"

    # Add to watchlist
    watchlist.append(ticker)

    # Save updated watchlist
    if save_watchlist(watchlist):
        return True, f"{ticker} added to watchlist"
    else:
        return False, "Failed to save watchlist"


def remove_from_watchlist(ticker: str) -> tuple[bool, str]:
    """
    Remove a ticker from the watchlist.

    Args:
        ticker: Stock ticker symbol to remove

    Returns:
        Tuple of (success, message)
    """
    ticker = ticker.strip().upper()

    # Load current watchlist
    watchlist = get_watchlist()

    # Check if in watchlist
    if ticker not in watchlist:
        return False, f"{ticker} is not in your watchlist"

    # Remove from watchlist
    watchlist.remove(ticker)

    # Save updated watchlist
    if save_watchlist(watchlist):
        return True, f"{ticker} removed from watchlist"
    else:
        return False, "Failed to save watchlist"


def clear_watchlist() -> bool:
    """
    Clear all tickers from the watchlist.

    Returns:
        True if successful, False otherwise
    """
    return save_watchlist([])


def is_in_watchlist(ticker: str) -> bool:
    """
    Check if a ticker is in the watchlist.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if in watchlist, False otherwise
    """
    ticker = ticker.strip().upper()
    watchlist = get_watchlist()
    return ticker in watchlist


def get_watchlist_count() -> int:
    """
    Get the number of tickers in the watchlist.

    Returns:
        Number of tickers
    """
    return len(get_watchlist())


def get_watchlist_data(
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Get detailed data for all watchlist tickers.

    Args:
        progress_callback: Optional callback function(current, total) for progress

    Returns:
        List of dictionaries with watchlist item data
    """
    watchlist = get_watchlist()
    data = []
    total = len(watchlist)

    for i, ticker in enumerate(watchlist):
        if progress_callback:
            progress_callback(i + 1, total)

        item = get_watchlist_item_data(ticker)
        if item:
            data.append(item)

    return data


def get_watchlist_item_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed data for a single watchlist ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with ticker data or None if fetch fails
    """
    try:
        # Get current price info
        price_info = get_latest_price(ticker)
        if price_info is None:
            return {
                "ticker": ticker,
                "current_price": None,
                "change_pct": None,
                "rsi": None,
                "probability_up": None,
                "signal": "N/A",
                "status": "Error fetching data"
            }

        # Try to get RSI from cached data
        rsi = None
        df = load_training_data(ticker)
        if df is not None:
            if 'rsi' not in df.columns:
                df = engineer_all_features(df)
            rsi = get_current_rsi(df)

        # Try to get prediction if model is trained
        probability_up = None
        signal = "N/A"
        if check_models_trained(ticker):
            prediction = get_quick_prediction(ticker)
            if prediction:
                probability_up = prediction['probability_up']
                signal = prediction['signal']

        return {
            "ticker": ticker,
            "current_price": price_info['current_price'],
            "change_pct": price_info['change_pct'],
            "rsi": rsi,
            "probability_up": probability_up,
            "signal": signal,
            "status": "OK"
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "current_price": None,
            "change_pct": None,
            "rsi": None,
            "probability_up": None,
            "signal": "N/A",
            "status": f"Error: {str(e)}"
        }


def watchlist_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert watchlist data to a pandas DataFrame for display.

    Args:
        data: List of watchlist item dictionaries

    Returns:
        Formatted DataFrame
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Format columns
    if 'current_price' in df.columns:
        df['current_price'] = df['current_price'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )

    if 'change_pct' in df.columns:
        df['change_pct'] = df['change_pct'].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
        )

    if 'rsi' in df.columns:
        df['rsi'] = df['rsi'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

    if 'probability_up' in df.columns:
        df['probability_up'] = df['probability_up'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )

    # Rename columns for display
    column_names = {
        'ticker': 'Ticker',
        'current_price': 'Price',
        'change_pct': 'Change',
        'rsi': 'RSI',
        'probability_up': 'Prob Up',
        'signal': 'Signal'
    }

    # Select and rename columns
    display_cols = ['ticker', 'current_price', 'change_pct', 'rsi', 'probability_up', 'signal']
    available_cols = [c for c in display_cols if c in df.columns]
    df = df[available_cols]
    df = df.rename(columns=column_names)

    return df


def get_watchlist_summary() -> Dict[str, Any]:
    """
    Get a summary of the watchlist.

    Returns:
        Dictionary with summary statistics
    """
    watchlist = get_watchlist()

    if not watchlist:
        return {
            "total_tickers": 0,
            "with_models": 0,
            "without_models": 0
        }

    with_models = sum(1 for t in watchlist if check_models_trained(t))

    return {
        "total_tickers": len(watchlist),
        "with_models": with_models,
        "without_models": len(watchlist) - with_models
    }
