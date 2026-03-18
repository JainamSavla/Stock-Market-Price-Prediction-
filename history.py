"""
Search history management for the stock prediction dashboard.
Handles saving, loading, and updating search history entries.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from utils import HISTORY_FILE, read_json_file, write_json_file, ensure_cache_dirs


# Maximum number of entries to keep in history
MAX_HISTORY_ENTRIES = 20


def get_history() -> List[Dict[str, Any]]:
    """
    Load search history from disk.

    Returns:
        List of history entries, sorted by timestamp (most recent first)
    """
    data = read_json_file(HISTORY_FILE)
    entries = data.get("entries", [])
    # Sort by timestamp descending (most recent first)
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries


def add_to_history(
    ticker: str,
    predicted_price: float,
    signal: str,
    probability_up: float,
    current_price: float
) -> bool:
    """
    Add or update a search entry in history.

    If the ticker already exists, updates its entry with new data.
    If it's a new ticker, adds it to history (removing oldest if at limit).

    Args:
        ticker: Stock ticker symbol
        predicted_price: Model's predicted price
        signal: Trading signal (BUY/SELL/HOLD)
        probability_up: Probability the price will go up
        current_price: Current stock price

    Returns:
        True if successful, False otherwise
    """
    ensure_cache_dirs()

    # Load existing history
    data = read_json_file(HISTORY_FILE)
    entries = data.get("entries", [])

    # Create new entry
    new_entry = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "predicted_price": predicted_price,
        "signal": signal,
        "probability_up": probability_up,
        "current_price": current_price
    }

    # Check if ticker already exists in history
    existing_idx = None
    for idx, entry in enumerate(entries):
        if entry.get("ticker") == ticker:
            existing_idx = idx
            break

    if existing_idx is not None:
        # Update existing entry
        entries[existing_idx] = new_entry
    else:
        # Add new entry at the beginning
        entries.insert(0, new_entry)

    # Keep only the most recent entries
    if len(entries) > MAX_HISTORY_ENTRIES:
        entries = entries[:MAX_HISTORY_ENTRIES]

    # Sort by timestamp descending
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Save back to file
    return write_json_file(HISTORY_FILE, {"entries": entries})


def remove_from_history(ticker: str) -> bool:
    """
    Remove a ticker from search history.

    Args:
        ticker: Stock ticker symbol to remove

    Returns:
        True if successful, False otherwise
    """
    data = read_json_file(HISTORY_FILE)
    entries = data.get("entries", [])

    # Filter out the ticker
    entries = [e for e in entries if e.get("ticker") != ticker]

    return write_json_file(HISTORY_FILE, {"entries": entries})


def clear_history() -> bool:
    """
    Clear all search history.

    Returns:
        True if successful, False otherwise
    """
    return write_json_file(HISTORY_FILE, {"entries": []})


def get_history_entry(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific ticker's history entry.

    Args:
        ticker: Stock ticker symbol

    Returns:
        History entry dict or None if not found
    """
    entries = get_history()
    for entry in entries:
        if entry.get("ticker") == ticker:
            return entry
    return None


def ticker_in_history(ticker: str) -> bool:
    """
    Check if a ticker exists in search history.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if ticker is in history, False otherwise
    """
    return get_history_entry(ticker) is not None


def get_recent_tickers(limit: int = 20) -> List[str]:
    """
    Get a list of recently searched tickers.

    Args:
        limit: Maximum number of tickers to return

    Returns:
        List of ticker symbols
    """
    entries = get_history()
    return [e.get("ticker") for e in entries[:limit] if e.get("ticker")]


def update_history_prediction(
    ticker: str,
    predicted_price: float,
    signal: str,
    probability_up: float
) -> bool:
    """
    Update only the prediction fields for an existing history entry.

    Args:
        ticker: Stock ticker symbol
        predicted_price: New predicted price
        signal: New trading signal
        probability_up: New probability up value

    Returns:
        True if successful, False otherwise
    """
    data = read_json_file(HISTORY_FILE)
    entries = data.get("entries", [])

    for entry in entries:
        if entry.get("ticker") == ticker:
            entry["predicted_price"] = predicted_price
            entry["signal"] = signal
            entry["probability_up"] = probability_up
            entry["timestamp"] = datetime.now().isoformat()
            break

    return write_json_file(HISTORY_FILE, {"entries": entries})


def format_history_for_display(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a history entry for display in the UI.

    Args:
        entry: Raw history entry dictionary

    Returns:
        Formatted dictionary suitable for display
    """
    timestamp = entry.get("timestamp", "")
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            formatted_time = timestamp
    else:
        formatted_time = "Unknown"

    return {
        "ticker": entry.get("ticker", "Unknown"),
        "timestamp": formatted_time,
        "predicted_price": f"₹{entry.get('predicted_price', 0):.2f}",
        "signal": entry.get("signal", "N/A"),
        "probability_up": f"{entry.get('probability_up', 0) * 100:.1f}%"
    }


def get_formatted_history() -> List[Dict[str, str]]:
    """
    Get all history entries formatted for display.

    Returns:
        List of formatted history entry dictionaries
    """
    entries = get_history()
    return [format_history_for_display(e) for e in entries]
