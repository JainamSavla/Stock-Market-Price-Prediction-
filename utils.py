"""
Utility functions for the stock prediction dashboard.
Handles cache paths, file I/O operations, and shared helper functions.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import joblib


# Base cache directory - all cached data stored here
CACHE_DIR = Path(__file__).parent / "cache"
MODELS_DIR = CACHE_DIR / "models"
DATA_DIR = CACHE_DIR / "data"
HISTORY_FILE = CACHE_DIR / "history.json"
WATCHLIST_FILE = CACHE_DIR / "watchlist.json"


def ensure_cache_dirs() -> None:
    """Create all required cache directories if they don't exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def get_model_dir(ticker: str) -> Path:
    """Get the directory path for a ticker's saved models."""
    # Sanitize ticker symbol for filesystem (replace dots with underscores)
    safe_ticker = ticker.replace(".", "_").replace("/", "_")
    return MODELS_DIR / safe_ticker


def get_data_path(ticker: str) -> Path:
    """Get the parquet file path for a ticker's saved training data."""
    safe_ticker = ticker.replace(".", "_").replace("/", "_")
    return DATA_DIR / f"{safe_ticker}.parquet"


def save_model(model: Any, ticker: str, model_name: str) -> bool:
    """
    Save a trained model to disk using joblib.

    Args:
        model: The trained model object
        ticker: Stock ticker symbol
        model_name: Name of the model (e.g., 'model_next_day')

    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_cache_dirs()
        model_dir = get_model_dir(ticker)
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        return True
    except Exception as e:
        print(f"Error saving model {model_name} for {ticker}: {e}")
        return False


def load_model(ticker: str, model_name: str) -> Optional[Any]:
    """
    Load a saved model from disk.

    Args:
        ticker: Stock ticker symbol
        model_name: Name of the model to load

    Returns:
        The loaded model or None if not found
    """
    try:
        model_path = get_model_dir(ticker) / f"{model_name}.joblib"
        if model_path.exists():
            return joblib.load(model_path)
        return None
    except Exception as e:
        print(f"Error loading model {model_name} for {ticker}: {e}")
        return None


def model_exists(ticker: str) -> bool:
    """Check if models exist for a given ticker."""
    model_dir = get_model_dir(ticker)
    if not model_dir.exists():
        return False
    # Check for at least one model file
    model_files = list(model_dir.glob("*.joblib"))
    return len(model_files) >= 3


def save_training_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Save training data to parquet format.

    Args:
        df: DataFrame containing training data
        ticker: Stock ticker symbol

    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_cache_dirs()
        data_path = get_data_path(ticker)
        df.to_parquet(data_path, index=True)
        return True
    except Exception as e:
        print(f"Error saving training data for {ticker}: {e}")
        return False


def load_training_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load saved training data from parquet.

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame or None if not found
    """
    try:
        data_path = get_data_path(ticker)
        if data_path.exists():
            return pd.read_parquet(data_path)
        return None
    except Exception as e:
        print(f"Error loading training data for {ticker}: {e}")
        return None


def training_data_exists(ticker: str) -> bool:
    """Check if training data exists for a given ticker."""
    return get_data_path(ticker).exists()


def get_last_data_date(ticker: str) -> Optional[datetime]:
    """
    Get the last date in saved training data.

    Args:
        ticker: Stock ticker symbol

    Returns:
        The last date in the data or None if data doesn't exist
    """
    df = load_training_data(ticker)
    if df is not None and len(df) > 0:
        # Ensure index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.max().to_pydatetime()
        elif 'Date' in df.columns:
            return pd.to_datetime(df['Date']).max().to_pydatetime()
    return None


def read_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Read a JSON file and return its contents.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the file contents, or empty dict if file doesn't exist
    """
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return {}


def write_json_file(file_path: Path, data: Dict[str, Any]) -> bool:
    """
    Write data to a JSON file.

    Args:
        file_path: Path to the JSON file
        data: Dictionary to write

    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_cache_dirs()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except IOError as e:
        print(f"Error writing {file_path}: {e}")
        return False


def construct_ticker(symbol: str, exchange: str) -> str:
    """
    Construct a full ticker symbol with exchange suffix.

    Args:
        symbol: Base stock symbol (e.g., 'RELIANCE')
        exchange: Exchange name ('NSE' or 'BSE')

    Returns:
        Full ticker symbol (e.g., 'RELIANCE.NS')
    """
    symbol = symbol.strip().upper()

    # If symbol already has a suffix, return as-is
    if '.' in symbol:
        return symbol

    if exchange == "BSE":
        return f"{symbol}.BO"
    else:  # Default to NSE
        return f"{symbol}.NS"


def get_signal_from_probability(probability: float) -> str:
    """
    Convert prediction probability to trading signal.

    Args:
        probability: Probability of price going up (0-1)

    Returns:
        'BUY', 'SELL', or 'HOLD'
    """
    if probability > 0.6:
        return "BUY"
    elif probability < 0.4:
        return "SELL"
    else:
        return "HOLD"


def get_signal_color(signal: str) -> str:
    """Get color code for a trading signal."""
    colors = {
        "BUY": "#00C853",   # Green
        "SELL": "#FF1744",  # Red
        "HOLD": "#FFD600"   # Yellow
    }
    return colors.get(signal, "#9E9E9E")


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as a percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:+.{decimals}f}%"


def format_currency(value: float, symbol: str = "₹") -> str:
    """Format a number as currency (default: Indian Rupee)."""
    if pd.isna(value):
        return "N/A"
    return f"{symbol}{value:,.2f}"


def format_timestamp(dt: datetime) -> str:
    """Format a datetime object as a readable string."""
    return dt.strftime("%Y-%m-%d %H:%M")


def sanitize_ticker_for_display(ticker: str) -> str:
    """Clean up ticker symbol for display purposes."""
    return ticker.upper().strip()


def calculate_confidence_score(probability: float) -> float:
    """
    Calculate a confidence score based on how far prediction is from 0.5.

    Args:
        probability: Model's prediction probability

    Returns:
        Confidence score from 0-100
    """
    # Distance from neutral (0.5)
    distance = abs(probability - 0.5)
    # Scale to 0-100 (max distance is 0.5)
    confidence = (distance / 0.5) * 100
    return round(confidence, 1)


def get_feature_columns() -> List[str]:
    """Return the list of feature column names used for training."""
    return [
        'return_5d',
        'return_10d',
        'return_20d',
        'ma_20',
        'ma_50',
        'ma_200',
        'rsi',
        'macd',
        'macd_signal',
        'macd_hist',
        'volatility_20d',
        'volume_change',
        'volume_ma_20',
        'price_dist_ma20',
        'price_dist_ma50'
    ]


def validate_dataframe_for_training(df: pd.DataFrame, min_rows: int = 100) -> tuple[bool, str]:
    """
    Validate that a DataFrame has enough data for training.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "No data available"

    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows (minimum {min_rows} required)"

    # Check for required columns
    required_cols = get_feature_columns()
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    return True, ""
