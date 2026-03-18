"""
Feature engineering module for the stock prediction dashboard.
Creates technical indicators and features for machine learning models.
Uses the 'ta' library for RSI and MACD calculations.
"""

from typing import Tuple, Optional

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

from utils import get_feature_columns


def compute_returns(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """
    Compute return features for various time horizons.

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column

    Returns:
        DataFrame with return columns added
    """
    df = df.copy()

    # 5-day return
    df['return_5d'] = df[close_col].pct_change(periods=5)

    # 10-day return
    df['return_10d'] = df[close_col].pct_change(periods=10)

    # 20-day return
    df['return_20d'] = df[close_col].pct_change(periods=20)

    return df


def compute_moving_averages(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """
    Compute moving average indicators.

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column

    Returns:
        DataFrame with moving average columns added
    """
    df = df.copy()

    # Moving averages
    df['ma_20'] = df[close_col].rolling(window=20).mean()
    df['ma_50'] = df[close_col].rolling(window=50).mean()
    df['ma_200'] = df[close_col].rolling(window=200).mean()

    return df


def compute_rsi(df: pd.DataFrame, close_col: str = 'Close', window: int = 14) -> pd.DataFrame:
    """
    Compute RSI using the ta library.

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column
        window: RSI calculation window (default 14)

    Returns:
        DataFrame with RSI column added
    """
    df = df.copy()

    rsi_indicator = RSIIndicator(close=df[close_col], window=window)
    df['rsi'] = rsi_indicator.rsi()

    return df


def compute_macd(
    df: pd.DataFrame,
    close_col: str = 'Close',
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9
) -> pd.DataFrame:
    """
    Compute MACD indicators using the ta library.

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column
        window_slow: Slow EMA window (default 26)
        window_fast: Fast EMA window (default 12)
        window_sign: Signal line window (default 9)

    Returns:
        DataFrame with MACD columns added
    """
    df = df.copy()

    macd_indicator = MACD(
        close=df[close_col],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign
    )

    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    return df


def compute_volatility(df: pd.DataFrame, close_col: str = 'Close', window: int = 20) -> pd.DataFrame:
    """
    Compute volatility indicator (rolling standard deviation).

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column
        window: Rolling window for std calculation (default 20)

    Returns:
        DataFrame with volatility column added
    """
    df = df.copy()

    # Calculate rolling standard deviation of returns
    returns = df[close_col].pct_change()
    df['volatility_20d'] = returns.rolling(window=window).std()

    return df


def compute_volume_features(df: pd.DataFrame, volume_col: str = 'Volume') -> pd.DataFrame:
    """
    Compute volume-related features.

    Args:
        df: DataFrame with volume data
        volume_col: Name of the volume column

    Returns:
        DataFrame with volume feature columns added
    """
    df = df.copy()

    # Volume change (day-over-day)
    df['volume_change'] = df[volume_col].pct_change()

    # 20-day volume moving average
    df['volume_ma_20'] = df[volume_col].rolling(window=20).mean()

    return df


def compute_price_distance(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """
    Compute price distance from moving averages (as percentages).

    Args:
        df: DataFrame with price and MA data (must have ma_20 and ma_50 columns)
        close_col: Name of the close price column

    Returns:
        DataFrame with price distance columns added
    """
    df = df.copy()

    # Distance from MA20 as percentage
    df['price_dist_ma20'] = (df[close_col] - df['ma_20']) / df['ma_20'] * 100

    # Distance from MA50 as percentage
    df['price_dist_ma50'] = (df[close_col] - df['ma_50']) / df['ma_50'] * 100

    return df


def compute_target_variables(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """
    Compute target variables for model training.

    Targets:
    - next_day_return: Return from t to t+1
    - week_return: Return from t to t+5
    - month_return: Return from t to t+20

    Binary labels: 1 = positive return, 0 = negative return

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column

    Returns:
        DataFrame with target columns added
    """
    df = df.copy()

    # Calculate future returns (shift by negative to get future values)
    df['next_day_return'] = df[close_col].shift(-1) / df[close_col] - 1
    df['week_return'] = df[close_col].shift(-5) / df[close_col] - 1
    df['month_return'] = df[close_col].shift(-20) / df[close_col] - 1

    # Convert to binary classification labels (1 = up, 0 = down)
    df['target_next_day'] = (df['next_day_return'] > 0).astype(int)
    df['target_1_week'] = (df['week_return'] > 0).astype(int)
    df['target_1_month'] = (df['month_return'] > 0).astype(int)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a DataFrame.

    Args:
        df: Raw OHLCV DataFrame from yfinance

    Returns:
        DataFrame with all features computed
    """
    df = df.copy()

    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply all feature computations in order
    df = compute_returns(df)
    df = compute_moving_averages(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_volatility(df)
    df = compute_volume_features(df)
    df = compute_price_distance(df)
    df = compute_target_variables(df)

    return df


def prepare_features_for_training(
    df: pd.DataFrame,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature matrix X and target matrix y for training.

    Args:
        df: DataFrame with all features computed
        drop_na: Whether to drop rows with NaN values

    Returns:
        Tuple of (X, y) DataFrames ready for training
    """
    feature_cols = get_feature_columns()
    target_cols = ['target_next_day', 'target_1_week', 'target_1_month']

    # Check that all required columns exist
    all_cols = feature_cols + target_cols
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    # Select feature and target columns
    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    if drop_na:
        # Find rows where either X or y has NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]

    return X, y


def get_latest_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract the most recent feature row for making predictions.

    Args:
        df: DataFrame with all features computed

    Returns:
        Single-row DataFrame with latest features, or None if invalid
    """
    feature_cols = get_feature_columns()

    if df is None or df.empty:
        return None

    # Get the last row
    latest = df[feature_cols].iloc[[-1]].copy()

    # Check for NaN values
    if latest.isna().any().any():
        # Try to get the last valid row
        valid_rows = df[feature_cols].dropna()
        if valid_rows.empty:
            return None
        latest = valid_rows.iloc[[-1]].copy()

    return latest


def get_historical_returns_on_positive_predictions(
    df: pd.DataFrame,
    horizon: str = 'next_day'
) -> float:
    """
    Calculate the average return when the model would have predicted UP.

    This is used to estimate expected move for predictions.

    Args:
        df: DataFrame with features and target columns
        horizon: 'next_day', '1_week', or '1_month'

    Returns:
        Average return as a decimal (e.g., 0.02 for 2%)
    """
    return_col_map = {
        'next_day': 'next_day_return',
        '1_week': 'week_return',
        '1_month': 'month_return'
    }

    target_col_map = {
        'next_day': 'target_next_day',
        '1_week': 'target_1_week',
        '1_month': 'target_1_month'
    }

    return_col = return_col_map.get(horizon, 'next_day_return')
    target_col = target_col_map.get(horizon, 'target_next_day')

    if return_col not in df.columns or target_col not in df.columns:
        return 0.01  # Default 1% expected move

    # Filter for days where target was positive (what model would predict as UP)
    positive_days = df[df[target_col] == 1]

    if positive_days.empty:
        return 0.01

    # Calculate average return on positive days
    avg_return = positive_days[return_col].mean()

    # Handle NaN
    if pd.isna(avg_return):
        return 0.01

    return float(avg_return)


def get_volume_ratio(df: pd.DataFrame) -> float:
    """
    Calculate current volume relative to 20-day average.

    Args:
        df: DataFrame with volume data

    Returns:
        Volume ratio (e.g., 1.5 means 50% above average)
    """
    if 'Volume' not in df.columns or 'volume_ma_20' not in df.columns:
        return 1.0

    current_volume = df['Volume'].iloc[-1]
    avg_volume = df['volume_ma_20'].iloc[-1]

    if pd.isna(avg_volume) or avg_volume == 0:
        return 1.0

    return float(current_volume / avg_volume)


def get_current_rsi(df: pd.DataFrame) -> float:
    """
    Get the most recent RSI value.

    Args:
        df: DataFrame with RSI computed

    Returns:
        Current RSI value or 50 if not available
    """
    if 'rsi' not in df.columns:
        return 50.0

    rsi = df['rsi'].iloc[-1]

    if pd.isna(rsi):
        return 50.0

    return float(rsi)
