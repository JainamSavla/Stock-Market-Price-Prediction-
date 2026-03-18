"""
Prediction module for the stock prediction dashboard.
Handles making predictions and computing expected price moves.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass

import pandas as pd

from model import get_model_for_horizon, predict_with_model, check_models_trained
from features import (
    engineer_all_features,
    get_latest_features,
    get_historical_returns_on_positive_predictions
)
from utils import (
    get_signal_from_probability,
    calculate_confidence_score,
    load_training_data
)
from data_loader import get_latest_price


@dataclass
class PredictionResult:
    """Container for prediction results."""
    ticker: str
    horizon: str
    current_price: float
    predicted_price: float
    expected_move_pct: float
    probability_up: float
    signal: str
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "horizon": self.horizon,
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "expected_move_pct": self.expected_move_pct,
            "probability_up": self.probability_up,
            "signal": self.signal,
            "confidence_score": self.confidence_score
        }


def make_prediction(
    ticker: str,
    horizon: str,
    df: pd.DataFrame,
    current_price: Optional[float] = None
) -> Optional[PredictionResult]:
    """
    Make a price prediction for a given ticker and horizon.

    Args:
        ticker: Stock ticker symbol
        horizon: 'next_day', '1_week', or '1_month'
        df: DataFrame with feature-engineered data
        current_price: Current stock price (will be fetched if not provided)

    Returns:
        PredictionResult object or None if prediction fails
    """
    # Check if models are trained
    if not check_models_trained(ticker):
        return None

    # Get the appropriate model
    model = get_model_for_horizon(ticker, horizon)
    if model is None:
        return None

    # Get current price if not provided
    if current_price is None:
        price_info = get_latest_price(ticker)
        if price_info is None:
            return None
        current_price = price_info['current_price']

    # Ensure features are computed
    if 'rsi' not in df.columns:
        df = engineer_all_features(df)

    # Get latest features for prediction
    latest_features = get_latest_features(df)
    if latest_features is None:
        return None

    # Make prediction
    prediction, probability_up = predict_with_model(model, latest_features)

    # Calculate expected move based on historical returns when model predicted UP
    expected_move = get_historical_returns_on_positive_predictions(df, horizon)

    # If model predicts down, flip the expected move
    if prediction == 0:
        expected_move = -abs(expected_move)
    else:
        expected_move = abs(expected_move)

    # Calculate predicted price
    predicted_price = current_price * (1 + expected_move)

    # Get signal based on probability
    signal = get_signal_from_probability(probability_up)

    # Calculate confidence score
    confidence = calculate_confidence_score(probability_up)

    return PredictionResult(
        ticker=ticker,
        horizon=horizon,
        current_price=current_price,
        predicted_price=predicted_price,
        expected_move_pct=expected_move * 100,
        probability_up=probability_up,
        signal=signal,
        confidence_score=confidence
    )


def predict_all_horizons(
    ticker: str,
    df: pd.DataFrame,
    current_price: Optional[float] = None
) -> Dict[str, Optional[PredictionResult]]:
    """
    Make predictions for all available horizons.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with feature-engineered data
        current_price: Current stock price

    Returns:
        Dictionary mapping horizon names to PredictionResult objects
    """
    horizons = ['next_day', '1_week', '1_month']
    results = {}

    for horizon in horizons:
        result = make_prediction(ticker, horizon, df, current_price)
        results[horizon] = result

    return results


def get_quick_prediction(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get a quick prediction for a ticker using cached data and models.

    This is useful for the watchlist and scanner pages.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with prediction info or None if fails
    """
    # Check if models exist
    if not check_models_trained(ticker):
        return None

    # Load cached training data
    df = load_training_data(ticker)
    if df is None:
        return None

    # Get current price
    price_info = get_latest_price(ticker)
    if price_info is None:
        return None

    # Make next-day prediction (default)
    result = make_prediction(
        ticker,
        'next_day',
        df,
        price_info['current_price']
    )

    if result is None:
        return None

    return {
        "ticker": ticker,
        "current_price": price_info['current_price'],
        "change_pct": price_info['change_pct'],
        "predicted_price": result.predicted_price,
        "probability_up": result.probability_up,
        "signal": result.signal,
        "confidence": result.confidence_score
    }


def predict_for_today(
    ticker: str,
    df: pd.DataFrame,
    current_price: float
) -> Optional[PredictionResult]:
    """
    Special prediction for "Today's Closing Price".

    Uses the next-day model but applies a smaller expected move
    since we're predicting for the same day.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with feature-engineered data
        current_price: Current stock price

    Returns:
        PredictionResult or None if fails
    """
    # Get next-day model (closest to intraday prediction)
    model = get_model_for_horizon(ticker, 'next_day')
    if model is None:
        return None

    # Ensure features are computed
    if 'rsi' not in df.columns:
        df = engineer_all_features(df)

    # Get latest features
    latest_features = get_latest_features(df)
    if latest_features is None:
        return None

    # Make prediction
    prediction, probability_up = predict_with_model(model, latest_features)

    # For same-day prediction, use a smaller expected move
    # Roughly half of what we'd expect for next day
    expected_move = get_historical_returns_on_positive_predictions(df, 'next_day')
    expected_move = expected_move * 0.5  # Reduce for intraday

    if prediction == 0:
        expected_move = -abs(expected_move)
    else:
        expected_move = abs(expected_move)

    predicted_price = current_price * (1 + expected_move)
    signal = get_signal_from_probability(probability_up)
    confidence = calculate_confidence_score(probability_up)

    return PredictionResult(
        ticker=ticker,
        horizon='today',
        current_price=current_price,
        predicted_price=predicted_price,
        expected_move_pct=expected_move * 100,
        probability_up=probability_up,
        signal=signal,
        confidence_score=confidence
    )


def get_prediction_for_horizon(
    ticker: str,
    horizon_display: str,
    df: pd.DataFrame,
    current_price: float
) -> Optional[PredictionResult]:
    """
    Get prediction based on display horizon name.

    Args:
        ticker: Stock ticker symbol
        horizon_display: Display name like "Today's Closing Price", "Next Day Close", etc.
        df: DataFrame with data
        current_price: Current price

    Returns:
        PredictionResult or None
    """
    horizon_map = {
        "Today's Closing Price": "today",
        "Next Day Close": "next_day",
        "1 Week Close": "1_week",
        "1 Month Close": "1_month"
    }

    horizon = horizon_map.get(horizon_display, "next_day")

    if horizon == "today":
        return predict_for_today(ticker, df, current_price)
    else:
        return make_prediction(ticker, horizon, df, current_price)


def batch_predict(
    tickers: list,
    horizon: str = 'next_day'
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Make predictions for multiple tickers.

    Useful for market scanner functionality.

    Args:
        tickers: List of ticker symbols
        horizon: Prediction horizon

    Returns:
        Dictionary mapping tickers to prediction results
    """
    results = {}

    for ticker in tickers:
        # Load cached data
        df = load_training_data(ticker)
        if df is None:
            results[ticker] = None
            continue

        # Get price
        price_info = get_latest_price(ticker)
        if price_info is None:
            results[ticker] = None
            continue

        # Make prediction
        pred = make_prediction(
            ticker,
            horizon,
            df,
            price_info['current_price']
        )

        if pred:
            results[ticker] = pred.to_dict()
        else:
            results[ticker] = None

    return results
