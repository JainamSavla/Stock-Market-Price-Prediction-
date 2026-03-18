"""
Market scanner module for the stock prediction dashboard.
Scans stocks for trading opportunities based on technical indicators and model predictions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd

from data_loader import load_or_download_data, get_latest_price
from features import engineer_all_features, get_current_rsi, get_volume_ratio
from model import train_all_models, check_models_trained
from predictor import make_prediction
from utils import load_training_data, get_signal_from_probability


@dataclass
class ScanResult:
    """Container for scan results for a single stock."""
    ticker: str
    current_price: float
    change_pct: float
    rsi: float
    volume_ratio: float
    probability_up: float
    signal: str
    rsi_oversold: bool
    volume_spike: bool
    high_probability: bool
    opportunity_score: int  # Number of criteria met

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Scanner criteria thresholds
RSI_OVERSOLD_THRESHOLD = 35
VOLUME_SPIKE_THRESHOLD = 1.5
HIGH_PROBABILITY_THRESHOLD = 0.6


def check_rsi_oversold(rsi: float) -> bool:
    """Check if RSI indicates oversold condition."""
    return rsi < RSI_OVERSOLD_THRESHOLD


def check_volume_spike(volume_ratio: float) -> bool:
    """Check if volume is spiking above average."""
    return volume_ratio > VOLUME_SPIKE_THRESHOLD


def check_high_probability(probability: float) -> bool:
    """Check if model probability is above threshold."""
    return probability > HIGH_PROBABILITY_THRESHOLD


def calculate_opportunity_score(
    rsi_oversold: bool,
    volume_spike: bool,
    high_probability: bool
) -> int:
    """
    Calculate opportunity score based on criteria met.

    Returns:
        Integer from 0-3 indicating how many criteria are met
    """
    score = 0
    if rsi_oversold:
        score += 1
    if volume_spike:
        score += 1
    if high_probability:
        score += 1
    return score


def scan_single_ticker(
    ticker: str,
    train_if_needed: bool = True
) -> Optional[ScanResult]:
    """
    Scan a single ticker for opportunities.

    Args:
        ticker: Stock ticker symbol
        train_if_needed: If True, train model if not already trained

    Returns:
        ScanResult or None if scan fails
    """
    try:
        # Check if models exist, train if needed
        if not check_models_trained(ticker):
            if train_if_needed:
                # Load/download data and train
                df, is_new, msg = load_or_download_data(ticker)
                if df is None:
                    return None

                success, train_msg, metrics = train_all_models(df, ticker)
                if not success:
                    return None
            else:
                return None

        # Load training data (which has features)
        df = load_training_data(ticker)
        if df is None:
            return None

        # Ensure features are computed
        if 'rsi' not in df.columns:
            df = engineer_all_features(df)

        # Get current price info
        price_info = get_latest_price(ticker)
        if price_info is None:
            return None

        current_price = price_info['current_price']
        change_pct = price_info['change_pct']

        # Get current RSI
        rsi = get_current_rsi(df)

        # Get volume ratio
        volume_ratio = get_volume_ratio(df)

        # Get model prediction
        prediction = make_prediction(ticker, 'next_day', df, current_price)
        if prediction is None:
            probability_up = 0.5
            signal = "HOLD"
        else:
            probability_up = prediction.probability_up
            signal = prediction.signal

        # Check criteria
        rsi_oversold = check_rsi_oversold(rsi)
        volume_spike = check_volume_spike(volume_ratio)
        high_probability = check_high_probability(probability_up)

        # Calculate opportunity score
        opportunity_score = calculate_opportunity_score(
            rsi_oversold, volume_spike, high_probability
        )

        return ScanResult(
            ticker=ticker,
            current_price=current_price,
            change_pct=change_pct,
            rsi=rsi,
            volume_ratio=volume_ratio,
            probability_up=probability_up,
            signal=signal,
            rsi_oversold=rsi_oversold,
            volume_spike=volume_spike,
            high_probability=high_probability,
            opportunity_score=opportunity_score
        )

    except Exception as e:
        print(f"Error scanning {ticker}: {e}")
        return None


def scan_multiple_tickers(
    tickers: List[str],
    train_if_needed: bool = True,
    min_opportunity_score: int = 0,
    progress_callback: Optional[callable] = None
) -> List[ScanResult]:
    """
    Scan multiple tickers for opportunities.

    Args:
        tickers: List of ticker symbols to scan
        train_if_needed: If True, train models for tickers that need it
        min_opportunity_score: Minimum score to include in results
        progress_callback: Optional callback function(current, total) for progress

    Returns:
        List of ScanResult objects, sorted by opportunity score
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i + 1, total)

        result = scan_single_ticker(ticker, train_if_needed)

        if result is not None and result.opportunity_score >= min_opportunity_score:
            results.append(result)

    # Sort by opportunity score (descending), then by probability (descending)
    results.sort(key=lambda x: (x.opportunity_score, x.probability_up), reverse=True)

    return results


def scan_results_to_dataframe(results: List[ScanResult]) -> pd.DataFrame:
    """
    Convert scan results to a pandas DataFrame for display.

    Args:
        results: List of ScanResult objects

    Returns:
        DataFrame with formatted scan results
    """
    if not results:
        return pd.DataFrame()

    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)

    # Rename columns for display
    column_names = {
        'ticker': 'Ticker',
        'current_price': 'Price',
        'change_pct': 'Change %',
        'rsi': 'RSI',
        'volume_ratio': 'Vol Ratio',
        'probability_up': 'Prob Up',
        'signal': 'Signal',
        'rsi_oversold': 'RSI < 35',
        'volume_spike': 'Vol Spike',
        'high_probability': 'High Prob',
        'opportunity_score': 'Score'
    }

    df = df.rename(columns=column_names)

    return df


def filter_opportunities(
    results: List[ScanResult],
    require_rsi_oversold: bool = False,
    require_volume_spike: bool = False,
    require_high_probability: bool = False
) -> List[ScanResult]:
    """
    Filter scan results based on specific criteria.

    Args:
        results: List of ScanResult objects
        require_rsi_oversold: Only include if RSI < 35
        require_volume_spike: Only include if volume > 1.5x average
        require_high_probability: Only include if probability > 0.6

    Returns:
        Filtered list of ScanResult objects
    """
    filtered = []

    for result in results:
        include = True

        if require_rsi_oversold and not result.rsi_oversold:
            include = False
        if require_volume_spike and not result.volume_spike:
            include = False
        if require_high_probability and not result.high_probability:
            include = False

        if include:
            filtered.append(result)

    return filtered


def get_top_opportunities(
    tickers: List[str],
    top_n: int = 10,
    train_if_needed: bool = True
) -> List[ScanResult]:
    """
    Get the top N opportunities from a list of tickers.

    Args:
        tickers: List of ticker symbols
        top_n: Number of top results to return
        train_if_needed: Whether to train models if needed

    Returns:
        Top N ScanResult objects
    """
    results = scan_multiple_tickers(tickers, train_if_needed)
    return results[:top_n]


def get_scan_summary(results: List[ScanResult]) -> Dict[str, Any]:
    """
    Get a summary of scan results.

    Args:
        results: List of ScanResult objects

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "total_scanned": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "rsi_oversold_count": 0,
            "volume_spike_count": 0,
            "high_probability_count": 0,
            "avg_probability": 0,
            "avg_rsi": 0
        }

    buy_count = sum(1 for r in results if r.signal == "BUY")
    sell_count = sum(1 for r in results if r.signal == "SELL")
    hold_count = sum(1 for r in results if r.signal == "HOLD")

    return {
        "total_scanned": len(results),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "hold_signals": hold_count,
        "rsi_oversold_count": sum(1 for r in results if r.rsi_oversold),
        "volume_spike_count": sum(1 for r in results if r.volume_spike),
        "high_probability_count": sum(1 for r in results if r.high_probability),
        "avg_probability": sum(r.probability_up for r in results) / len(results),
        "avg_rsi": sum(r.rsi for r in results) / len(results)
    }
