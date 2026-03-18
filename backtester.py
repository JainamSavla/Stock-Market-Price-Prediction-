"""
Backtesting engine for the stock prediction dashboard.
Implements a simple rule-based trading strategy and calculates performance metrics.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

from data_loader import get_data_for_date_range
from features import engineer_all_features, get_feature_columns
from model import get_model_for_horizon, predict_with_model, check_models_trained
from utils import load_training_data


@dataclass
class BacktestResult:
    """Container for backtest results."""
    ticker: str
    start_date: str
    end_date: str
    total_return_pct: float
    win_rate_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    equity_curve: pd.DataFrame
    trades: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding DataFrame and trades list)."""
        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_return_pct": self.total_return_pct,
            "win_rate_pct": self.win_rate_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades
        }


# Strategy parameters
ENTRY_RSI_THRESHOLD = 30
ENTRY_PROBABILITY_THRESHOLD = 0.65
HOLDING_PERIOD = 5  # Number of trading days to hold


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio of a returns series.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Annualize (assuming 252 trading days)
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()

    return float(sharpe) if not np.isnan(sharpe) else 0.0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown from an equity curve.

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Maximum drawdown as a percentage (e.g., -15.5 for 15.5% drawdown)
    """
    if len(equity_curve) == 0:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max * 100

    # Return the minimum (most negative) drawdown
    max_dd = drawdown.min()

    return float(max_dd) if not np.isnan(max_dd) else 0.0


def generate_signals(
    df: pd.DataFrame,
    model: Any,
    rsi_threshold: float = ENTRY_RSI_THRESHOLD,
    prob_threshold: float = ENTRY_PROBABILITY_THRESHOLD
) -> pd.DataFrame:
    """
    Generate entry signals based on strategy rules.

    Entry signal when:
    - RSI < 30
    - Model probability > 0.65

    Args:
        df: DataFrame with features
        model: Trained XGBoost model
        rsi_threshold: RSI threshold for entry
        prob_threshold: Probability threshold for entry

    Returns:
        DataFrame with 'signal' column added (1 = entry signal, 0 = no signal)
    """
    df = df.copy()

    # Initialize signal column
    df['signal'] = 0

    # Get feature columns
    feature_cols = get_feature_columns()

    # Ensure all feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Generate predictions for each row
    probabilities = []
    for idx in df.index:
        try:
            features = df.loc[[idx], feature_cols]
            if features.isna().any().any():
                probabilities.append(0.5)
            else:
                proba = model.predict_proba(features)[0]
                prob_up = proba[1] if len(proba) > 1 else proba[0]
                probabilities.append(prob_up)
        except Exception:
            probabilities.append(0.5)

    df['model_probability'] = probabilities

    # Generate entry signals where both conditions are met
    df['signal'] = (
        (df['rsi'] < rsi_threshold) &
        (df['model_probability'] > prob_threshold)
    ).astype(int)

    return df


def simulate_trades(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    holding_period: int = HOLDING_PERIOD
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Simulate trades based on entry signals.

    Each trade:
    - Enters at close price on signal day
    - Exits after holding_period trading days at close price
    - Invests full capital in each trade (no overlapping positions)

    Args:
        df: DataFrame with 'signal' and 'Close' columns
        initial_capital: Starting capital
        holding_period: Number of days to hold each trade

    Returns:
        Tuple of (trades_list, equity_curve_df)
    """
    trades = []
    equity = initial_capital
    equity_history = []

    # Track position
    in_position = False
    entry_price = 0
    entry_date = None
    position_size = 0
    days_held = 0

    for idx, row in df.iterrows():
        date = idx
        close = row['Close']

        # Check if we should exit current position
        if in_position:
            days_held += 1
            if days_held >= holding_period:
                # Exit trade
                exit_price = close
                pnl = (exit_price - entry_price) / entry_price * position_size
                equity += pnl

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl': pnl,
                    'equity_after': equity
                })

                in_position = False
                days_held = 0

        # Check for entry signal (only if not in position)
        if not in_position and row.get('signal', 0) == 1:
            in_position = True
            entry_price = close
            entry_date = date
            position_size = equity
            days_held = 0

        # Record equity
        if in_position:
            # Mark to market
            current_value = position_size * (close / entry_price)
            equity_history.append({
                'date': date,
                'equity': current_value,
                'in_position': True
            })
        else:
            equity_history.append({
                'date': date,
                'equity': equity,
                'in_position': False
            })

    # If still in position at end, close it
    if in_position and len(df) > 0:
        last_date = df.index[-1]
        last_close = df['Close'].iloc[-1]
        pnl = (last_close - entry_price) / entry_price * position_size
        equity += pnl

        trades.append({
            'entry_date': entry_date,
            'exit_date': last_date,
            'entry_price': entry_price,
            'exit_price': last_close,
            'return_pct': (last_close - entry_price) / entry_price * 100,
            'pnl': pnl,
            'equity_after': equity
        })

    equity_df = pd.DataFrame(equity_history)
    if not equity_df.empty:
        equity_df.set_index('date', inplace=True)

    return trades, equity_df


def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0
) -> Optional[BacktestResult]:
    """
    Run a full backtest for a ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        initial_capital: Starting capital

    Returns:
        BacktestResult object or None if backtest fails
    """
    try:
        # Check if model is trained
        if not check_models_trained(ticker):
            return None

        # Load the model (use 1-week model for 5-day holding period)
        model = get_model_for_horizon(ticker, '1_week')
        if model is None:
            return None

        # Get historical data for the period
        df = get_data_for_date_range(ticker, start_date, end_date)
        if df is None or len(df) < 50:
            return None

        # Engineer features
        df = engineer_all_features(df)

        # Drop NaN rows
        df = df.dropna(subset=get_feature_columns())

        if len(df) < 20:
            return None

        # Generate signals
        df = generate_signals(df, model)

        # Simulate trades
        trades, equity_df = simulate_trades(df, initial_capital)

        if equity_df.empty:
            # No trades executed
            equity_df = pd.DataFrame({
                'date': df.index,
                'equity': [initial_capital] * len(df),
                'in_position': [False] * len(df)
            }).set_index('date')

        # Calculate metrics
        if len(trades) == 0:
            total_return = 0.0
            win_rate = 0.0
            winning_trades = 0
            losing_trades = 0
        else:
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            winning_trades = sum(1 for t in trades if t['return_pct'] > 0)
            losing_trades = len(trades) - winning_trades
            win_rate = (winning_trades / len(trades)) * 100

        # Calculate Sharpe ratio from equity curve
        equity_series = equity_df['equity']
        returns = equity_series.pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)

        # Calculate max drawdown
        max_drawdown = calculate_max_drawdown(equity_series)

        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_return_pct=total_return,
            win_rate_pct=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            equity_curve=equity_df.reset_index(),
            trades=trades
        )

    except Exception as e:
        print(f"Backtest error for {ticker}: {e}")
        return None


def get_equity_curve_for_plot(result: BacktestResult) -> pd.DataFrame:
    """
    Prepare equity curve data for Plotly visualization.

    Args:
        result: BacktestResult object

    Returns:
        DataFrame suitable for plotting
    """
    df = result.equity_curve.copy()

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif df.index.name == 'date':
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])

    return df


def get_trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    """
    Convert trades list to a DataFrame for display.

    Args:
        result: BacktestResult object

    Returns:
        DataFrame with trade information
    """
    if not result.trades:
        return pd.DataFrame()

    df = pd.DataFrame(result.trades)

    # Format dates
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.strftime('%Y-%m-%d')
    df['exit_date'] = pd.to_datetime(df['exit_date']).dt.strftime('%Y-%m-%d')

    # Round numbers
    df['entry_price'] = df['entry_price'].round(2)
    df['exit_price'] = df['exit_price'].round(2)
    df['return_pct'] = df['return_pct'].round(2)
    df['pnl'] = df['pnl'].round(2)
    df['equity_after'] = df['equity_after'].round(2)

    # Rename columns
    df = df.rename(columns={
        'entry_date': 'Entry Date',
        'exit_date': 'Exit Date',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'return_pct': 'Return %',
        'pnl': 'P&L',
        'equity_after': 'Equity'
    })

    return df


def format_metrics_for_display(result: BacktestResult) -> Dict[str, str]:
    """
    Format backtest metrics for UI display.

    Args:
        result: BacktestResult object

    Returns:
        Dictionary with formatted metric strings
    """
    return {
        "Total Return": f"{result.total_return_pct:+.2f}%",
        "Win Rate": f"{result.win_rate_pct:.1f}%",
        "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
        "Max Drawdown": f"{result.max_drawdown_pct:.2f}%",
        "Total Trades": str(result.total_trades),
        "Winning Trades": str(result.winning_trades),
        "Losing Trades": str(result.losing_trades)
    }
