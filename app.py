"""
AI-Powered Stock Prediction Dashboard
Main Streamlit application file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils import (
    construct_ticker,
    get_signal_color,
    format_number,
    format_percentage,
    format_currency,
    ensure_cache_dirs
)
from data_loader import (
    validate_ticker,
    load_or_download_data,
    get_latest_price
)
from features import engineer_all_features
from model import train_all_models, check_models_trained
from predictor import get_prediction_for_horizon, PredictionResult


# Page configuration
st.set_page_config(
    page_title="Stock AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ""
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = ""
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = {}


def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create a candlestick chart with moving averages (no volume subplot)."""
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )
    )

    # Moving averages
    if 'ma_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ma_20'],
                name='MA20', line=dict(color='orange', width=1)
            )
        )

    if 'ma_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ma_50'],
                name='MA50', line=dict(color='blue', width=1)
            )
        )

    if 'ma_200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ma_200'],
                name='MA200', line=dict(color='red', width=1)
            )
        )

    fig.update_layout(
        title=f"{ticker} — Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=550,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create a standalone volume bar chart."""
    fig = go.Figure()

    colors = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#26a69a'
              for i in range(len(df))]

    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors)
    )

    fig.update_layout(
        title=f"{ticker} — Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=350,
        showlegend=False
    )

    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create RSI chart with overbought/oversold lines."""
    fig = go.Figure()

    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=2))
        )

        # Overbought line (70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")

        # Oversold line (30)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")

        # Neutral line (50)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.4)

    fig.update_layout(
        title="RSI (Relative Strength Index — 14 Period)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=350,
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )

    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """Create MACD chart with signal line and histogram."""
    fig = go.Figure()

    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='#2196F3', width=2))
        )

    if 'macd_signal' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='#FF9800', width=2))
        )

    if 'macd_hist' in df.columns:
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['macd_hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['macd_hist'], name='Histogram', marker_color=colors)
        )

    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Date",
        yaxis_title="MACD",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_bollinger_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create a Bollinger Bands chart using MA20 ± 2×std."""
    fig = go.Figure()

    close = df['Close']
    if 'ma_20' in df.columns:
        ma20 = df['ma_20']
        std20 = close.rolling(window=20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20

        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Band',
                                 line=dict(color='rgba(173,216,230,0.6)', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Band',
                                 line=dict(color='rgba(173,216,230,0.6)', dash='dot'),
                                 fill='tonexty', fillcolor='rgba(173,216,230,0.15)'))
        fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20 (Middle)',
                                 line=dict(color='orange', width=1.5)))

    fig.add_trace(go.Scatter(x=df.index, y=close, name='Close',
                             line=dict(color='white', width=1.5)))

    fig.update_layout(
        title=f"{ticker} — Bollinger Bands (20, 2)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def compute_support_resistance(df: pd.DataFrame, current_price: float):
    """Compute nearest support and resistance from recent pivots."""
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # Use recent swing highs/lows as levels
    window = 10
    levels = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            levels.append(highs[i])
        if lows[i] == min(lows[i - window:i + window + 1]):
            levels.append(lows[i])

    if not levels:
        # Fallback: use recent high/low
        recent = df.tail(60)
        return float(recent['Low'].min()), float(recent['High'].max())

    supports = [l for l in levels if l < current_price]
    resistances = [l for l in levels if l > current_price]

    nearest_support = max(supports) if supports else float(df.tail(60)['Low'].min())
    nearest_resistance = min(resistances) if resistances else float(df.tail(60)['High'].max())

    return float(nearest_support), float(nearest_resistance)


def compute_trend_analysis(df: pd.DataFrame):
    """
    Calculate trend strength score from 0 to 100.
    Uses ADX indicator plus price/MA/MACD checks.

    df must contain columns: High, Low, Close, ma_20, ma_50, ma_200, macd, macd_signal.
    """
    from ta.trend import ADXIndicator

    try:
        # Use last 100 rows for calculation
        recent = df.tail(100).copy()

        if len(recent) < 20:
            return {
                "strength": 0,
                "strength_label": "Unknown",
                "direction": "Unknown",
                "alignment": "Unknown"
            }

        # Calculate ADX using ta library
        adx_indicator = ADXIndicator(
            high=recent['High'],
            low=recent['Low'],
            close=recent['Close'],
            window=14
        )

        adx_value = adx_indicator.adx().iloc[-1]

        # Get latest values
        latest = recent.iloc[-1]
        current_price = latest['Close']
        ma20 = latest['ma_20'] if 'ma_20' in recent.columns and pd.notna(latest['ma_20']) else None
        ma50 = latest['ma_50'] if 'ma_50' in recent.columns and pd.notna(latest['ma_50']) else None
        ma200 = latest['ma_200'] if 'ma_200' in recent.columns and pd.notna(latest['ma_200']) else None
        macd_line = latest['macd'] if 'macd' in recent.columns and pd.notna(latest['macd']) else None
        macd_signal = latest['macd_signal'] if 'macd_signal' in recent.columns and pd.notna(latest['macd_signal']) else None

        # Score calculation
        score = 0

        if pd.notna(adx_value):
            if adx_value > 25:
                score += 25
            if adx_value > 40:
                score += 15

        if ma20 is not None and current_price > ma20:
            score += 15
        if ma50 is not None and current_price > ma50:
            score += 15
        if ma200 is not None and current_price > ma200:
            score += 15

        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                score += 15

        # Cap at 100
        score = min(score, 100)

        # Label
        if score <= 25:
            strength_label = "Very Weak"
        elif score <= 45:
            strength_label = "Weak"
        elif score <= 65:
            strength_label = "Moderate"
        elif score <= 80:
            strength_label = "Strong"
        else:
            strength_label = "Very Strong"

        # Direction
        if ma20 is not None and ma50 is not None:
            if current_price > ma20 and current_price > ma50:
                direction = "Uptrend"
            elif current_price < ma20 and current_price < ma50:
                direction = "Downtrend"
            else:
                direction = "Sideways"
        else:
            direction = "Unknown"

        # MA alignment
        if ma20 is not None and ma50 is not None and ma200 is not None:
            if ma20 > ma50 > ma200:
                alignment = "Bullish"
            elif ma20 < ma50 < ma200:
                alignment = "Bearish"
            else:
                alignment = "Mixed"
        else:
            alignment = "Unknown"

        return {
            "strength": score,
            "strength_label": strength_label,
            "direction": direction,
            "alignment": alignment
        }

    except Exception as e:
        return {
            "strength": 0,
            "strength_label": "Unknown",
            "direction": "Unknown",
            "alignment": "Unknown"
        }


def calculate_expected_range(
    current_price: float,
    atr: float,
    horizon: str
) -> tuple:
    """
    Calculate expected price range using ATR scaled by horizon.

    Args:
        current_price: Current stock price
        atr: Average True Range value
        horizon: 'today', 'next_day', '1_week', or '1_month'

    Returns:
        Tuple of (lower, upper) price bounds
    """
    multipliers = {
        "next_day": 1.0,
        "1_week": 2.5,
        "1_month": 5.0,
        "today": 0.5
    }

    multiplier = multipliers.get(horizon, 1.0)
    lower = current_price - (atr * multiplier)
    upper = current_price + (atr * multiplier)
    return round(lower, 2), round(upper, 2)


def display_prediction_card(prediction: PredictionResult, df: pd.DataFrame, accuracy, selected_horizon: str):
    """Display the full prediction panel with 4 sections."""
    current_price = prediction.current_price
    predicted_price = prediction.predicted_price
    prob_up = prediction.probability_up
    signal = prediction.signal
    confidence = prediction.confidence_score
    move_pct = prediction.expected_move_pct

    # -- Compute additional analytics --
    support, resistance = compute_support_resistance(df, current_price)
    trend = compute_trend_analysis(df)

    # Expected range using ATR scaled by horizon (BUG 4 fix)
    from ta.volatility import AverageTrueRange
    try:
        atr_indicator = AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        )
        atr_value = atr_indicator.average_true_range().iloc[-1]
        if pd.isna(atr_value):
            atr_value = current_price * 0.02  # fallback 2%
    except Exception:
        atr_value = current_price * 0.02

    range_low, range_high = calculate_expected_range(current_price, atr_value, selected_horizon)

    # Risk / Reward
    risk = abs(current_price - support)
    reward = abs(resistance - current_price)
    rr_ratio = reward / risk if risk > 0 else 0
    rr_label = "Favorable" if rr_ratio >= 1.5 else "Moderate" if rr_ratio >= 1.0 else "Not favorable"

    # Confidence banner (BUG 3 fix)
    if prob_up > 0.6:
        signal = "BUY"
        if confidence > 60:
            confidence_note = "✅ High confidence BUY signal"
        else:
            confidence_note = "⚠️ Low confidence — treat with caution"
    elif prob_up < 0.4:
        signal = "SELL"
        if confidence > 60:
            confidence_note = "🔴 High confidence SELL signal"
        else:
            confidence_note = "⚠️ Low confidence — treat with caution"
    else:
        signal = "HOLD"
        confidence_note = "⚠️ Uncertain — model suggests waiting"

    # Signal color
    signal_bg = {"BUY": "#1b5e20", "SELL": "#b71c1c", "HOLD": "#e65100"}.get(signal, "#424242")
    trend_color = {"Uptrend": "#26a69a", "Downtrend": "#ef5350", "Sideways": "#ffb74d"}.get(trend["direction"], "#fff")
    alignment_color = {"Bullish": "#26a69a", "Bearish": "#ef5350", "Mixed": "#ffb74d"}.get(trend["alignment"], "#fff")

    # Dynamic card title (BUG 5 fix)
    horizon_labels = {
        "today": "Today's Closing Price",
        "next_day": "Next Day Close",
        "1_week": "1 Week Close",
        "1_month": "1 Month Close"
    }
    card_title = f"🤖 AI PREDICTION ({horizon_labels.get(selected_horizon, 'Next Day Close')})"

    # Model accuracy display (BUG 1 fix — show N/A if 0 or None)
    if accuracy is None or accuracy == 0:
        accuracy_display = "N/A"
    else:
        accuracy_display = f"{accuracy*100:.1f}%"

    # ── SECTION 1: AI PREDICTION ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    ">
        <h3 style="margin:0 0 16px 0; color:#00d4ff; font-size:18px;">{card_title}</h3>
        <div style="display:flex; flex-wrap:wrap; gap:24px; align-items:center;">
            <div>
                <span style="color:#aaa; font-size:13px;">Predicted Price</span><br>
                <span style="color:#fff; font-size:28px; font-weight:bold;">₹{predicted_price:.2f}</span>
                <span style="color:{'#26a69a' if move_pct >= 0 else '#ef5350'}; font-size:16px; margin-left:8px;">
                    ({move_pct:+.2f}%)
                </span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Probability Up</span><br>
                <span style="color:#fff; font-size:22px; font-weight:bold;">{prob_up * 100:.1f}%</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Signal</span><br>
                <span style="
                    background:{signal_bg}; color:white; padding:6px 18px;
                    border-radius:6px; font-weight:bold; font-size:16px;
                ">{signal}</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Model Accuracy</span><br>
                <span style="color:#fff; font-size:22px; font-weight:bold;">{accuracy_display}</span>
            </div>
        </div>
        <div style="margin-top:12px; color:#ffab40; font-size:14px;">{confidence_note}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: PRICE LEVELS ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    ">
        <h3 style="margin:0 0 16px 0; color:#81d4fa; font-size:18px;">📊 PRICE LEVELS (Rule-based, reliable)</h3>
        <div style="display:flex; flex-wrap:wrap; gap:40px;">
            <div>
                <span style="color:#aaa; font-size:13px;">Nearest Support</span><br>
                <span style="color:#26a69a; font-size:20px; font-weight:bold;">₹{support:.2f}</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Nearest Resistance</span><br>
                <span style="color:#ef5350; font-size:20px; font-weight:bold;">₹{resistance:.2f}</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Expected Range</span><br>
                <span style="color:#fff; font-size:20px; font-weight:bold;">₹{range_low:.2f} — ₹{range_high:.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 3: TREND ANALYSIS ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    ">
        <h3 style="margin:0 0 16px 0; color:#ce93d8; font-size:18px;">📈 TREND ANALYSIS</h3>
        <div style="display:flex; flex-wrap:wrap; gap:40px;">
            <div>
                <span style="color:#aaa; font-size:13px;">Trend Strength</span><br>
                <span style="color:#fff; font-size:20px; font-weight:bold;">
                    {trend["strength"]}/100 — {trend["strength_label"]}
                </span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Trend Direction</span><br>
                <span style="color:{trend_color}; font-size:20px; font-weight:bold;">
                    {trend["direction"]}
                </span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">MA Alignment</span><br>
                <span style="color:{alignment_color}; font-size:20px; font-weight:bold;">
                    {trend["alignment"]}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 4: RISK / REWARD ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    ">
        <h3 style="margin:0 0 16px 0; color:#ffcc80; font-size:18px;">⚖️ RISK / REWARD</h3>
        <div style="display:flex; flex-wrap:wrap; gap:40px;">
            <div>
                <span style="color:#aaa; font-size:13px;">Risk</span><br>
                <span style="color:#ef5350; font-size:20px; font-weight:bold;">₹{risk:.2f}</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">Reward</span><br>
                <span style="color:#26a69a; font-size:20px; font-weight:bold;">₹{reward:.2f}</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:13px;">R:R Ratio</span><br>
                <span style="color:#fff; font-size:20px; font-weight:bold;">
                    1:{rr_ratio:.2f} — {rr_label}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stock_analysis_page():
    """Stock Analysis page implementation."""
    st.title("Stock Analysis")

    # Search area
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        ticker_input = st.text_input(
            "Enter Ticker Symbol",
            value=st.session_state.selected_ticker,
            placeholder="e.g., RELIANCE, TCS, INFY"
        )

    with col2:
        exchange = st.selectbox(
            "Exchange",
            ["NSE", "BSE"],
            index=0
        )

    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_clicked = st.button("Search / Load", type="primary", use_container_width=True)

    # Process search
    if search_clicked and ticker_input:
        ticker = construct_ticker(ticker_input, exchange)

        with st.spinner(f"Validating {ticker}..."):
            is_valid, error = validate_ticker(ticker)

        if not is_valid:
            st.error(error)
        else:
            with st.spinner(f"Loading data for {ticker}..."):
                df, is_new_data, status_msg = load_or_download_data(ticker)

            if df is None:
                st.error("Failed to load data. Please try again.")
            else:
                st.info(status_msg)

                # Engineer features
                df = engineer_all_features(df)

                # Train or retrain models if needed
                if is_new_data or not check_models_trained(ticker):
                    with st.spinner("Training prediction models..."):
                        success, train_msg, metrics = train_all_models(df, ticker)
                        if success:
                            st.session_state.model_accuracy[ticker] = metrics
                            st.success(
                                f"Models trained! Accuracy — "
                                f"Next Day: {metrics.get('next_day_accuracy', 0)*100:.1f}%, "
                                f"Week: {metrics.get('1_week_accuracy', 0)*100:.1f}%, "
                                f"Month: {metrics.get('1_month_accuracy', 0)*100:.1f}%"
                            )
                        else:
                            st.warning(f"Training warning: {train_msg}")

                # Store in session state
                st.session_state.current_ticker = ticker
                st.session_state.current_df = df

    # Display stock data if available
    if st.session_state.current_ticker and st.session_state.current_df is not None:
        ticker = st.session_state.current_ticker
        df = st.session_state.current_df

        # Get latest price info
        price_info = get_latest_price(ticker)

        # Summary metrics
        st.subheader(f"{ticker} Summary")

        if price_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{price_info['current_price']:.2f}")
            with col2:
                st.metric("Daily Change", f"₹{price_info['change']:.2f}",
                         delta=f"{price_info['change_pct']:.2f}%")
            with col3:
                st.metric("Volume", f"{price_info['volume']:,}")

        # ── Charts in st.tabs() ──
        st.subheader("Charts")

        # Limit data for display (last 180 days)
        display_df = df.tail(180)

        chart_tabs = st.tabs([
            "📈 Candlestick & MAs",
            "📊 Volume",
            "💪 RSI",
            "📉 MACD",
            "🎯 Bollinger Bands"
        ])

        with chart_tabs[0]:
            fig = create_candlestick_chart(display_df, ticker)
            st.plotly_chart(fig, use_container_width=True)

        with chart_tabs[1]:
            vol_fig = create_volume_chart(display_df, ticker)
            st.plotly_chart(vol_fig, use_container_width=True)

        with chart_tabs[2]:
            rsi_fig = create_rsi_chart(display_df)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with chart_tabs[3]:
            macd_fig = create_macd_chart(display_df)
            st.plotly_chart(macd_fig, use_container_width=True)

        with chart_tabs[4]:
            bb_fig = create_bollinger_chart(display_df, ticker)
            st.plotly_chart(bb_fig, use_container_width=True)

        # ── Prediction Panel ──
        st.subheader("AI Prediction")

        if check_models_trained(ticker):
            horizon = st.selectbox(
                "Select Prediction Horizon",
                ["Next Day Close", "1 Week Close", "1 Month Close", "Today's Closing Price"],
                index=0
            )

            # Map display name to internal horizon key
            horizon_internal_map = {
                "Next Day Close": "next_day",
                "1 Week Close": "1_week",
                "1 Month Close": "1_month",
                "Today's Closing Price": "today",
            }
            selected_horizon = horizon_internal_map.get(horizon, "next_day")

            current_price = price_info['current_price'] if price_info else df['Close'].iloc[-1]
            prediction = get_prediction_for_horizon(ticker, horizon, df, current_price)

            # Get accuracy for the chosen horizon
            horizon_acc_map = {
                "Next Day Close": "next_day_accuracy",
                "1 Week Close": "1_week_accuracy",
                "1 Month Close": "1_month_accuracy",
                "Today's Closing Price": "next_day_accuracy",
            }
            acc_key = horizon_acc_map.get(horizon, "next_day_accuracy")
            accuracy = st.session_state.model_accuracy.get(ticker, {}).get(acc_key, None)
            if accuracy == 0:
                accuracy = None  # Never show 0.0%

            if prediction:
                display_prediction_card(prediction, df, accuracy, selected_horizon)
            else:
                st.warning("Unable to generate prediction. Please try retraining the model.")
        else:
            st.info("Model not yet trained for this ticker. Please search for it first.")


def main():
    """Main application entry point."""
    # Ensure cache directories exist
    ensure_cache_dirs()

    # Initialize session state
    init_session_state()

    # Sidebar
    st.sidebar.title("Stock AI Dashboard")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        **Tips:**
        - NSE stocks: e.g., RELIANCE, TCS
        - BSE stocks: e.g., RELIANCE, INFY
        - All prices shown in ₹ (INR)
        """
    )

    # Only Stock Analysis page
    stock_analysis_page()


if __name__ == "__main__":
    main()
