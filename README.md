# AI-Powered Stock Prediction Dashboard:-https://stockmarketpredicto.streamlit.app/



A local stock analysis and prediction dashboard built with Python, Streamlit, and XGBoost. Analyze any stock globally, view technical indicators, and get AI-powered price predictions.

## Features

- **Stock Analysis**: Search and analyze any stock (US, NSE, BSE markets)
- **Technical Indicators**: RSI, MACD, Moving Averages (MA20, MA50, MA200)
- **AI Predictions**: XGBoost-based price predictions for multiple horizons
- **Watchlist**: Track your favorite stocks
- **Market Scanner**: Scan for trading opportunities
- **Backtesting**: Test a simple RSI + AI strategy

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Stock Search

Enter ticker symbols in the following format:
- **US Markets**: `AAPL`, `MSFT`, `GOOGL`
- **NSE (India)**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- **BSE (India)**: `RELIANCE.BO`, `TCS.BO`, `INFY.BO`

Or use the Exchange dropdown to automatically append the correct suffix.

## Technical Indicators

The dashboard computes the following indicators using the `ta` library:

| Indicator | Description |
|-----------|-------------|
| RSI (14) | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| MA20/50/200 | Simple Moving Averages |
| Volatility | 20-day rolling standard deviation |
| Volume Ratio | Current volume vs 20-day average |

## AI Prediction Model

The system trains three XGBoost classifiers for each stock:
- **Next Day**: Predicts tomorrow's direction
- **1 Week**: Predicts direction over 5 trading days
- **1 Month**: Predicts direction over 20 trading days

### Features Used
- 5/10/20-day returns
- Moving averages (20/50/200 day)
- RSI, MACD line, MACD signal, MACD histogram
- 20-day volatility
- Volume change and volume moving average
- Price distance from MA20 and MA50

### Signal Interpretation
- **BUY**: Probability > 60%
- **SELL**: Probability < 40%
- **HOLD**: Probability between 40-60%

## Market Scanner

Scans stocks for opportunities meeting these criteria:
- RSI < 35 (Oversold)
- Volume > 1.5x 20-day average (Volume Spike)
- AI Probability > 60% (Model predicts upside)

## Backtesting Strategy

The built-in backtester implements a simple strategy:

**Entry Rules:**
- RSI < 30
- AI probability > 65%

**Exit Rules:**
- Hold for 5 trading days
- Exit at market close

**Metrics Calculated:**
- Total Return (%)
- Win Rate (%)
- Sharpe Ratio
- Maximum Drawdown (%)

## Data Storage

All data is cached locally in the `/cache` directory:

```
cache/
  models/     # Trained XGBoost models (joblib)
  data/       # Historical data (parquet)
  history.json    # Search history
  watchlist.json  # User watchlist
```

## Incremental Learning

The system implements incremental learning:
1. On first search: Downloads full history (2015-present), trains models
2. On revisit: Downloads only new data, merges with cached data, retrains

## Project Structure

```
stock-ai-dashboard/
  app.py           # Main Streamlit app
  data_loader.py   # Data fetching and caching
  features.py      # Feature engineering
  model.py         # XGBoost model training
  predictor.py     # Prediction logic
  scanner.py       # Market scanner
  backtester.py    # Backtesting engine
  watchlist.py     # Watchlist management
  utils.py         # Shared utilities
  history.py       # Search history
  requirements.txt
  README.md
```

## Requirements

- Python 3.11+
- streamlit
- pandas
- numpy
- yfinance
- plotly
- xgboost
- scikit-learn
- ta
- joblib

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always do your own research before making investment decisions. Past performance does not guarantee future results.

## License

MIT License
