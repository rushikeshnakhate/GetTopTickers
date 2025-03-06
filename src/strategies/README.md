# PyStrategyBuilder Portfolio Optimization

## Overview

The `PyStrategyBuilder` is a Python library designed for running and evaluating various trading strategies. It provides a structured framework for implementing indicator-based and performance-based strategies to analyze financial data and select top-performing assets

### Features

- Multiple Strategy Support: Implements indicator-based, volatility-based, trend-based, momentum-based, and performance matrix-based strategies.
- Flexible Data Input: Accepts indicator and performance data as Pandas DataFrames.
- Strategy Execution: Run individual strategies or all available strategies.
- Logging and Error Handling: Logs detailed outputs and errors during execution.

### Strategy List

The library includes the following strategies:

###  Indicator-Based Strategies
- Trend Strategies: Aroon, Exponential Moving Average, MACD Crossover, Moving Average
- Volatility Strategies: Bollinger Bands, Keltner Channels, Chaikin Money Flow
- Momentum Strategies: RSI, CCI, Price Rate of Change, Williams %R
- Mean Reversion Strategies: Bollinger Bands, Keltner Channels, Donchian Channel
- Volume Strategies: On-Balance Volume

### Performance-Based Strategies
- Benchmark Relative Metrics: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Gain-to-Pain Ratio
- Return Metrics: Annualized Return, Average Daily Return, Cumulative Return
- Risk Metrics: Value at Risk (VaR), Conditional VaR, Maximum Drawdown, Ulcer Index, Volatility
- Distribution Metrics: Win Rate, Loss Rate, Profit Factor

#### Example Configuration

```
import pandas as pd
from PyStrategyBuilder.strategy_factory import StrategyFactory

# Sample data (replace with actual data)
indicators_df = pd.DataFrame({...})  # Load market indicators data
performance_df = pd.DataFrame({...})  # Load stock performance data

# Initialize the Strategy Factory
strategy_factory = StrategyFactory(indicators_df, performance_df, top_n=10)
```