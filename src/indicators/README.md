# PyTickerIndicators

PyTickerIndicators is a Python package designed to calculate a variety of financial indicators used in technical analysis for stock market data. This package allows users to calculate indicators based on close prices and historical price data (Open, High, Low, Volume) for any stock ticker over a given time range.

## Features

- **Close Price Indicators**: Includes a wide range of indicators such as Bollinger Bands, Moving Average, Exponential Moving Average, Relative Strength Index, and many more.
- **Historical Price Indicators**: Includes indicators like On-Balance Volume, Aroon, Commodity Channel Index, Keltner Channel, and more.
- **Caching**: Uses a cache to store calculated indicators for faster future retrieval.
- **Flexible API**: Provides functions to calculate indicators for single or multiple tickers at once, along with options to specify selected indicators.
- **Pandas Integration**: Returns results as DataFrames for easy manipulation and analysis.


### Available Indicators

This document provides a breakdown of available indicators, grouped by their type.

### 1. Close Price Indicators
These indicators are calculated using the **Close** price of a stock or asset.

- **Bollinger Bands**: Measures volatility based on standard deviation around a moving average.
- **Exponential Moving Average (EMA)**: A type of moving average that gives more weight to recent prices.
- **Moving Average (MA)**: A simple moving average of prices over a defined period.
- **Relative Strength Index (RSI)**: Measures the speed and change of price movements to evaluate overbought or oversold conditions.
- **Price Rate of Change (ROC)**: Measures the percentage change in price over a defined period.

### 2. Historical Price Indicators
These indicators are calculated using multiple price points: **Open**, **High**, **Low**, **Volume**.

- **Accumulation Distribution Line (ADL)**: Measures the cumulative flow of money into and out of an asset.
- **Aroon**: Indicates trends in the market by measuring the time between highs and lows over a period.
- **Chaikin Money Flow (CMF)**: Combines price and volume to indicate the accumulation or distribution of an asset.
- **Commodity Channel Index (CCI)**: Measures the deviation of price from its average to identify cyclical trends.
- **Donchian Channel**: A trend-following indicator that calculates the highest high and lowest low over a period.
- **Ease of Movement (EoM)**: Measures the relationship between price and volume to assess the ease with which price moves.
- **Force Index (FI)**: Uses both price change and volume to assess the strength behind a price move.
- **Keltner Channel**: A volatility-based indicator using the exponential moving average and average true range.
- **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages.
- **Money Flow Index (MFI)**: Combines price and volume to identify the strength of price movements.
- **On Balance Volume (OBV)**: Measures the buying and selling pressure based on volume and price.
- **Vortex Indicator**: Identifies trends and trend reversals based on price movements.
- **Williams %R**: A momentum indicator that identifies overbought or oversold conditions.

### Usage
You can use these indicators to analyze different aspects of price movements and market trends. They are often used in combination to create more accurate trading strategies.

You can calculate indicators for multiple tickers at once by passing a list of tickers to the get_indicator_bulk function.



## Usage
```
from PyTickerIndicators import IndicatorFactory
import pandas as pd

# Example stock data as a pandas DataFrame
ticker_data = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Close': [100, 102, 101],
    'Open': [98, 101, 100],
    'High': [101, 103, 102],
    'Low': [97, 100, 99],
    'Volume': [1000, 1500, 1200],
})

# Initialize the factory with the desired period (default is 14)
indicator_factory = IndicatorFactory(period=14)

# Calculate indicators for a single ticker
result = indicator_factory.calculate_all_indicators(
    ticker_data_df=ticker_data, ticker="AAPL", start_date="2023-01-01", end_date="2023-01-03"
)
print(result)
```

### List of indicators to calculate
```
indicators_to_calculate = [
    'BollingerBands', 
    'ExponentialMovingAverage', 
    'Aroon',
    'OnBalanceVolume'
]
```

### Caching
- The package utilizes a cache (Pandas DataFrame) to store results of indicator calculations. This improves performance by avoiding recalculating indicators for the same ticker and date range multiple times.
- The cache is handled by the CacheFactory class.