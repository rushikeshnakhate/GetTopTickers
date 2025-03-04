# PyStrategyBuilder Portfolio Optimization

## Overview

The `PyStrategyBuilder` is a flexible framework designed for portfolio optimization, allowing users to choose and configure different strategies for expected returns, covariance matrices, and optimizations. It supports various indicators and custom strategies for building and optimizing portfolios. This readme outlines the available strategies, indicators, how to add new strategies, and how to create composite strategies.

## Available Indicators

### 1. **Trend Indicators**
   - **Moving Average (MA)**: Uses different periods of moving averages (simple, exponential) to identify trends.
   - **Moving Average Convergence Divergence (MACD)**: Identifies changes in the strength, direction, momentum, and duration of a trend in a stock’s price.
   - **Relative Strength Index (RSI)**: Measures the speed and change of price movements to identify overbought or oversold conditions.

### 2. **Volatility Indicators**
   - **Bollinger Bands**: Measures volatility using standard deviation bands placed above and below the moving average.
   - **Average True Range (ATR)**: Measures market volatility by analyzing the range of price movement.

### 3. **Volume Indicators**
   - **On-Balance Volume (OBV)**: Uses volume flow to predict changes in stock price.
   - **Accumulation/Distribution (A/D)**: Measures the cumulative flow of money into and out of a security.

### 4. **Momentum Indicators**
   - **Momentum Oscillator**: Measures the rate of change of stock prices.
   - **Stochastic Oscillator**: Compares a stock’s closing price to its price range over a specific period to determine momentum.

### 5. **Mean Reversion Indicators**
   - **Bollinger Bands**: Also used to detect potential reversals when price moves away from the mean.
   - **Z-Score**: Measures the deviation of a stock’s price from its historical mean to predict potential reversals.

## Usage

### Setting Up the Strategy

To use the `PyStrategyBuilder` for portfolio optimization, you need to select from the available strategies and indicators. Below is an example of how to set up a strategy using the available indicators.

#### Example Configuration

```yaml
strategy:
  expected_returns: 
    method: mean_historical
  covariance_matrix: 
    method: historical_covariance
  optimizer:
    method: mean_variance
  indicators:
    trend_indicators:
      - moving_average
      - rsi
    volatility_indicators:
      - bollinger_bands
    momentum_indicators:
      - momentum_oscillator
    mean_reversion_indicators:
      - z_score
Adding a New Indicator
To add a new indicator to the system, follow these steps:

Create a Python file for your new indicator in the appropriate folder, for example, under indicator_based.
Define the logic for your indicator (e.g., in the indicator_name.py file).
Update the strategy configuration to include the new indicator, using the appropriate method.
Adding a New Strategy
To add a new strategy for expected returns, covariance matrix, or optimization:

Create a Python file for the new strategy in the respective folder (e.g., base_strategy.py, covariance_matrix.py).
Implement the algorithm or logic for the new strategy.
Modify the PyStrategyBuilder configuration to add your new strategy.
Example of Adding a Custom Expected Return Strategy
python
Copy
Edit
# In expected_returns/custom_return_strategy.py

class CustomReturnStrategy:
    def __init__(self, data):
        self.data = data
    
    def calculate(self):
        # Custom logic for expected returns
        return self.data.mean()  # Example placeholder

# In strategy configuration (YAML)
strategy:
  expected_returns: 
    method: custom_return_strategy
Creating Composite Strategies
Composite strategies allow you to combine multiple individual strategies to create a more complex, hybrid approach. Here’s how to define a composite strategy:

Create individual strategies (e.g., expected returns, covariance matrices, and optimization).
Combine the strategies into a single composite strategy.
Example: Composite Strategy
python
Copy
Edit
# In composite_strategy.py

class CompositeStrategy:
    def __init__(self, expected_returns, covariance_matrix, optimizer):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.optimizer = optimizer
    
    def run(self):
        returns = self.expected_returns.calculate()
        covariance = self.covariance_matrix.calculate()
        optimized_portfolio = self.optimizer.optimize(returns, covariance)
        return optimized_portfolio
Example of Using a Composite Strategy in YAML Configuration
yaml
Copy
Edit
strategy:
  composite:
    expected_returns: 
      method: mean_historical
    covariance_matrix:
      method: shrinkage_estimator
    optimizer:
      method: mean_variance