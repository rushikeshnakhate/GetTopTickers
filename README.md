# GetTopTickers

GetTopTickers is a Python-based stock selection method designed to generate the top N tickers for a given period based on various predefined and user-defined strategies. This library is extension  of the **PyPort Optimization Pipeline**, where selected tickers are optimized to construct an optimal portfolio.

## Features
- Selects top-performing tickers for a given period (e.g., monthly, quarterly, yearly).
- Supports multiple **indicator-based** and **performance-metrics-based** selection strategies.
- Allows users to define custom selection strategies.
- Provides flexibility to choose specific indicators, performance matrices, and rebalancing periods.
- Outputs results in CSV format for further analysis or portfolio optimization.


###  Usage
### Example: Running the Ticker Selection Process
```python
from pyport_ticker_selector import run_pyport_ticker_selector

# Run for 2024 with default settings
run_pyport_ticker_selector(
  years=[2024],
  rebalancing_period="monthly"
)
```


Example 1: Running the Ticker Selection Process for a Single Year (2024)
```
from pyport_ticker_selector import run_pyport_ticker_selector

# Run for the year 2024 with default settings (monthly rebalancing)
run_pyport_ticker_selector(
    years=[2024],
    rebalancing_period="monthly"
)
```
Example 2: Running for Multiple Years with Quarterly Rebalancing

```from pyport_ticker_selector import run_pyport_ticker_selector

# Run for 2024 and 2025 with quarterly rebalancing
run_pyport_ticker_selector(
    years=[2024, 2025],
    rebalancing_period="quarterly"
)
```
Example 3: Using Specific Tickers and Custom Indicators

```
from pyport_ticker_selector import run_pyport_ticker_selector

# Specify a list of tickers and custom performance metrics for selection
run_pyport_ticker_selector(
    years=[2024],
    tickers=["AAPL", "GOOG", "AMZN"],
    rebalancing_period="monthly",
    indicators=["RSI", "MACD"],
    performance_matrix=["Sharpe Ratio", "Max Drawdown"]
)
```

### Parameters:

### **years** *(list of int)*:
- **Description**: List of years to process for the ticker selection. For example, `[2024, 2025]`.
- **Required**: Yes
- **Example**: `[2024]`

### **tickers** *(list of str, optional)*:
- **Description**: List of specific stock tickers to analyze. If `None`, all available tickers (such as Nifty50) will be used.
- **Required**: No
- **Default**: `None` (analyzed for all available tickers)
- **Example**: `["AAPL", "GOOG", "AMZN"]`

### **rebalancing_days** *(int, optional)*:
- **Description**: Defines the rebalancing day. This is the frequency at which the ticker list is recalculated.
- **Values**: `1`, `2`
- **Required**: No
- **Default**: `None`
- **Example**: `1`

### **rebalancing_months** *(int, optional)*:
- **Description**: Defines the rebalancing months. This is the frequency at which the ticker list is recalculated.
- **Values**: `1`, `2`
- **Required**: No
- **Default**: `1`
- **Example**: `1`

### **indicators** *(list of str, optional)*:
- **Description**: List of technical indicators to be used for stock selection. You can use predefined indicators such as RSI, MACD, etc.
- **Required**: No
- **Default**: Depends on the selected strategy (usually includes RSI, MACD, etc.).
- **Example**: `["RSI", "MACD"]`

### **performance_matrix** *(list of str, optional)*:
- **Description**: List of performance metrics used to evaluate the tickers. Metrics include Sharpe ratio, Sortino ratio, and others.
- **Required**: No
- **Default**: Uses predefined metrics like Sharpe Ratio, Sortino Ratio.
- **Example**: `["Sharpe Ratio", "Max Drawdown"]`

### **strategies** *(list of str, optional)*:
- **Description**: List of selection strategies to apply. You can use predefined strategies or define custom strategies.
- **Required**: No
- **Default**: Uses predefined strategies (such as RSI Momentum, Sharpe Ratio, etc.)
- **Example**: `["RSI Momentum", "Sharpe Ratio"]`

## Strategies

### Predefined Strategies

#### **Indicator-Based Strategies**:
- **RSI Momentum Strategy**: Uses the Relative Strength Index (RSI) to determine momentum.
- **MACD Crossover Strategy**: Uses the Moving Average Convergence Divergence (MACD) to identify potential buy or sell signals.
- **Bollinger Bands Mean Reversion Strategy**: Uses Bollinger Bands for mean-reversion trading signals.
- More to come...

#### **Performance Metrics-Based Strategies**:
- **Sharpe Ratio**: Maximizes risk-adjusted return.
- **Sortino Ratio**: Focuses on downside risk to reward ratio.
- **Maximum Drawdown**: Minimizes large losses from peak to trough.
- **Win Rate Strategy**: Selects tickers with a higher win rate of price movements.
- More to come...

### Custom Strategies
Users can define their own strategies by extending the `BaseStrategy` class and implementing custom ranking logic.

## Output
- The results are stored as a CSV file: `strategy_results.csv`
- The output includes:
  - **start_date** and **end_date**
  - **strategy_name**
  - **selected tickers**

### Integration with PyPort Optimization Pipeline
This library serves as an input generator for **PyPort Optimization Pipeline**, where the selected tickers are used for portfolio construction and optimization.

