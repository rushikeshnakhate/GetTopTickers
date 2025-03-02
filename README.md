# PyPort Ticker Selector

PyPort Ticker Selector is a Python-based stock selection library designed to generate the top N tickers for a given period based on various predefined and user-defined strategies. This library is extension  of the **PyPort Optimization Pipeline**, where selected tickers are optimized to construct an optimal portfolio.

## Features
- Selects top-performing tickers for a given period (e.g., monthly, quarterly, yearly).
- Supports multiple **indicator-based** and **performance-metrics-based** selection strategies.
- Allows users to define custom selection strategies.
- Provides flexibility to choose specific indicators, performance matrices, and rebalancing periods.
- Outputs results in CSV format for further analysis or portfolio optimization.

## Installation
Clone the repository and install dependencies:
```bash
 git clone https://github.com/yourusername/pyport-ticker-selector.git
 cd pyport-ticker-selector
 pip install -r requirements.txt
```

## Usage
### Example: Running the Ticker Selection Process
```python
from pyport_ticker_selector import run_pyport_ticker_selector

# Run for 2024 with default settings
run_pyport_ticker_selector(
    years=[2024],
    rebalancing_period="monthly"
)
```

### Parameters:
- **years** *(list of int)*: List of years to process (e.g., `[2024]`).
- **tickers** *(list of str, optional)*: List of specific tickers to analyze. If None, all available tickers are used.
- **rebalancing_period** *(str, optional)*: Defines the rebalancing period (e.g., "monthly", "quarterly", "yearly"). Default is "monthly".
- **indicators** *(list, optional)*: List of specific indicators to use for selection.
- **performance_matrix** *(list, optional)*: List of performance metrics for evaluation.
- **strategies** *(list, optional)*: Selection strategies to be applied. Default includes predefined strategies.

## Strategies
### Predefined Strategies
1. **Indicator-Based Strategies:**
   - RSI Momentum Strategy
   - MACD Crossover Strategy
   - Bollinger Bands Mean Reversion Strategy
   - And more...

2. **Performance Metrics-Based Strategies:**
   - Sharpe Ratio
   - Sortino Ratio
   - Maximum Drawdown
   - Win Rate Strategy
   - And more...

### Custom Strategies
Users can define their own strategies by extending the `BaseStrategy` class and implementing custom ranking logic.

## Output
- The results are stored as a CSV file: `strategy_results.csv`
- The output includes:
  - **start_date** and **end_date**
  - **strategy_name**
  - **selected tickers**

## Integration with PyPort Optimization Pipeline
This library serves as an input generator for **PyPort Optimization Pipeline**, where the selected tickers are used for portfolio construction and optimization.

