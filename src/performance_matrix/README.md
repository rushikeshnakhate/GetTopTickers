# PyTickerPerformanceMatrix

## Overview
The `PyTickerPerformanceMatrix` module provides a structured approach to computing various performance metrics for trading strategies. It enables users to create different types of performance metric calculators dynamically based on input parameters.

## Features
- Supports multiple performance metrics for evaluating trading strategies.
- Utilizes the Factory pattern for flexible and scalable metric calculation.
- Easily extendable to include additional performance metrics.

## Metric Groups
The following metric groups are supported:

- benchmark_relative: Metrics comparing stock performance to a benchmark.
- distribution: Metrics analyzing the distribution of returns.
- return: Metrics related to returns.
- risk: Metrics quantifying risk.
- risk_adjusted: Metrics adjusting returns for risk.
- trade: Metrics related to trading performance.


## Usage

### Importing the Factory and Calculators
```python
from performance_matrix_factory import PerformanceMatrixFactory
```

### Creating a Performance Metric Instance
```python
metric_type = "sharpe_ratio"  # Example metric type
metric_calculator = PerformanceMatrixFactory.create(metric_type)

# Example data
returns = [0.02, 0.03, -0.01, 0.05, 0.04]
risk_free_rate = 0.01

# Calculate the metric
result = metric_calculator.calculate(returns, risk_free_rate)
print(f"{metric_type} result: {result}")
```

## Available Performance Metrics
- **Sharpe Ratio** - Measures risk-adjusted return.
- **Sortino Ratio** - Similar to Sharpe Ratio but focuses on downside risk.
- **Maximum Drawdown** - Measures the largest drop from a peak to a trough in returns.
- **Calmar Ratio** - Evaluates return relative to maximum drawdown.
- **Other Metrics** - Additional metrics can be added by extending the factory.

## Extending the Factory
To add a new performance metric:
1. Create a new class implementing the required metric logic.
2. Register it in `PerformanceMatrixFactory`.

Example:
```python
class NewPerformanceMetric:
    @staticmethod
    def calculate(returns, *args):
        # Custom calculation logic
        return result

PerformanceMatrixFactory.register("new_metric", NewPerformanceMetric)
```

