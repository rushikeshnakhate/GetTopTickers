# PyPortTickerSelector Validation Suite - API Documentation

This document provides detailed API documentation for each folder and module in the validation suite.

## 📊 benchmark_comparison/

**Purpose**: Compare PyPortTickerSelector results against market benchmarks and perform statistical validation.

### Files and APIs:

#### `market_benchmark_fetcher.py`
**Purpose**: Fetch and calculate performance metrics for NSE benchmarks (NSE 50, NSE 100, NSE 500).

**Key Classes**:
- `MarketBenchmarkFetcher`: Main class for fetching benchmark data

**API**:
```python
from src.validation.benchmark_comparison import MarketBenchmarkFetcher

# Initialize
fetcher = MarketBenchmarkFetcher()

# Get benchmark tickers
tickers = fetcher.get_benchmark_tickers('nse_50')  # Returns list of NSE 50 tickers

# Get benchmark performance
performance = fetcher.get_benchmark_performance(
    benchmark='nse_50',           # 'nse_50', 'nse_100', 'nse_500'
    start_date='2023-01-01',      # Start date (YYYY-MM-DD)
    end_date='2023-12-31',        # End date (YYYY-MM-DD)
    rebalance_frequency='monthly' # 'daily', 'monthly', 'quarterly'
)

# Get benchmark returns series for statistical testing
returns = fetcher.get_benchmark_returns_series('nse_50', '2023-01-01', '2023-12-31')
```

**Parameters**:
- `benchmark`: Benchmark name ('nse_50', 'nse_100', 'nse_500')
- `start_date`: Start date in 'YYYY-MM-DD' format
- `end_date`: End date in 'YYYY-MM-DD' format
- `rebalance_frequency`: Rebalancing frequency ('daily', 'monthly', 'quarterly')

**Returns**: Dictionary with performance metrics (annual_return, sharpe_ratio, volatility, max_drawdown, etc.)

#### `benchmark_comparator.py`
**Purpose**: Compare selected tickers with market benchmarks and analyze selection accuracy.

**Key Classes**:
- `BenchmarkComparator`: Main class for benchmark comparison

**API**:
```python
from src.validation.benchmark_comparison import BenchmarkComparator

# Initialize
comparator = BenchmarkComparator(market_fetcher, performance_profiler)

# Compare with benchmarks
comparison = comparator.compare_with_benchmarks(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # List of selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    benchmarks=['nse_50', 'nse_100']             # List of benchmarks to compare
)

# Analyze selection accuracy
accuracy = comparator.selection_accuracy_analysis(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # List of selected tickers
    benchmark='nse_50',                          # Benchmark for comparison
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31'                        # End date
)

# Calculate risk-adjusted metrics
risk_metrics = comparator.calculate_risk_adjusted_metrics(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    benchmark_tickers=['RELIANCE.NS', 'TCS.NS'], # Benchmark tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31'                        # End date
)
```

**Parameters**:
- `selected_tickers`: List of ticker symbols you selected
- `benchmark_tickers`: List of benchmark ticker symbols
- `start_date`: Start date for analysis
- `end_date`: End date for analysis
- `benchmarks`: List of benchmark names to compare against

**Returns**: Dictionary with comparison results, accuracy metrics, and risk-adjusted performance

#### `performance_validator.py`
**Purpose**: Validate selection performance against various criteria and standards.

**Key Classes**:
- `PerformanceValidator`: Main class for performance validation

**API**:
```python
from src.validation.benchmark_comparison import PerformanceValidator

# Initialize
validator = PerformanceValidator()

# Validate selection performance
validation = validator.validate_selection_performance(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # List of selected tickers
    performance_metrics={                        # Performance metrics dictionary
        'annual_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'volatility': 0.18
    },
    benchmark_metrics={                          # Optional benchmark metrics
        'annual_return': 0.12,
        'sharpe_ratio': 1.0,
        'max_drawdown': -0.10,
        'volatility': 0.20
    }
)

# Validate consistency over time
consistency = validator.validate_consistency([
    {'hit_rate': 0.7, 'sharpe_ratio': 1.2, 'annual_return': 0.15},
    {'hit_rate': 0.65, 'sharpe_ratio': 1.1, 'annual_return': 0.14}
])

# Validate risk metrics
risk_validation = validator.validate_risk_metrics({
    'max_drawdown': -0.08,
    'volatility': 0.18,
    'returns': [0.01, -0.02, 0.03, ...]  # List of returns
})

# Set custom validation criteria
validator.set_validation_criteria({
    'min_hit_rate': 0.6,        # Minimum 60% hit rate
    'min_sharpe_ratio': 1.0,    # Minimum Sharpe ratio of 1.0
    'max_drawdown_threshold': -0.2,  # Maximum 20% drawdown
    'min_annual_return': 0.1    # Minimum 10% annual return
})
```

**Parameters**:
- `selected_tickers`: List of selected ticker symbols
- `performance_metrics`: Dictionary with performance metrics
- `benchmark_metrics`: Optional benchmark metrics for comparison
- `historical_results`: List of historical performance results

**Returns**: Dictionary with validation results, criteria checks, and recommendations

#### `statistical_tests.py`
**Purpose**: Perform comprehensive statistical tests for validation and significance testing.

**Key Classes**:
- `StatisticalTests`: Main class for statistical testing

**API**:
```python
from src.validation.benchmark_comparison import StatisticalTests

# Initialize
stats = StatisticalTests()

# Perform comprehensive tests
comprehensive_results = stats.perform_comprehensive_tests(
    your_returns=pd.Series([0.01, -0.02, 0.03, ...]),      # Your portfolio returns
    benchmark_returns=pd.Series([0.008, -0.015, 0.025, ...]), # Benchmark returns
    significance_level=0.05                                  # Significance level
)

# Test outperformance significance
outperformance = stats.test_outperformance_significance(
    your_returns=pd.Series([0.01, -0.02, 0.03, ...]),      # Your returns
    benchmark_returns=pd.Series([0.008, -0.015, 0.025, ...]), # Benchmark returns
    test_type='both'                                        # 'parametric', 'nonparametric', 'both'
)

# Test risk-adjusted performance
risk_adjusted = stats.test_risk_adjusted_performance(
    your_returns=pd.Series([0.01, -0.02, 0.03, ...]),      # Your returns
    benchmark_returns=pd.Series([0.008, -0.015, 0.025, ...]), # Benchmark returns
    risk_free_rate=0.05                                     # Risk-free rate (annual)
)
```

**Parameters**:
- `your_returns`: Pandas Series of your portfolio returns
- `benchmark_returns`: Pandas Series of benchmark returns
- `significance_level`: Statistical significance level (default: 0.05)
- `test_type`: Type of test ('parametric', 'nonparametric', 'both')
- `risk_free_rate`: Risk-free rate for risk-adjusted calculations

**Returns**: Dictionary with statistical test results, p-values, and significance indicators

---

## ⚡ performance_profiling/

**Purpose**: Profile execution performance, memory usage, and measure selection accuracy.

### Files and APIs:

#### `latency_profiler.py`
**Purpose**: Profile execution time and memory usage of ticker selection operations.

**Key Classes**:
- `PerformanceProfiler`: Main class for performance profiling

**API**:
```python
from src.validation.performance_profiling import PerformanceProfiler

# Initialize
profiler = PerformanceProfiler()

# Profile a function with decorator
@profiler.profile_execution_time
def your_ticker_selection_function(years, top_n_tickers):
    # Your ticker selection logic
    return selected_tickers

# Benchmark against competitors
benchmark_results = profiler.benchmark_vs_competitors(
    ticker_count=15,                    # Number of tickers to select
    time_periods=['2023', '2024'],      # List of time periods to test
    competitor_functions={              # Optional: custom competitor functions
        'random_selection': random_func,
        'momentum_strategy': momentum_func
    }
)

# Profile memory usage
memory_stats = profiler.profile_memory_usage(
    func=your_function,                 # Function to profile
    *args,                             # Function arguments
    **kwargs                           # Function keyword arguments
)

# Generate performance report
report_df, summary_stats = profiler.generate_performance_report()

# Get performance summary
summary = profiler.get_performance_summary()

# Use context manager for profiling
with profiler.profile_context('ticker_selection'):
    selected_tickers = your_ticker_selection_function([2023], 15)
```

**Parameters**:
- `ticker_count`: Number of tickers to select for benchmarking
- `time_periods`: List of time periods to test
- `competitor_functions`: Dictionary of competitor functions to benchmark against
- `func`: Function to profile for memory usage

**Returns**: Dictionary with performance statistics, execution times, and memory usage

#### `memory_profiler.py`
**Purpose**: Detailed memory usage profiling and leak detection.

**Key Classes**:
- `MemoryProfiler`: Main class for memory profiling

**API**:
```python
from src.validation.performance_profiling import MemoryProfiler

# Initialize
memory_profiler = MemoryProfiler()

# Start continuous monitoring
memory_profiler.start_memory_monitoring(interval=0.1)  # Monitor every 0.1 seconds

# Take memory snapshot
snapshot = memory_profiler.take_memory_snapshot(label="before_selection")

# Profile memory usage of a function
memory_stats = memory_profiler.profile_memory_usage(
    func=your_function,                 # Function to profile
    *args,                             # Function arguments
    **kwargs                           # Function keyword arguments
)

# Analyze memory patterns
patterns = memory_profiler.analyze_memory_patterns()

# Get memory timeline
timeline_df = memory_profiler.get_memory_timeline()

# Get memory snapshots
snapshots_df = memory_profiler.get_memory_snapshots()

# Use context manager
with memory_profiler.memory_context('selection_process'):
    selected_tickers = your_ticker_selection_function([2023], 15)

# Stop monitoring
memory_profiler.stop_memory_monitoring()
```

**Parameters**:
- `interval`: Monitoring interval in seconds
- `label`: Label for memory snapshots
- `func`: Function to profile for memory usage

**Returns**: Dictionary with memory usage statistics, leak detection results, and patterns

#### `selection_accuracy.py`
**Purpose**: Measure and analyze ticker selection accuracy and performance.

**Key Classes**:
- `SelectionAccuracy`: Main class for accuracy measurement

**API**:
```python
from src.validation.performance_profiling import SelectionAccuracy

# Initialize
accuracy = SelectionAccuracy()

# Calculate selection accuracy
accuracy_results = accuracy.calculate_selection_accuracy(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    benchmark_tickers=['RELIANCE.NS', 'TCS.NS'], # Benchmark tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    method='overlap'                             # 'overlap', 'performance', 'rank_correlation'
)

# Calculate hit rate
hit_rate = accuracy.calculate_hit_rate(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    actual_top_performers=['RELIANCE.NS', 'TCS.NS']  # Actual top performing tickers
)

# Calculate rank accuracy
rank_accuracy = accuracy.calculate_rank_accuracy(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    actual_rankings={                            # Dictionary mapping tickers to performance scores
        'RELIANCE.NS': 0.15,
        'TCS.NS': 0.12,
        'HDFCBANK.NS': 0.10
    }
)

# Calculate forward-looking accuracy
forward_accuracy = accuracy.calculate_forward_looking_accuracy(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    start_date='2023-01-01',                     # Selection date
    end_date='2023-12-31',                       # End date for evaluation
    lookforward_days=30                          # Days to look forward
)

# Generate accuracy report
report_df, summary = accuracy.generate_accuracy_report([
    {'hit_rate': 0.7, 'precision': 0.8, 'recall': 0.6},
    {'hit_rate': 0.65, 'precision': 0.75, 'recall': 0.55}
])
```

**Parameters**:
- `selected_tickers`: List of your selected ticker symbols
- `benchmark_tickers`: List of benchmark ticker symbols
- `actual_top_performers`: List of actual top performing tickers
- `actual_rankings`: Dictionary mapping tickers to performance scores
- `start_date`: Start date for analysis
- `end_date`: End date for analysis
- `method`: Accuracy calculation method
- `lookforward_days`: Days to look forward for performance evaluation

**Returns**: Dictionary with accuracy metrics, hit rates, precision, recall, and F1 scores

#### `performance_reporter.py`
**Purpose**: Generate comprehensive performance reports combining all metrics.

**Key Classes**:
- `PerformanceReporter`: Main class for report generation

**API**:
```python
from src.validation.performance_profiling import PerformanceReporter

# Initialize
reporter = PerformanceReporter()

# Generate comprehensive report
comprehensive_report = reporter.generate_comprehensive_report(
    execution_stats=[{...}],              # List of execution statistics
    memory_stats=[{...}],                 # List of memory statistics
    accuracy_stats=[{...}],               # List of accuracy statistics
    benchmark_results={...}               # Benchmark comparison results
)

# Generate execution report
execution_report = reporter.generate_execution_report([
    {
        'function': 'ticker_selection',
        'execution_time': 2.5,
        'memory_used': 45.2,
        'success': True
    }
])

# Generate memory report
memory_report = reporter.generate_memory_report([
    {
        'rss_mb': 120.5,
        'vms_mb': 200.3,
        'percent': 1.2
    }
])

# Generate accuracy report
accuracy_report = reporter.generate_accuracy_report([
    {
        'hit_rate': 0.7,
        'precision': 0.8,
        'recall': 0.6,
        'f1_score': 0.7
    }
])

# Save report to file
success = reporter.save_report_to_file(
    report=comprehensive_report,          # Report dictionary
    filename='performance_report.json',   # Output filename
    format='json'                         # 'json', 'csv', 'html'
)
```

**Parameters**:
- `execution_stats`: List of execution statistics dictionaries
- `memory_stats`: List of memory statistics dictionaries
- `accuracy_stats`: List of accuracy statistics dictionaries
- `benchmark_results`: Dictionary with benchmark comparison results
- `report`: Report dictionary to save
- `filename`: Output filename
- `format`: Output format ('json', 'csv', 'html')

**Returns**: Dictionary with comprehensive reports, performance summaries, and recommendations

---

## 🔄 backtesting/

**Purpose**: Perform out-of-sample testing, market regime analysis, and portfolio simulation.

### Files and APIs:

#### `walk_forward_validator.py`
**Purpose**: Implement walk-forward analysis for out-of-sample testing and validation.

**Key Classes**:
- `WalkForwardValidator`: Main class for walk-forward validation

**API**:
```python
from src.validation.backtesting import WalkForwardValidator

# Initialize with your ticker selector function
validator = WalkForwardValidator(ticker_selector_func=your_function)

# Perform walk-forward analysis
walk_forward_results = validator.walk_forward_analysis(
    start_year=2022,                      # Starting year
    end_year=2024,                        # Ending year
    train_months=12,                      # Training period in months
    test_months=3,                        # Testing period in months
    rebalance_frequency='monthly'         # Rebalancing frequency
)

# Perform rolling window analysis
rolling_results = validator.rolling_window_analysis(
    start_year=2022,                      # Starting year
    end_year=2024,                        # Ending year
    window_size_months=24,                # Window size in months
    step_size_months=3                    # Step size in months
)

# Perform expanding window analysis
expanding_results = validator.expanding_window_analysis(
    start_year=2022,                      # Starting year
    end_year=2024,                        # Ending year
    min_window_months=12,                 # Minimum window size
    step_size_months=3                    # Step size in months
)

# Get walk-forward history
history_df = validator.get_walk_forward_history()
```

**Parameters**:
- `start_year`: Starting year for analysis
- `end_year`: Ending year for analysis
- `train_months`: Number of months for training period
- `test_months`: Number of months for testing period
- `window_size_months`: Size of rolling window in months
- `step_size_months`: Step size for rolling/expanding windows
- `min_window_months`: Minimum window size for expanding analysis
- `rebalance_frequency`: Rebalancing frequency ('daily', 'monthly', 'quarterly')

**Returns**: Dictionary with walk-forward analysis results, performance metrics, and consistency scores

#### `market_regime_analyzer.py`
**Purpose**: Analyze performance across different market regimes and conditions.

**Key Classes**:
- `MarketRegimeAnalyzer`: Main class for market regime analysis

**API**:
```python
from src.validation.backtesting import MarketRegimeAnalyzer

# Initialize
analyzer = MarketRegimeAnalyzer()

# Analyze market regimes
regime_results = analyzer.analyze_market_regimes(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    benchmark_ticker='^NSEI'                     # Benchmark ticker for regime classification
)

# Analyze volatility regimes
volatility_results = analyzer.analyze_volatility_regimes(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    benchmark_ticker='^NSEI'                     # Benchmark ticker
)

# Analyze sector rotation
sector_results = analyzer.analyze_sector_rotation(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Your selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31'                        # End date
)

# Get regime analysis history
history_df = analyzer.get_regime_analysis_history()
```

**Parameters**:
- `selected_tickers`: List of your selected ticker symbols
- `start_date`: Start date for analysis
- `end_date`: End date for analysis
- `benchmark_ticker`: Benchmark ticker for regime classification (default: '^NSEI')

**Returns**: Dictionary with regime analysis results, performance by regime, and consistency metrics

#### `portfolio_simulator.py`
**Purpose**: Simulate actual trading performance with realistic constraints and costs.

**Key Classes**:
- `PortfolioSimulator`: Main class for portfolio simulation

**API**:
```python
from src.validation.backtesting import PortfolioSimulator

# Initialize with simulation parameters
simulator = PortfolioSimulator(
    initial_capital=1000000,              # Initial portfolio capital
    transaction_cost=0.001,               # Transaction cost (0.1%)
    market_impact=0.0005                  # Market impact cost (0.05%)
)

# Simulate portfolio trading
simulation_results = simulator.simulate_portfolio_trading(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    rebalance_frequency='monthly',               # Rebalancing frequency
    rebalance_threshold=0.05                     # Rebalancing threshold
)

# Simulate with different cost scenarios
cost_scenarios = [
    {'name': 'low_cost', 'transaction_cost': 0.0005, 'market_impact': 0.0002},
    {'name': 'medium_cost', 'transaction_cost': 0.001, 'market_impact': 0.0005},
    {'name': 'high_cost', 'transaction_cost': 0.002, 'market_impact': 0.001}
]

cost_analysis = simulator.simulate_with_transaction_costs(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    cost_scenarios=cost_scenarios                # Cost scenarios to test
)

# Simulate with liquidity constraints
liquidity_analysis = simulator.simulate_liquidity_constraints(
    selected_tickers=['RELIANCE.NS', 'TCS.NS'],  # Selected tickers
    start_date='2023-01-01',                     # Start date
    end_date='2023-12-31',                       # End date
    liquidity_constraints={                      # Liquidity constraints
        'max_position_size': 0.1,               # Maximum 10% position size
        'min_liquidity': 1000000                # Minimum daily liquidity
    }
)

# Get simulation history
history_df = simulator.get_simulation_history()
```

**Parameters**:
- `initial_capital`: Initial portfolio capital amount
- `transaction_cost`: Transaction cost as percentage of trade value
- `market_impact`: Market impact cost as percentage of trade value
- `selected_tickers`: List of selected ticker symbols
- `start_date`: Start date for simulation
- `end_date`: End date for simulation
- `rebalance_frequency`: Rebalancing frequency ('daily', 'monthly', 'quarterly')
- `rebalance_threshold`: Threshold for triggering rebalancing
- `cost_scenarios`: List of cost scenario dictionaries
- `liquidity_constraints`: Dictionary with liquidity constraints

**Returns**: Dictionary with simulation results, performance metrics, transaction costs, and trading statistics

---

## 📈 visualization/

**Purpose**: Create comprehensive visualizations for performance analysis and results presentation.

### Files and APIs:

#### `performance_plots.py`
**Purpose**: Create performance comparison and validation plots.

**Key Classes**:
- `PerformancePlotter`: Main class for performance visualization

**API**:
```python
from src.validation.visualization import PerformancePlotter

# Initialize
plotter = PerformancePlotter(style='seaborn-v0_8', figsize=(12, 8))

# Create validation plots
plots = plotter.create_validation_plots(
    benchmark_comparison={...},           # Benchmark comparison results
    walk_forward_results={...}            # Walk-forward analysis results
)

# Plot benchmark comparison
fig = plotter.plot_benchmark_comparison({
    'your_performance': {...},
    'benchmark_comparisons': {...}
})

# Plot walk-forward analysis
fig = plotter.plot_walk_forward_analysis({
    'detailed_results': [...],
    'summary': {...}
})

# Plot risk-return scatter
fig = plotter.plot_risk_return_scatter({
    'your_performance': {...},
    'benchmark_comparisons': {...}
})

# Plot drawdown comparison
fig = plotter.plot_drawdown_comparison({
    'your_performance': {...},
    'benchmark_comparisons': {...}
})

# Plot performance over time
fig = plotter.plot_performance_over_time({
    'detailed_results': [...]
})

# Save all plots
saved_files = plotter.save_all_plots(
    plots={...},                          # Dictionary of plots
    output_dir="plots"                    # Output directory
)
```

**Parameters**:
- `style`: Matplotlib style to use
- `figsize`: Default figure size tuple
- `benchmark_comparison`: Dictionary with benchmark comparison results
- `walk_forward_results`: Dictionary with walk-forward analysis results
- `plots`: Dictionary of plot information
- `output_dir`: Output directory for saved plots

**Returns**: Matplotlib figure objects and file paths for saved plots

#### `indicator_plots.py`
**Purpose**: Create plots for technical indicators and strategy analysis.

**Key Classes**:
- `IndicatorPlotter`: Main class for indicator visualization

**API**:
```python
from src.validation.visualization import IndicatorPlotter

# Initialize
plotter = IndicatorPlotter(style='seaborn-v0_8', figsize=(12, 8))

# Plot indicator performance
fig = plotter.plot_indicator_performance({
    'RSI': {'hit_rate': 0.7, 'accuracy': 0.8, 'frequency': 15, 'performance_score': 0.75},
    'MACD': {'hit_rate': 0.65, 'accuracy': 0.75, 'frequency': 12, 'performance_score': 0.70}
})

# Plot indicator correlation
fig = plotter.plot_indicator_correlation(
    correlation_data=pd.DataFrame({...})  # Correlation matrix DataFrame
)

# Plot indicator signals
fig = plotter.plot_indicator_signals(
    price_data=pd.DataFrame({...}),       # Price data DataFrame
    indicator_data={...},                 # Indicator data dictionary
    ticker='RELIANCE.NS'                  # Ticker symbol
)

# Plot strategy performance
fig = plotter.plot_strategy_performance({
    'momentum': {'total_return': 0.15, 'sharpe_ratio': 1.2, 'win_rate': 0.65, 'max_drawdown': -0.08},
    'mean_reversion': {'total_return': 0.12, 'sharpe_ratio': 1.0, 'win_rate': 0.60, 'max_drawdown': -0.10}
})

# Plot indicator distribution
fig = plotter.plot_indicator_distribution({
    'RSI': [30, 45, 60, 70, 80, 90, ...],  # RSI values
    'MACD': [-0.5, -0.2, 0.1, 0.3, 0.8, ...]  # MACD values
})

# Plot signal analysis
fig = plotter.plot_signal_analysis({
    'signal_frequency': {'buy': 25, 'sell': 20, 'hold': 55},
    'signal_accuracy': {'buy': 0.7, 'sell': 0.65, 'hold': 0.8}
})

# Save plots
saved_files = plotter.save_plots(
    plots={...},                          # Dictionary of plots
    output_dir="indicator_plots"          # Output directory
)
```

**Parameters**:
- `indicator_results`: Dictionary with indicator performance results
- `correlation_data`: Pandas DataFrame with correlation matrix
- `price_data`: Pandas DataFrame with price data
- `indicator_data`: Dictionary with indicator values
- `ticker`: Ticker symbol for analysis
- `strategy_results`: Dictionary with strategy performance results
- `indicator_values`: Dictionary with indicator value distributions
- `signal_data`: Dictionary with signal analysis data

**Returns**: Matplotlib figure objects and file paths for saved plots

#### `benchmark_plots.py`
**Purpose**: Create specialized plots for benchmark comparison and market analysis.

**Key Classes**:
- `BenchmarkPlotter`: Main class for benchmark visualization

**API**:
```python
from src.validation.visualization import BenchmarkPlotter

# Initialize
plotter = BenchmarkPlotter(style='seaborn-v0_8', figsize=(12, 8))

# Plot detailed benchmark comparison
fig = plotter.plot_benchmark_comparison_detailed({
    'your_performance': {...},
    'benchmark_comparisons': {...}
})

# Plot correlation heatmap
fig = plotter.plot_benchmark_correlation_heatmap(
    correlation_data=pd.DataFrame({...})  # Correlation matrix DataFrame
)

# Plot rolling performance
fig = plotter.plot_rolling_performance({
    'dates': [...],                       # List of dates
    'your_returns': [...],                # Your portfolio returns
    'benchmark_returns': {...}            # Benchmark returns dictionary
})

# Plot performance attribution
fig = plotter.plot_benchmark_attribution({
    'sector_attribution': {...},          # Sector attribution data
    'factor_attribution': {...},          # Factor attribution data
    'selection_effect': 0.05,             # Selection effect
    'allocation_effect': 0.02,            # Allocation effect
    'risk_attribution': {...}             # Risk attribution data
})

# Plot benchmark consistency
fig = plotter.plot_benchmark_consistency({
    'win_rates': {...},                   # Win rates over time
    'performance_std': {...},             # Performance standard deviations
    'outperformance_frequency': {...},    # Outperformance frequencies
    'risk_adjusted_consistency': {...}    # Risk-adjusted consistency scores
})

# Save benchmark plots
saved_files = plotter.save_benchmark_plots(
    plots={...},                          # Dictionary of plots
    output_dir="benchmark_plots"          # Output directory
)
```

**Parameters**:
- `benchmark_data`: Dictionary with benchmark comparison data
- `correlation_data`: Pandas DataFrame with correlation matrix
- `rolling_data`: Dictionary with rolling performance data
- `attribution_data`: Dictionary with performance attribution data
- `consistency_data`: Dictionary with consistency analysis data

**Returns**: Matplotlib figure objects and file paths for saved plots

---

## 🚀 Main ValidationSuite Class

**Purpose**: Main class that ties all validation components together.

**API**:
```python
from src.validation import ValidationSuite

# Initialize with your ticker selector function
validator = ValidationSuite(ticker_selector_func=your_function)

# Run complete validation
results = validator.run_complete_validation(
    years=[2022, 2023, 2024],            # List of years to validate
    top_n_tickers=15,                    # Number of top tickers to select
    rebalancing_months=1,                # Rebalancing frequency in months
    train_months=12,                     # Training period for walk-forward
    test_months=3,                       # Testing period for walk-forward
    window_size_months=24,               # Window size for rolling analysis
    rebalance_frequency='monthly'        # Rebalancing frequency
)

# Run individual validation components
performance_stats = validator._run_performance_profiling([2023], top_n_tickers=15)
benchmark_comparison = validator._run_benchmark_comparison([2023], top_n_tickers=15)
walk_forward_results = validator._run_walk_forward_validation([2023], top_n_tickers=15)

# Generate visualizations
visualizations = validator._generate_visualizations(
    benchmark_comparison, walk_forward_results, {}
)

# Save validation results
validator.save_validation_results(results, "validation_results.json")

# Get validation history
history_df = validator.get_validation_history()
```

**Parameters**:
- `ticker_selector_func`: Your ticker selection function
- `years`: List of years to analyze
- `top_n_tickers`: Number of top tickers to select
- `rebalancing_months`: Rebalancing frequency in months
- `train_months`: Training period for walk-forward analysis
- `test_months`: Testing period for walk-forward analysis
- `window_size_months`: Window size for rolling analysis
- `rebalance_frequency`: Rebalancing frequency ('daily', 'monthly', 'quarterly')

**Returns**: Dictionary with comprehensive validation results, performance metrics, and analysis

---

## Dependencies

Required packages:

```
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.9.0
psutil>=5.9.0
```

Install dependencies:

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy psutil
```

## Quick Start Example

```python
from src.validation import ValidationSuite

def your_ticker_selector(years, top_n_tickers=15, **kwargs):
    # Your ticker selection logic here
    return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', ...]

# Initialize validation suite
validator = ValidationSuite(ticker_selector_func=your_ticker_selector)

# Run complete validation
results = validator.run_complete_validation(
    years=[2023, 2024],
    top_n_tickers=15,
    rebalancing_months=1
)

# Save results
validator.save_validation_results(results, "my_validation_results.json")

print("Validation completed successfully!")
print(f"Results saved to: my_validation_results.json")
```

This comprehensive API documentation provides detailed information about each module, its purpose, available methods, parameters, and return values. Each folder serves a specific purpose in the validation process, and all components work together to provide comprehensive validation of your ticker selection algorithm.
