"""
Validation module for PyPortTickerSelector.

This module provides comprehensive validation, benchmarking, and performance
measurement capabilities to address reviewer concerns about:
- Performance benchmarking against competitors
- Market benchmark comparison
- Selection accuracy validation
- Out-of-sample testing
- Statistical significance testing
"""

from .benchmark_comparison.market_benchmark_fetcher import MarketBenchmarkFetcher
from .benchmark_comparison.benchmark_comparator import BenchmarkComparator
from .performance_profiling.latency_profiler import PerformanceProfiler
from .backtesting.walk_forward_validator import WalkForwardValidator
from .visualization.performance_plots import PerformancePlotter

__all__ = [
    'MarketBenchmarkFetcher',
    'BenchmarkComparator', 
    'PerformanceProfiler',
    'WalkForwardValidator',
    'PerformancePlotter'
]
