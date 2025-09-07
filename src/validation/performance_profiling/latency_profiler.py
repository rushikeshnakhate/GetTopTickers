"""
Performance Profiler for execution time and memory tracking.

This module provides comprehensive performance profiling capabilities to measure
execution time, memory usage, and compare performance against competitors.
"""

import time
import functools
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, List, Optional
import logging
import threading
from contextlib import contextmanager
import gc

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profile execution performance of ticker selection and related operations."""
    
    def __init__(self):
        """Initialize the PerformanceProfiler."""
        self.execution_stats = []
        self.memory_stats = []
        self.competitor_benchmarks = {}
        self._lock = threading.Lock()
        
    def profile_execution_time(self, func: Callable) -> Callable:
        """
        Decorator to profile execution time and memory usage.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Force garbage collection before measurement
            gc.collect()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = None
                success = False
                error_msg = str(e)
                logger.error(f"Error in profiled function {func.__name__}: {error_msg}")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Log performance statistics
            with self._lock:
                self._log_performance({
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'memory_used': memory_used,
                    'start_memory': start_memory,
                    'end_memory': end_memory,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs),
                    'success': success,
                    'error_message': error_msg,
                    'timestamp': time.time()
                })
            
            return result
        return wrapper
    
    def benchmark_vs_competitors(self, ticker_count: int, 
                                time_periods: List[str], 
                                competitor_functions: Optional[Dict[str, Callable]] = None) -> Dict:
        """
        Benchmark against competitor tools and methods.
        
        Args:
            ticker_count: Number of tickers to select
            time_periods: List of time periods to test
            competitor_functions: Dictionary of competitor functions to test
            
        Returns:
            Dictionary with benchmark results
        """
        if competitor_functions is None:
            competitor_functions = self._get_default_competitors()
        
        results = {}
        
        for period in time_periods:
            logger.info(f"Benchmarking for period: {period}")
            period_results = {}
            
            # Test your library
            try:
                start_time = time.time()
                your_result = self._run_your_selection(ticker_count, period)
                your_time = time.time() - start_time
                your_accuracy = self._calculate_accuracy(your_result, period)
                your_memory = self._get_memory_usage()
                
                period_results['your_library'] = {
                    'execution_time': your_time,
                    'memory_usage': your_memory,
                    'accuracy': your_accuracy,
                    'result_count': len(your_result) if your_result else 0,
                    'success': True
                }
            except Exception as e:
                logger.error(f"Error in your library benchmark: {str(e)}")
                period_results['your_library'] = {
                    'execution_time': float('inf'),
                    'memory_usage': 0,
                    'accuracy': 0,
                    'result_count': 0,
                    'success': False,
                    'error': str(e)
                }
            
            # Test competitors
            for competitor_name, competitor_func in competitor_functions.items():
                try:
                    start_time = time.time()
                    competitor_result = competitor_func(ticker_count, period)
                    competitor_time = time.time() - start_time
                    competitor_accuracy = self._calculate_accuracy(competitor_result, period)
                    competitor_memory = self._get_memory_usage()
                    
                    period_results[competitor_name] = {
                        'execution_time': competitor_time,
                        'memory_usage': competitor_memory,
                        'accuracy': competitor_accuracy,
                        'result_count': len(competitor_result) if competitor_result else 0,
                        'success': True
                    }
                except Exception as e:
                    logger.error(f"Error in {competitor_name} benchmark: {str(e)}")
                    period_results[competitor_name] = {
                        'execution_time': float('inf'),
                        'memory_usage': 0,
                        'accuracy': 0,
                        'result_count': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate relative performance
            if 'your_library' in period_results and period_results['your_library']['success']:
                your_time = period_results['your_library']['execution_time']
                your_accuracy = period_results['your_library']['accuracy']
                
                for competitor_name, competitor_data in period_results.items():
                    if competitor_name != 'your_library' and competitor_data['success']:
                        competitor_time = competitor_data['execution_time']
                        competitor_accuracy = competitor_data['accuracy']
                        
                        period_results[competitor_name]['speedup'] = competitor_time / your_time
                        period_results[competitor_name]['accuracy_improvement'] = your_accuracy - competitor_accuracy
            
            results[period] = period_results
        
        # Calculate summary statistics
        summary = self._calculate_benchmark_summary(results)
        results['summary'] = summary
        
        return results
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Profile memory usage of a specific function.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with memory usage statistics
        """
        # Force garbage collection
        gc.collect()
        
        # Get initial memory
        initial_memory = self._get_memory_usage()
        peak_memory = initial_memory
        
        # Monitor memory during execution
        def monitor_memory():
            nonlocal peak_memory
            while True:
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.01)  # Check every 10ms
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        
        # Stop monitoring and get final memory
        final_memory = self._get_memory_usage()
        
        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'peak_memory': peak_memory,
            'memory_used': final_memory - initial_memory,
            'peak_memory_used': peak_memory - initial_memory,
            'success': success,
            'error_message': error_msg
        }
    
    def generate_performance_report(self) -> pd.DataFrame:
        """
        Generate comprehensive performance report.
        
        Returns:
            DataFrame with performance statistics
        """
        if not self.execution_stats:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.execution_stats)
        
        # Add summary statistics
        summary_stats = {
            'total_functions_called': len(df),
            'total_execution_time': df['execution_time'].sum(),
            'average_execution_time': df['execution_time'].mean(),
            'median_execution_time': df['execution_time'].median(),
            'max_execution_time': df['execution_time'].max(),
            'min_execution_time': df['execution_time'].min(),
            'total_memory_used': df['memory_used'].sum(),
            'average_memory_used': df['memory_used'].mean(),
            'success_rate': df['success'].mean(),
            'functions_by_name': df['function'].value_counts().to_dict()
        }
        
        return df, summary_stats
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary statistics.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.execution_stats:
            return {}
        
        df = pd.DataFrame(self.execution_stats)
        
        return {
            'total_calls': len(df),
            'successful_calls': df['success'].sum(),
            'failed_calls': (~df['success']).sum(),
            'success_rate': df['success'].mean(),
            'total_execution_time': df['execution_time'].sum(),
            'average_execution_time': df['execution_time'].mean(),
            'median_execution_time': df['execution_time'].median(),
            'max_execution_time': df['execution_time'].max(),
            'total_memory_used': df['memory_used'].sum(),
            'average_memory_used': df['memory_used'].mean(),
            'max_memory_used': df['memory_used'].max(),
            'function_breakdown': df.groupby('function').agg({
                'execution_time': ['count', 'mean', 'sum'],
                'memory_used': ['mean', 'sum'],
                'success': 'mean'
            }).to_dict()
        }
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _log_performance(self, stats: Dict):
        """
        Log performance statistics.
        
        Args:
            stats: Performance statistics dictionary
        """
        self.execution_stats.append(stats.copy())
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.execution_stats) > 1000:
            self.execution_stats = self.execution_stats[-1000:]
    
    def _run_your_selection(self, ticker_count: int, period: str) -> List[str]:
        """
        Run your ticker selection method.
        
        Args:
            ticker_count: Number of tickers to select
            period: Time period
            
        Returns:
            List of selected tickers
        """
        # This would integrate with your actual ticker selection logic
        # For now, return a mock result
        try:
            from ...main import run_pyport_ticker_selector
            year = int(period.split('-')[0]) if '-' in period else int(period)
            result = run_pyport_ticker_selector(years=[year], top_n_tickers=ticker_count)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Error running your selection: {str(e)}")
            return []
    
    def _calculate_accuracy(self, selected_tickers: List[str], period: str) -> float:
        """
        Calculate selection accuracy.
        
        Args:
            selected_tickers: List of selected tickers
            period: Time period
            
        Returns:
            Accuracy score (0-1)
        """
        if not selected_tickers:
            return 0.0
        
        # This would implement actual accuracy calculation
        # For now, return a mock accuracy
        return np.random.uniform(0.6, 0.9)
    
    def _get_default_competitors(self) -> Dict[str, Callable]:
        """
        Get default competitor functions for benchmarking.
        
        Returns:
            Dictionary of competitor functions
        """
        def random_selection(ticker_count: int, period: str) -> List[str]:
            """Mock random selection competitor."""
            time.sleep(0.1)  # Simulate processing time
            return [f"TICKER{i}.NS" for i in range(1, ticker_count + 1)]
        
        def simple_momentum(ticker_count: int, period: str) -> List[str]:
            """Mock simple momentum strategy."""
            time.sleep(0.2)  # Simulate processing time
            return [f"MOMENTUM{i}.NS" for i in range(1, ticker_count + 1)]
        
        def technical_analysis(ticker_count: int, period: str) -> List[str]:
            """Mock technical analysis strategy."""
            time.sleep(0.3)  # Simulate processing time
            return [f"TECH{i}.NS" for i in range(1, ticker_count + 1)]
        
        return {
            'random_selection': random_selection,
            'simple_momentum': simple_momentum,
            'technical_analysis': technical_analysis
        }
    
    def _calculate_benchmark_summary(self, results: Dict) -> Dict:
        """
        Calculate summary statistics for benchmark results.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_periods': len([k for k in results.keys() if k != 'summary']),
            'your_library_wins': 0,
            'your_library_avg_speedup': 0,
            'your_library_avg_accuracy': 0,
            'competitor_comparisons': {}
        }
        
        speedups = []
        accuracies = []
        
        for period, period_results in results.items():
            if period == 'summary':
                continue
                
            if 'your_library' in period_results and period_results['your_library']['success']:
                your_accuracy = period_results['your_library']['accuracy']
                accuracies.append(your_accuracy)
                
                for competitor_name, competitor_data in period_results.items():
                    if competitor_name != 'your_library' and competitor_data['success']:
                        if 'speedup' in competitor_data:
                            speedups.append(competitor_data['speedup'])
                        
                        if competitor_name not in summary['competitor_comparisons']:
                            summary['competitor_comparisons'][competitor_name] = {
                                'wins': 0,
                                'total_comparisons': 0,
                                'avg_speedup': 0
                            }
                        
                        summary['competitor_comparisons'][competitor_name]['total_comparisons'] += 1
                        
                        if competitor_data.get('speedup', 0) > 1:
                            summary['your_library_wins'] += 1
                            summary['competitor_comparisons'][competitor_name]['wins'] += 1
        
        if speedups:
            summary['your_library_avg_speedup'] = np.mean(speedups)
        
        if accuracies:
            summary['your_library_avg_accuracy'] = np.mean(accuracies)
        
        # Calculate win rates for competitors
        for competitor_name, comp_data in summary['competitor_comparisons'].items():
            if comp_data['total_comparisons'] > 0:
                comp_data['win_rate'] = comp_data['wins'] / comp_data['total_comparisons']
        
        return summary
    
    def clear_stats(self):
        """Clear all performance statistics."""
        with self._lock:
            self.execution_stats.clear()
            self.memory_stats.clear()
        logger.info("Performance statistics cleared")
    
    @contextmanager
    def profile_context(self, operation_name: str):
        """
        Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            success = True
            error_msg = None
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            with self._lock:
                self._log_performance({
                    'function': operation_name,
                    'execution_time': execution_time,
                    'memory_used': memory_used,
                    'start_memory': start_memory,
                    'end_memory': end_memory,
                    'args_count': 0,
                    'kwargs_count': 0,
                    'success': success,
                    'error_message': error_msg,
                    'timestamp': time.time()
                })
