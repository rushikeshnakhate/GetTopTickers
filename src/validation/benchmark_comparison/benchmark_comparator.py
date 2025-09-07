"""
Benchmark Comparator for comparing PyPortTickerSelector results against market benchmarks.

This module provides comprehensive comparison capabilities to evaluate how well
the ticker selection algorithm performs compared to various market benchmarks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf

logger = logging.getLogger(__name__)


class BenchmarkComparator:
    """Compare PyPortTickerSelector results against market benchmarks."""
    
    def __init__(self, market_fetcher, performance_profiler=None):
        """
        Initialize the BenchmarkComparator.
        
        Args:
            market_fetcher: MarketBenchmarkFetcher instance
            performance_profiler: PerformanceProfiler instance (optional)
        """
        self.market_fetcher = market_fetcher
        self.profiler = performance_profiler
        self.comparison_results = []
        
    def compare_with_benchmarks(self, selected_tickers: List[str], 
                              start_date: str, end_date: str,
                              benchmarks: List[str] = None) -> Dict:
        """
        Compare selected tickers with multiple benchmarks.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for comparison
            end_date: End date for comparison
            benchmarks: List of benchmarks to compare against
            
        Returns:
            Dictionary with benchmark comparison results
        """
        if benchmarks is None:
            benchmarks = ['nse_50', 'nse_100', 'nse_500']
        
        if not selected_tickers:
            logger.warning("No selected tickers provided for comparison")
            return {}
        
        logger.info(f"Comparing {len(selected_tickers)} selected tickers with benchmarks")
        
        # Get your portfolio performance
        your_performance = self._calculate_portfolio_performance(
            selected_tickers, start_date, end_date
        )
        
        if not your_performance:
            logger.error("Failed to calculate portfolio performance")
            return {}
        
        # Compare with benchmarks
        comparisons = {}
        for benchmark in benchmarks:
            try:
                benchmark_perf = self.market_fetcher.get_benchmark_performance(
                    benchmark, start_date, end_date
                )
                
                comparison = self._compare_performances(
                    your_performance, benchmark_perf, benchmark
                )
                
                # Add statistical significance testing
                statistical_tests = self._perform_statistical_tests(
                    selected_tickers, benchmark, start_date, end_date
                )
                comparison['statistical_tests'] = statistical_tests
                
                comparisons[benchmark] = comparison
                
            except Exception as e:
                logger.error(f"Error comparing with benchmark {benchmark}: {str(e)}")
                comparisons[benchmark] = {
                    'error': str(e),
                    'benchmark': benchmark
                }
        
        # Store results
        self.comparison_results.append({
            'timestamp': datetime.now(),
            'selected_tickers': selected_tickers,
            'start_date': start_date,
            'end_date': end_date,
            'comparisons': comparisons
        })
        
        return {
            'your_performance': your_performance,
            'benchmark_comparisons': comparisons,
            'summary': self._generate_comparison_summary(comparisons)
        }
    
    def selection_accuracy_analysis(self, selected_tickers: List[str], 
                                  benchmark: str = 'nse_50',
                                  start_date: str = None, end_date: str = None) -> Dict:
        """
        Analyze how many selected tickers overlap with top performers.
        
        Args:
            selected_tickers: List of selected tickers
            benchmark: Benchmark to compare against
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with accuracy analysis
        """
        if not selected_tickers:
            return {
                'overlap_count': 0,
                'total_selected': 0,
                'accuracy_percentage': 0.0
            }
        
        try:
            # Get benchmark tickers
            benchmark_tickers = self.market_fetcher.get_benchmark_tickers(benchmark)
            
            # Calculate which benchmark tickers performed best
            if start_date and end_date:
                benchmark_performance = self._rank_tickers_by_performance(
                    benchmark_tickers, start_date, end_date
                )
            else:
                # Use a default ranking method
                benchmark_performance = self._get_default_benchmark_ranking(benchmark_tickers)
            
            # Get top performers (same number as selected)
            top_performers = benchmark_performance.head(len(selected_tickers))
            
            # Calculate overlap
            overlap_count = len(set(selected_tickers) & set(top_performers.index))
            accuracy = overlap_count / len(selected_tickers)
            
            # Calculate additional metrics
            missed_opportunities = list(set(top_performers.index) - set(selected_tickers))
            false_positives = list(set(selected_tickers) - set(top_performers.index))
            
            return {
                'overlap_count': overlap_count,
                'total_selected': len(selected_tickers),
                'accuracy_percentage': accuracy * 100,
                'selected_tickers': selected_tickers,
                'top_performers': top_performers.index.tolist(),
                'missed_opportunities': missed_opportunities,
                'false_positives': false_positives,
                'precision': overlap_count / len(selected_tickers) if selected_tickers else 0,
                'recall': overlap_count / len(top_performers) if top_performers else 0,
                'f1_score': self._calculate_f1_score(overlap_count, len(selected_tickers), len(top_performers))
            }
            
        except Exception as e:
            logger.error(f"Error in selection accuracy analysis: {str(e)}")
            return {
                'error': str(e),
                'overlap_count': 0,
                'total_selected': len(selected_tickers),
                'accuracy_percentage': 0.0
            }
    
    def calculate_risk_adjusted_metrics(self, selected_tickers: List[str],
                                      benchmark_tickers: List[str],
                                      start_date: str, end_date: str) -> Dict:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            selected_tickers: List of selected tickers
            benchmark_tickers: List of benchmark tickers
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        try:
            # Get returns for both portfolios
            your_returns = self._get_portfolio_returns(selected_tickers, start_date, end_date)
            benchmark_returns = self._get_portfolio_returns(benchmark_tickers, start_date, end_date)
            
            if your_returns.empty or benchmark_returns.empty:
                return {'error': 'Insufficient data for risk-adjusted analysis'}
            
            # Align returns
            common_dates = your_returns.index.intersection(benchmark_returns.index)
            your_returns = your_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]
            
            # Calculate metrics
            your_annual_return = (1 + your_returns.mean()) ** 252 - 1
            benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
            
            your_volatility = your_returns.std() * (252 ** 0.5)
            benchmark_volatility = benchmark_returns.std() * (252 ** 0.5)
            
            # Sharpe ratios
            your_sharpe = your_annual_return / your_volatility if your_volatility > 0 else 0
            benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Information ratio
            excess_returns = your_returns - benchmark_returns
            tracking_error = excess_returns.std() * (252 ** 0.5)
            information_ratio = (your_annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
            
            # Beta calculation
            covariance = np.cov(your_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha calculation
            alpha = your_annual_return - (0.05 + beta * (benchmark_annual_return - 0.05))  # Assuming 5% risk-free rate
            
            return {
                'your_annual_return': your_annual_return,
                'benchmark_annual_return': benchmark_annual_return,
                'your_volatility': your_volatility,
                'benchmark_volatility': benchmark_volatility,
                'your_sharpe_ratio': your_sharpe,
                'benchmark_sharpe_ratio': benchmark_sharpe,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error,
                'excess_return': your_annual_return - benchmark_annual_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_portfolio_performance(self, tickers: List[str], 
                                       start_date: str, end_date: str) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get portfolio returns
            returns = self._get_portfolio_returns(tickers, start_date, end_date)
            
            if returns.empty:
                return {}
            
            # Calculate metrics using the same method as MarketBenchmarkFetcher
            return self.market_fetcher._calculate_metrics(returns)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {str(e)}")
            return {}
    
    def _get_portfolio_returns(self, tickers: List[str], start_date: str, end_date: str) -> pd.Series:
        """
        Get portfolio returns for given tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Series of portfolio returns
        """
        try:
            # Fetch data for all tickers
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if data.empty:
                return pd.Series(dtype=float)
            
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data.xs('Adj Close', level=1, axis=1)
            else:
                close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
            
            # Clean data
            close_prices = close_prices.dropna(how='all')
            
            if close_prices.empty:
                return pd.Series(dtype=float)
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Equal weight portfolio
            portfolio_returns = returns.mean(axis=1)
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {str(e)}")
            return pd.Series(dtype=float)
    
    def _compare_performances(self, your_perf: Dict, benchmark_perf: Dict, 
                            benchmark_name: str) -> Dict:
        """
        Compare two performance dictionaries.
        
        Args:
            your_perf: Your portfolio performance
            benchmark_perf: Benchmark performance
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with comparison results
        """
        return {
            'benchmark': benchmark_name,
            'your_return': your_perf.get('annual_return', 0),
            'benchmark_return': benchmark_perf.get('annual_return', 0),
            'outperformance': your_perf.get('annual_return', 0) - benchmark_perf.get('annual_return', 0),
            'outperformance_pct': ((your_perf.get('annual_return', 0) / benchmark_perf.get('annual_return', 1)) - 1) * 100 if benchmark_perf.get('annual_return', 0) != 0 else 0,
            'your_sharpe': your_perf.get('sharpe_ratio', 0),
            'benchmark_sharpe': benchmark_perf.get('sharpe_ratio', 0),
            'sharpe_improvement': your_perf.get('sharpe_ratio', 0) - benchmark_perf.get('sharpe_ratio', 0),
            'your_max_dd': your_perf.get('max_drawdown', 0),
            'benchmark_max_dd': benchmark_perf.get('max_drawdown', 0),
            'drawdown_improvement': benchmark_perf.get('max_drawdown', 0) - your_perf.get('max_drawdown', 0),
            'your_volatility': your_perf.get('volatility', 0),
            'benchmark_volatility': benchmark_perf.get('volatility', 0),
            'volatility_ratio': your_perf.get('volatility', 0) / benchmark_perf.get('volatility', 1) if benchmark_perf.get('volatility', 0) != 0 else 0
        }
    
    def _perform_statistical_tests(self, selected_tickers: List[str], 
                                 benchmark: str, start_date: str, end_date: str) -> Dict:
        """
        Perform statistical significance tests.
        
        Args:
            selected_tickers: List of selected tickers
            benchmark: Benchmark name
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with statistical test results
        """
        try:
            # Get returns for both portfolios
            your_returns = self._get_portfolio_returns(selected_tickers, start_date, end_date)
            benchmark_returns = self.market_fetcher.get_benchmark_returns_series(
                benchmark, start_date, end_date
            )
            
            if your_returns.empty or benchmark_returns.empty:
                return {'error': 'Insufficient data for statistical tests'}
            
            # Align returns
            common_dates = your_returns.index.intersection(benchmark_returns.index)
            your_returns = your_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]
            
            # T-test for mean difference
            t_stat, t_pvalue = stats.ttest_ind(your_returns, benchmark_returns)
            
            # Wilcoxon signed-rank test (non-parametric)
            try:
                wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(your_returns, benchmark_returns)
            except ValueError:
                wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan
            
            # Correlation test
            correlation, corr_pvalue = stats.pearsonr(your_returns, benchmark_returns)
            
            return {
                't_test': {
                    'statistic': t_stat,
                    'p_value': t_pvalue,
                    'significant': t_pvalue < 0.05
                },
                'wilcoxon_test': {
                    'statistic': wilcoxon_stat,
                    'p_value': wilcoxon_pvalue,
                    'significant': wilcoxon_pvalue < 0.05 if not np.isnan(wilcoxon_pvalue) else False
                },
                'correlation': {
                    'coefficient': correlation,
                    'p_value': corr_pvalue,
                    'significant': corr_pvalue < 0.05
                },
                'sample_size': len(common_dates)
            }
            
        except Exception as e:
            logger.error(f"Error performing statistical tests: {str(e)}")
            return {'error': str(e)}
    
    def _rank_tickers_by_performance(self, tickers: List[str], 
                                   start_date: str, end_date: str) -> pd.Series:
        """
        Rank tickers by their performance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Series with tickers ranked by performance
        """
        try:
            # Fetch data for all tickers
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if data.empty:
                return pd.Series(dtype=float)
            
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data.xs('Adj Close', level=1, axis=1)
            else:
                close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
            
            # Calculate total returns
            total_returns = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
            
            # Sort by performance (descending)
            ranked_returns = total_returns.sort_values(ascending=False)
            
            return ranked_returns
            
        except Exception as e:
            logger.error(f"Error ranking tickers by performance: {str(e)}")
            return pd.Series(dtype=float)
    
    def _get_default_benchmark_ranking(self, tickers: List[str]) -> pd.Series:
        """
        Get default benchmark ranking (mock implementation).
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Series with default rankings
        """
        # This would implement a default ranking method
        # For now, return a random ranking
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.uniform(-0.3, 0.5, len(tickers))
        return pd.Series(random_returns, index=tickers).sort_values(ascending=False)
    
    def _calculate_f1_score(self, overlap_count: int, selected_count: int, 
                          top_performers_count: int) -> float:
        """
        Calculate F1 score.
        
        Args:
            overlap_count: Number of overlapping tickers
            selected_count: Number of selected tickers
            top_performers_count: Number of top performers
            
        Returns:
            F1 score
        """
        if selected_count == 0 or top_performers_count == 0:
            return 0.0
        
        precision = overlap_count / selected_count
        recall = overlap_count / top_performers_count
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _generate_comparison_summary(self, comparisons: Dict) -> Dict:
        """
        Generate summary of benchmark comparisons.
        
        Args:
            comparisons: Dictionary with benchmark comparisons
            
        Returns:
            Dictionary with comparison summary
        """
        if not comparisons:
            return {}
        
        summary = {
            'total_benchmarks': len(comparisons),
            'outperforming_benchmarks': 0,
            'underperforming_benchmarks': 0,
            'avg_outperformance': 0,
            'best_outperformance': 0,
            'worst_outperformance': 0
        }
        
        outperformance_values = []
        
        for benchmark, comparison in comparisons.items():
            if 'error' not in comparison:
                outperformance = comparison.get('outperformance', 0)
                outperformance_values.append(outperformance)
                
                if outperformance > 0:
                    summary['outperforming_benchmarks'] += 1
                else:
                    summary['underperforming_benchmarks'] += 1
        
        if outperformance_values:
            summary['avg_outperformance'] = np.mean(outperformance_values)
            summary['best_outperformance'] = max(outperformance_values)
            summary['worst_outperformance'] = min(outperformance_values)
        
        return summary
    
    def get_comparison_history(self) -> pd.DataFrame:
        """
        Get history of benchmark comparisons.
        
        Returns:
            DataFrame with comparison history
        """
        if not self.comparison_results:
            return pd.DataFrame()
        
        # Convert to DataFrame format
        history_data = []
        for result in self.comparison_results:
            for benchmark, comparison in result['comparisons'].items():
                if 'error' not in comparison:
                    history_data.append({
                        'timestamp': result['timestamp'],
                        'benchmark': benchmark,
                        'outperformance': comparison.get('outperformance', 0),
                        'sharpe_improvement': comparison.get('sharpe_improvement', 0),
                        'num_selected_tickers': len(result['selected_tickers'])
                    })
        
        return pd.DataFrame(history_data)
    
    def clear_history(self):
        """Clear comparison history."""
        self.comparison_results.clear()
        logger.info("Benchmark comparison history cleared")
