"""
Walk Forward Validator for out-of-sample testing and validation.

This module implements walk-forward analysis to validate the ticker selection
algorithm's performance on out-of-sample data, addressing reviewer concerns
about overfitting and generalization.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Implement walk-forward analysis for out-of-sample testing."""
    
    def __init__(self, ticker_selector_func: Optional[Callable] = None):
        """
        Initialize the WalkForwardValidator.
        
        Args:
            ticker_selector_func: Function to use for ticker selection
        """
        self.ticker_selector = ticker_selector_func
        self.walk_forward_results = []
        
    def walk_forward_analysis(self, start_year: int, end_year: int, 
                            train_months: int = 12, test_months: int = 3,
                            rebalance_frequency: str = 'monthly') -> Dict:
        """
        Perform walk-forward analysis.
        
        Args:
            start_year: Starting year for analysis
            end_year: Ending year for analysis
            train_months: Number of months for training period
            test_months: Number of months for testing period
            rebalance_frequency: Rebalancing frequency ('daily', 'monthly', 'quarterly')
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        logger.info(f"Starting walk-forward analysis from {start_year} to {end_year}")
        
        results = []
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        iteration = 0
        while current_date < end_date:
            iteration += 1
            
            # Define training and testing periods
            train_start = current_date
            train_end = train_start + timedelta(days=train_months * 30)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_months * 30)
            
            if test_end > end_date:
                break
            
            logger.info(f"Iteration {iteration}: Training {train_start.date()} to {train_end.date()}, "
                       f"Testing {test_start.date()} to {test_end.date()}")
            
            try:
                # Train on historical data
                selected_tickers = self._run_ticker_selection(
                    train_start.strftime('%Y-%m-%d'),
                    train_end.strftime('%Y-%m-%d')
                )
                
                if not selected_tickers:
                    logger.warning(f"No tickers selected for iteration {iteration}")
                    continue
                
                # Test on out-of-sample period
                test_performance = self._calculate_out_of_sample_performance(
                    selected_tickers, 
                    test_start.strftime('%Y-%m-%d'), 
                    test_end.strftime('%Y-%m-%d'),
                    rebalance_frequency
                )
                
                # Calculate training performance for comparison
                train_performance = self._calculate_out_of_sample_performance(
                    selected_tickers,
                    train_start.strftime('%Y-%m-%d'),
                    train_end.strftime('%Y-%m-%d'),
                    rebalance_frequency
                )
                
                result = {
                    'iteration': iteration,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'selected_tickers': selected_tickers,
                    'num_tickers': len(selected_tickers),
                    'train_performance': train_performance,
                    'test_performance': test_performance,
                    'performance_degradation': self._calculate_performance_degradation(
                        train_performance, test_performance
                    )
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                continue
            
            # Move to next period
            current_date = test_start
        
        # Store results
        self.walk_forward_results.append({
            'timestamp': datetime.now(),
            'parameters': {
                'start_year': start_year,
                'end_year': end_year,
                'train_months': train_months,
                'test_months': test_months,
                'rebalance_frequency': rebalance_frequency
            },
            'results': results
        })
        
        return self._summarize_walk_forward_results(results)
    
    def rolling_window_analysis(self, start_year: int, end_year: int,
                              window_size_months: int = 24,
                              step_size_months: int = 3) -> Dict:
        """
        Perform rolling window analysis.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            window_size_months: Size of rolling window in months
            step_size_months: Step size for rolling window
            
        Returns:
            Dictionary with rolling window analysis results
        """
        logger.info(f"Starting rolling window analysis with {window_size_months} month windows")
        
        results = []
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        iteration = 0
        while current_date + timedelta(days=window_size_months * 30) <= end_date:
            iteration += 1
            
            # Define window period
            window_start = current_date
            window_end = window_start + timedelta(days=window_size_months * 30)
            
            logger.info(f"Rolling window {iteration}: {window_start.date()} to {window_end.date()}")
            
            try:
                # Select tickers for this window
                selected_tickers = self._run_ticker_selection(
                    window_start.strftime('%Y-%m-%d'),
                    window_end.strftime('%Y-%m-%d')
                )
                
                if not selected_tickers:
                    continue
                
                # Calculate performance for this window
                window_performance = self._calculate_out_of_sample_performance(
                    selected_tickers,
                    window_start.strftime('%Y-%m-%d'),
                    window_end.strftime('%Y-%m-%d')
                )
                
                result = {
                    'iteration': iteration,
                    'window_start': window_start,
                    'window_end': window_end,
                    'selected_tickers': selected_tickers,
                    'num_tickers': len(selected_tickers),
                    'performance': window_performance
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in rolling window {iteration}: {str(e)}")
                continue
            
            # Move to next window
            current_date += timedelta(days=step_size_months * 30)
        
        return self._summarize_rolling_window_results(results)
    
    def expanding_window_analysis(self, start_year: int, end_year: int,
                                min_window_months: int = 12,
                                step_size_months: int = 3) -> Dict:
        """
        Perform expanding window analysis.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            min_window_months: Minimum window size in months
            step_size_months: Step size for expanding window
            
        Returns:
            Dictionary with expanding window analysis results
        """
        logger.info(f"Starting expanding window analysis with {min_window_months} month minimum window")
        
        results = []
        base_start = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        current_end = base_start + timedelta(days=min_window_months * 30)
        iteration = 0
        
        while current_end <= end_date:
            iteration += 1
            
            logger.info(f"Expanding window {iteration}: {base_start.date()} to {current_end.date()}")
            
            try:
                # Select tickers for this expanding window
                selected_tickers = self._run_ticker_selection(
                    base_start.strftime('%Y-%m-%d'),
                    current_end.strftime('%Y-%m-%d')
                )
                
                if not selected_tickers:
                    continue
                
                # Calculate performance for this window
                window_performance = self._calculate_out_of_sample_performance(
                    selected_tickers,
                    base_start.strftime('%Y-%m-%d'),
                    current_end.strftime('%Y-%m-%d')
                )
                
                result = {
                    'iteration': iteration,
                    'window_start': base_start,
                    'window_end': current_end,
                    'window_size_months': (current_end - base_start).days / 30,
                    'selected_tickers': selected_tickers,
                    'num_tickers': len(selected_tickers),
                    'performance': window_performance
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in expanding window {iteration}: {str(e)}")
                continue
            
            # Expand window
            current_end += timedelta(days=step_size_months * 30)
        
        return self._summarize_expanding_window_results(results)
    
    def _run_ticker_selection(self, start_date: str, end_date: str) -> List[str]:
        """
        Run ticker selection for given period.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of selected tickers
        """
        if self.ticker_selector:
            try:
                # Extract year from start_date for compatibility
                year = int(start_date.split('-')[0])
                result = self.ticker_selector(years=[year], top_n_tickers=15)
                return result if isinstance(result, list) else []
            except Exception as e:
                logger.error(f"Error in ticker selection: {str(e)}")
                return []
        else:
            # Mock implementation for testing
            return [f"TICKER{i}.NS" for i in range(1, 16)]
    
    def _calculate_out_of_sample_performance(self, tickers: List[str], 
                                           start_date: str, end_date: str,
                                           rebalance_frequency: str = 'monthly') -> Dict:
        """
        Calculate performance on out-of-sample data.
        
        Args:
            tickers: List of selected tickers
            start_date: Start date
            end_date: End date
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Fetch data for all tickers
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if data.empty:
                return self._get_empty_performance()
            
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data.xs('Adj Close', level=1, axis=1)
            else:
                close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
            
            # Clean data
            close_prices = close_prices.dropna(how='all')
            
            if close_prices.empty:
                return self._get_empty_performance()
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Apply rebalancing frequency
            if rebalance_frequency == 'monthly':
                returns = returns.resample('M').last()
            elif rebalance_frequency == 'quarterly':
                returns = returns.resample('Q').last()
            
            # Equal weight portfolio
            portfolio_returns = returns.mean(axis=1)
            
            # Calculate performance metrics
            return self._calculate_performance_metrics(portfolio_returns)
            
        except Exception as e:
            logger.error(f"Error calculating out-of-sample performance: {str(e)}")
            return self._get_empty_performance()
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Dictionary with performance metrics
        """
        if returns.empty:
            return self._get_empty_performance()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * (252 ** 0.5)
        
        # Risk-adjusted metrics
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Additional metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'num_observations': len(returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_performance_degradation(self, train_perf: Dict, test_perf: Dict) -> Dict:
        """
        Calculate performance degradation from training to testing.
        
        Args:
            train_perf: Training performance metrics
            test_perf: Testing performance metrics
            
        Returns:
            Dictionary with degradation metrics
        """
        degradation = {}
        
        for metric in ['annual_return', 'sharpe_ratio', 'win_rate']:
            train_value = train_perf.get(metric, 0)
            test_value = test_perf.get(metric, 0)
            
            if train_value != 0:
                degradation[f'{metric}_degradation'] = (test_value - train_value) / abs(train_value)
            else:
                degradation[f'{metric}_degradation'] = 0
            
            degradation[f'{metric}_train'] = train_value
            degradation[f'{metric}_test'] = test_value
        
        return degradation
    
    def _summarize_walk_forward_results(self, results: List[Dict]) -> Dict:
        """
        Summarize walk-forward analysis results.
        
        Args:
            results: List of walk-forward results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {'error': 'No results to summarize'}
        
        # Extract performance metrics
        test_performances = [r['test_performance'] for r in results]
        train_performances = [r['train_performance'] for r in results]
        degradations = [r['performance_degradation'] for r in results]
        
        # Calculate summary statistics
        summary = {
            'total_iterations': len(results),
            'successful_iterations': len([r for r in results if r['test_performance']]),
            'average_test_return': np.mean([p.get('annual_return', 0) for p in test_performances]),
            'average_test_sharpe': np.mean([p.get('sharpe_ratio', 0) for p in test_performances]),
            'average_test_win_rate': np.mean([p.get('win_rate', 0) for p in test_performances]),
            'average_test_volatility': np.mean([p.get('volatility', 0) for p in test_performances]),
            'average_test_max_dd': np.mean([p.get('max_drawdown', 0) for p in test_performances]),
            'consistency_score': 1 - np.std([p.get('annual_return', 0) for p in test_performances]),
            'average_degradation': {
                'return_degradation': np.mean([d.get('annual_return_degradation', 0) for d in degradations]),
                'sharpe_degradation': np.mean([d.get('sharpe_ratio_degradation', 0) for d in degradations]),
                'win_rate_degradation': np.mean([d.get('win_rate_degradation', 0) for d in degradations])
            },
            'best_period': max(results, key=lambda x: x['test_performance'].get('annual_return', 0)),
            'worst_period': min(results, key=lambda x: x['test_performance'].get('annual_return', 0)),
            'detailed_results': results
        }
        
        # Calculate additional metrics
        positive_periods = len([p for p in test_performances if p.get('annual_return', 0) > 0])
        summary['positive_period_rate'] = positive_periods / len(test_performances)
        
        # Stability analysis
        returns = [p.get('annual_return', 0) for p in test_performances]
        summary['stability_metrics'] = {
            'return_std': np.std(returns),
            'return_cv': np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else 0,
            'positive_consistency': positive_periods / len(returns)
        }
        
        return summary
    
    def _summarize_rolling_window_results(self, results: List[Dict]) -> Dict:
        """Summarize rolling window analysis results."""
        if not results:
            return {'error': 'No results to summarize'}
        
        performances = [r['performance'] for r in results]
        
        return {
            'total_windows': len(results),
            'average_return': np.mean([p.get('annual_return', 0) for p in performances]),
            'average_sharpe': np.mean([p.get('sharpe_ratio', 0) for p in performances]),
            'return_volatility': np.std([p.get('annual_return', 0) for p in performances]),
            'detailed_results': results
        }
    
    def _summarize_expanding_window_results(self, results: List[Dict]) -> Dict:
        """Summarize expanding window analysis results."""
        if not results:
            return {'error': 'No results to summarize'}
        
        performances = [r['performance'] for r in results]
        window_sizes = [r['window_size_months'] for r in results]
        
        return {
            'total_windows': len(results),
            'average_return': np.mean([p.get('annual_return', 0) for p in performances]),
            'average_sharpe': np.mean([p.get('sharpe_ratio', 0) for p in performances]),
            'window_size_impact': self._analyze_window_size_impact(results),
            'detailed_results': results
        }
    
    def _analyze_window_size_impact(self, results: List[Dict]) -> Dict:
        """Analyze impact of window size on performance."""
        if len(results) < 3:
            return {}
        
        window_sizes = [r['window_size_months'] for r in results]
        returns = [r['performance'].get('annual_return', 0) for r in results]
        
        # Calculate correlation between window size and performance
        correlation = np.corrcoef(window_sizes, returns)[0, 1]
        
        return {
            'window_size_correlation': correlation,
            'min_window_size': min(window_sizes),
            'max_window_size': max(window_sizes),
            'avg_window_size': np.mean(window_sizes)
        }
    
    def _get_empty_performance(self) -> Dict:
        """Return empty performance metrics."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'num_observations': 0
        }
    
    def get_walk_forward_history(self) -> pd.DataFrame:
        """Get history of walk-forward analyses."""
        if not self.walk_forward_results:
            return pd.DataFrame()
        
        history_data = []
        for result in self.walk_forward_results:
            for iteration_result in result['results']:
                history_data.append({
                    'timestamp': result['timestamp'],
                    'iteration': iteration_result['iteration'],
                    'test_start': iteration_result['test_start'],
                    'test_end': iteration_result['test_end'],
                    'num_tickers': iteration_result['num_tickers'],
                    'test_return': iteration_result['test_performance'].get('annual_return', 0),
                    'test_sharpe': iteration_result['test_performance'].get('sharpe_ratio', 0),
                    'test_win_rate': iteration_result['test_performance'].get('win_rate', 0)
                })
        
        return pd.DataFrame(history_data)
    
    def clear_history(self):
        """Clear walk-forward analysis history."""
        self.walk_forward_results.clear()
        logger.info("Walk-forward analysis history cleared")
