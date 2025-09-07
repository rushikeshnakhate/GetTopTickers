"""
Market Regime Analyzer for analyzing performance across different market conditions.

This module analyzes how the ticker selection algorithm performs in different
market regimes (bull, bear, sideways markets) to provide comprehensive validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """Analyze performance across different market regimes."""
    
    def __init__(self):
        """Initialize the MarketRegimeAnalyzer."""
        self.regime_results = []
        
    def analyze_market_regimes(self, selected_tickers: List[str],
                             start_date: str, end_date: str,
                             benchmark_ticker: str = '^NSEI') -> Dict:
        """
        Analyze performance across different market regimes.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for analysis
            end_date: End date for analysis
            benchmark_ticker: Benchmark ticker for regime classification
            
        Returns:
            Dictionary with regime analysis results
        """
        logger.info(f"Analyzing market regimes from {start_date} to {end_date}")
        
        try:
            # Get benchmark data for regime classification
            benchmark_data = self._get_benchmark_data(benchmark_ticker, start_date, end_date)
            
            if benchmark_data.empty:
                return {'error': 'Unable to fetch benchmark data for regime analysis'}
            
            # Classify market regimes
            regime_periods = self._classify_market_regimes(benchmark_data)
            
            # Analyze performance in each regime
            regime_analysis = {}
            for regime, periods in regime_periods.items():
                regime_performance = self._analyze_regime_performance(
                    selected_tickers, periods, regime
                )
                regime_analysis[regime] = regime_performance
            
            # Calculate overall statistics
            overall_stats = self._calculate_overall_regime_stats(regime_analysis)
            
            results = {
                'regime_periods': regime_periods,
                'regime_analysis': regime_analysis,
                'overall_statistics': overall_stats,
                'regime_classification_summary': self._summarize_regime_classification(regime_periods)
            }
            
            # Store results
            self.regime_results.append({
                'timestamp': datetime.now(),
                'selected_tickers': selected_tickers,
                'start_date': start_date,
                'end_date': end_date,
                'results': results
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_volatility_regimes(self, selected_tickers: List[str],
                                 start_date: str, end_date: str,
                                 benchmark_ticker: str = '^NSEI') -> Dict:
        """
        Analyze performance across different volatility regimes.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for analysis
            end_date: End date for analysis
            benchmark_ticker: Benchmark ticker for volatility calculation
            
        Returns:
            Dictionary with volatility regime analysis
        """
        logger.info(f"Analyzing volatility regimes from {start_date} to {end_date}")
        
        try:
            # Get benchmark data
            benchmark_data = self._get_benchmark_data(benchmark_ticker, start_date, end_date)
            
            if benchmark_data.empty:
                return {'error': 'Unable to fetch benchmark data for volatility analysis'}
            
            # Calculate rolling volatility
            benchmark_data['returns'] = benchmark_data['Adj Close'].pct_change()
            benchmark_data['volatility'] = benchmark_data['returns'].rolling(window=30).std() * np.sqrt(252)
            
            # Classify volatility regimes
            volatility_periods = self._classify_volatility_regimes(benchmark_data)
            
            # Analyze performance in each volatility regime
            volatility_analysis = {}
            for regime, periods in volatility_periods.items():
                regime_performance = self._analyze_regime_performance(
                    selected_tickers, periods, f"volatility_{regime}"
                )
                volatility_analysis[regime] = regime_performance
            
            return {
                'volatility_periods': volatility_periods,
                'volatility_analysis': volatility_analysis,
                'volatility_summary': self._summarize_volatility_regimes(volatility_periods)
            }
            
        except Exception as e:
            logger.error(f"Error in volatility regime analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_sector_rotation(self, selected_tickers: List[str],
                              start_date: str, end_date: str) -> Dict:
        """
        Analyze performance during different sector rotation periods.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with sector rotation analysis
        """
        logger.info(f"Analyzing sector rotation from {start_date} to {end_date}")
        
        try:
            # Define sector mappings (simplified)
            sector_mapping = self._get_sector_mapping()
            
            # Analyze performance by sector
            sector_analysis = {}
            for sector, sector_tickers in sector_mapping.items():
                # Find selected tickers in this sector
                sector_selected = [ticker for ticker in selected_tickers if ticker in sector_tickers]
                
                if sector_selected:
                    sector_performance = self._analyze_regime_performance(
                        sector_selected, [(start_date, end_date)], f"sector_{sector}"
                    )
                    sector_analysis[sector] = {
                        'selected_tickers': sector_selected,
                        'performance': sector_performance,
                        'sector_weight': len(sector_selected) / len(selected_tickers)
                    }
            
            return {
                'sector_analysis': sector_analysis,
                'sector_diversification': self._calculate_sector_diversification(sector_analysis),
                'sector_rotation_summary': self._summarize_sector_rotation(sector_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in sector rotation analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_benchmark_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get benchmark data for regime classification."""
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            return data
        except Exception as e:
            logger.error(f"Error fetching benchmark data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _classify_market_regimes(self, benchmark_data: pd.DataFrame) -> Dict:
        """
        Classify market regimes based on benchmark performance.
        
        Args:
            benchmark_data: DataFrame with benchmark price data
            
        Returns:
            Dictionary with regime periods
        """
        # Calculate returns
        benchmark_data['returns'] = benchmark_data['Adj Close'].pct_change()
        
        # Calculate rolling metrics
        benchmark_data['rolling_return'] = benchmark_data['returns'].rolling(window=30).mean() * 252
        benchmark_data['rolling_volatility'] = benchmark_data['returns'].rolling(window=30).std() * np.sqrt(252)
        
        # Classify regimes
        regimes = []
        for date, row in benchmark_data.iterrows():
            if pd.isna(row['rolling_return']) or pd.isna(row['rolling_volatility']):
                continue
            
            annual_return = row['rolling_return']
            volatility = row['rolling_volatility']
            
            if annual_return > 0.15 and volatility < 0.25:
                regime = 'bull_low_vol'
            elif annual_return > 0.15 and volatility >= 0.25:
                regime = 'bull_high_vol'
            elif annual_return < -0.15:
                regime = 'bear'
            elif abs(annual_return) <= 0.05:
                regime = 'sideways'
            else:
                regime = 'transitional'
            
            regimes.append((date, regime))
        
        # Group consecutive periods of same regime
        regime_periods = self._group_consecutive_regimes(regimes)
        
        return regime_periods
    
    def _classify_volatility_regimes(self, benchmark_data: pd.DataFrame) -> Dict:
        """
        Classify volatility regimes.
        
        Args:
            benchmark_data: DataFrame with volatility data
            
        Returns:
            Dictionary with volatility regime periods
        """
        # Calculate volatility percentiles
        volatility_series = benchmark_data['volatility'].dropna()
        low_threshold = volatility_series.quantile(0.33)
        high_threshold = volatility_series.quantile(0.67)
        
        # Classify volatility regimes
        regimes = []
        for date, row in benchmark_data.iterrows():
            if pd.isna(row['volatility']):
                continue
            
            volatility = row['volatility']
            
            if volatility <= low_threshold:
                regime = 'low_volatility'
            elif volatility >= high_threshold:
                regime = 'high_volatility'
            else:
                regime = 'medium_volatility'
            
            regimes.append((date, regime))
        
        # Group consecutive periods
        regime_periods = self._group_consecutive_regimes(regimes)
        
        return regime_periods
    
    def _group_consecutive_regimes(self, regimes: List[Tuple]) -> Dict:
        """Group consecutive periods of the same regime."""
        if not regimes:
            return {}
        
        regime_periods = {}
        current_regime = regimes[0][1]
        start_date = regimes[0][0]
        
        for date, regime in regimes[1:]:
            if regime != current_regime:
                # End current regime
                if current_regime not in regime_periods:
                    regime_periods[current_regime] = []
                regime_periods[current_regime].append((start_date, date))
                
                # Start new regime
                current_regime = regime
                start_date = date
        
        # Add final regime
        if current_regime not in regime_periods:
            regime_periods[current_regime] = []
        regime_periods[current_regime].append((start_date, regimes[-1][0]))
        
        return regime_periods
    
    def _analyze_regime_performance(self, tickers: List[str], periods: List[Tuple], 
                                  regime_name: str) -> Dict:
        """
        Analyze performance for a specific regime.
        
        Args:
            tickers: List of tickers
            periods: List of (start_date, end_date) tuples
            regime_name: Name of the regime
            
        Returns:
            Dictionary with regime performance analysis
        """
        if not tickers or not periods:
            return {'error': 'No tickers or periods provided'}
        
        all_performances = []
        
        for start_date, end_date in periods:
            try:
                # Get data for this period
                data = yf.download(tickers, start=start_date, end=end_date, 
                                 progress=False, group_by='ticker')
                
                if data.empty:
                    continue
                
                # Handle different data structures
                if isinstance(data.columns, pd.MultiIndex):
                    close_prices = data.xs('Adj Close', level=1, axis=1)
                else:
                    close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
                
                close_prices = close_prices.dropna(how='all')
                
                if close_prices.empty:
                    continue
                
                # Calculate returns
                returns = close_prices.pct_change().dropna()
                portfolio_returns = returns.mean(axis=1)
                
                # Calculate performance metrics
                performance = self._calculate_regime_performance_metrics(portfolio_returns)
                performance['period_start'] = start_date
                performance['period_end'] = end_date
                
                all_performances.append(performance)
                
            except Exception as e:
                logger.warning(f"Error analyzing period {start_date} to {end_date}: {str(e)}")
                continue
        
        if not all_performances:
            return {'error': 'No valid performance data for this regime'}
        
        # Aggregate performance across all periods in this regime
        return self._aggregate_regime_performance(all_performances, regime_name)
    
    def _calculate_regime_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics for a regime period."""
        if returns.empty:
            return self._get_empty_regime_metrics()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * (252 ** 0.5)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_observations': len(returns)
        }
    
    def _aggregate_regime_performance(self, performances: List[Dict], regime_name: str) -> Dict:
        """Aggregate performance across multiple periods in a regime."""
        if not performances:
            return {'error': 'No performances to aggregate'}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 
                      'max_drawdown', 'win_rate']:
            values = [p.get(metric, 0) for p in performances]
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
            avg_metrics[f'min_{metric}'] = np.min(values)
            avg_metrics[f'max_{metric}'] = np.max(values)
        
        return {
            'regime_name': regime_name,
            'num_periods': len(performances),
            'aggregated_metrics': avg_metrics,
            'individual_periods': performances
        }
    
    def _calculate_overall_regime_stats(self, regime_analysis: Dict) -> Dict:
        """Calculate overall statistics across all regimes."""
        if not regime_analysis:
            return {}
        
        # Extract key metrics from each regime
        regime_returns = []
        regime_sharpes = []
        regime_volatilities = []
        
        for regime, analysis in regime_analysis.items():
            if 'aggregated_metrics' in analysis:
                metrics = analysis['aggregated_metrics']
                regime_returns.append(metrics.get('avg_annual_return', 0))
                regime_sharpes.append(metrics.get('avg_sharpe_ratio', 0))
                regime_volatilities.append(metrics.get('avg_volatility', 0))
        
        return {
            'regime_count': len(regime_analysis),
            'avg_return_across_regimes': np.mean(regime_returns) if regime_returns else 0,
            'avg_sharpe_across_regimes': np.mean(regime_sharpes) if regime_sharpes else 0,
            'avg_volatility_across_regimes': np.mean(regime_volatilities) if regime_volatilities else 0,
            'regime_consistency': 1 - np.std(regime_returns) if len(regime_returns) > 1 else 0
        }
    
    def _summarize_regime_classification(self, regime_periods: Dict) -> Dict:
        """Summarize regime classification results."""
        summary = {}
        
        for regime, periods in regime_periods.items():
            total_days = sum((end - start).days for start, end in periods)
            summary[regime] = {
                'num_periods': len(periods),
                'total_days': total_days,
                'avg_period_length': total_days / len(periods) if periods else 0
            }
        
        return summary
    
    def _summarize_volatility_regimes(self, volatility_periods: Dict) -> Dict:
        """Summarize volatility regime classification."""
        return self._summarize_regime_classification(volatility_periods)
    
    def _get_sector_mapping(self) -> Dict:
        """Get sector mapping for tickers (simplified version)."""
        # This is a simplified mapping - in practice, you'd have a comprehensive sector database
        return {
            'banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
            'technology': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'TECHM.NS', 'WIPRO.NS'],
            'pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
            'automobile': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS'],
            'energy': ['RELIANCE.NS', 'ONGC.NS', 'COALINDIA.NS', 'NTPC.NS']
        }
    
    def _calculate_sector_diversification(self, sector_analysis: Dict) -> Dict:
        """Calculate sector diversification metrics."""
        if not sector_analysis:
            return {}
        
        sector_weights = [analysis['sector_weight'] for analysis in sector_analysis.values()]
        
        # Calculate Herfindahl-Hirschman Index (concentration measure)
        hhi = sum(weight ** 2 for weight in sector_weights)
        
        # Calculate effective number of sectors
        effective_sectors = 1 / hhi if hhi > 0 else 0
        
        return {
            'num_sectors': len(sector_analysis),
            'hhi': hhi,
            'effective_sectors': effective_sectors,
            'max_sector_weight': max(sector_weights) if sector_weights else 0,
            'min_sector_weight': min(sector_weights) if sector_weights else 0
        }
    
    def _summarize_sector_rotation(self, sector_analysis: Dict) -> Dict:
        """Summarize sector rotation analysis."""
        if not sector_analysis:
            return {}
        
        # Find best and worst performing sectors
        sector_performances = {}
        for sector, analysis in sector_analysis.items():
            if 'performance' in analysis and 'aggregated_metrics' in analysis['performance']:
                metrics = analysis['performance']['aggregated_metrics']
                sector_performances[sector] = metrics.get('avg_annual_return', 0)
        
        if sector_performances:
            best_sector = max(sector_performances, key=sector_performances.get)
            worst_sector = min(sector_performances, key=sector_performances.get)
        else:
            best_sector = worst_sector = None
        
        return {
            'best_performing_sector': best_sector,
            'worst_performing_sector': worst_sector,
            'sector_performance_spread': max(sector_performances.values()) - min(sector_performances.values()) if sector_performances else 0
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _get_empty_regime_metrics(self) -> Dict:
        """Return empty regime performance metrics."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_observations': 0
        }
    
    def get_regime_analysis_history(self) -> pd.DataFrame:
        """Get history of regime analyses."""
        if not self.regime_results:
            return pd.DataFrame()
        
        history_data = []
        for result in self.regime_results:
            history_data.append({
                'timestamp': result['timestamp'],
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'num_tickers': len(result['selected_tickers']),
                'regime_count': len(result['results'].get('regime_analysis', {}))
            })
        
        return pd.DataFrame(history_data)
    
    def clear_history(self):
        """Clear regime analysis history."""
        self.regime_results.clear()
        logger.info("Market regime analysis history cleared")
