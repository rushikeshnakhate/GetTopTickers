"""
Selection Accuracy module for measuring ticker selection performance.

This module provides comprehensive accuracy measurement capabilities to evaluate
how well the ticker selection algorithm performs compared to actual market outcomes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class SelectionAccuracy:
    """Measure and analyze ticker selection accuracy."""
    
    def __init__(self):
        """Initialize the SelectionAccuracy analyzer."""
        self.accuracy_results = []
        
    def calculate_selection_accuracy(self, selected_tickers: List[str], 
                                   benchmark_tickers: List[str],
                                   start_date: str, end_date: str,
                                   method: str = 'overlap') -> Dict:
        """
        Calculate selection accuracy using various methods.
        
        Args:
            selected_tickers: List of selected tickers
            benchmark_tickers: List of benchmark tickers
            start_date: Start date for performance evaluation
            end_date: End date for performance evaluation
            method: Accuracy calculation method ('overlap', 'performance', 'rank_correlation')
            
        Returns:
            Dictionary with accuracy metrics
        """
        if method == 'overlap':
            return self._calculate_overlap_accuracy(selected_tickers, benchmark_tickers)
        elif method == 'performance':
            return self._calculate_performance_accuracy(
                selected_tickers, benchmark_tickers, start_date, end_date
            )
        elif method == 'rank_correlation':
            return self._calculate_rank_correlation_accuracy(
                selected_tickers, benchmark_tickers, start_date, end_date
            )
        else:
            raise ValueError(f"Unknown accuracy method: {method}")
    
    def calculate_hit_rate(self, selected_tickers: List[str], 
                          actual_top_performers: List[str]) -> Dict:
        """
        Calculate hit rate - percentage of selected tickers that were top performers.
        
        Args:
            selected_tickers: List of selected tickers
            actual_top_performers: List of actual top performing tickers
            
        Returns:
            Dictionary with hit rate metrics
        """
        if not selected_tickers or not actual_top_performers:
            return {
                'hit_rate': 0.0,
                'hits': 0,
                'total_selected': 0,
                'total_top_performers': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Calculate hits (intersection)
        hits = set(selected_tickers) & set(actual_top_performers)
        num_hits = len(hits)
        
        # Calculate metrics
        hit_rate = num_hits / len(selected_tickers)
        precision = num_hits / len(selected_tickers) if selected_tickers else 0
        recall = num_hits / len(actual_top_performers) if actual_top_performers else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'hits': num_hits,
            'total_selected': len(selected_tickers),
            'total_top_performers': len(actual_top_performers),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'hit_tickers': list(hits),
            'missed_tickers': list(set(actual_top_performers) - set(selected_tickers)),
            'false_positives': list(set(selected_tickers) - set(actual_top_performers))
        }
    
    def calculate_rank_accuracy(self, selected_tickers: List[str], 
                               actual_rankings: Dict[str, float]) -> Dict:
        """
        Calculate rank-based accuracy metrics.
        
        Args:
            selected_tickers: List of selected tickers
            actual_rankings: Dictionary mapping tickers to their actual performance scores
            
        Returns:
            Dictionary with rank accuracy metrics
        """
        if not selected_tickers or not actual_rankings:
            return {
                'rank_correlation': 0.0,
                'mean_rank': 0.0,
                'median_rank': 0.0,
                'top_quartile_hits': 0,
                'top_decile_hits': 0
            }
        
        # Get actual ranks
        sorted_tickers = sorted(actual_rankings.items(), key=lambda x: x[1], reverse=True)
        actual_ranks = {ticker: rank for rank, (ticker, _) in enumerate(sorted_tickers, 1)}
        
        # Calculate ranks for selected tickers
        selected_ranks = []
        for ticker in selected_tickers:
            if ticker in actual_ranks:
                selected_ranks.append(actual_ranks[ticker])
        
        if not selected_ranks:
            return {
                'rank_correlation': 0.0,
                'mean_rank': 0.0,
                'median_rank': 0.0,
                'top_quartile_hits': 0,
                'top_decile_hits': 0
            }
        
        # Calculate rank metrics
        mean_rank = np.mean(selected_ranks)
        median_rank = np.median(selected_ranks)
        
        # Calculate top quartile and decile hits
        total_tickers = len(actual_rankings)
        top_quartile_threshold = total_tickers * 0.25
        top_decile_threshold = total_tickers * 0.1
        
        top_quartile_hits = sum(1 for rank in selected_ranks if rank <= top_quartile_threshold)
        top_decile_hits = sum(1 for rank in selected_ranks if rank <= top_decile_threshold)
        
        # Calculate rank correlation (Spearman-like)
        expected_ranks = list(range(1, len(selected_tickers) + 1))
        rank_correlation = np.corrcoef(selected_ranks, expected_ranks)[0, 1] if len(selected_ranks) > 1 else 0
        
        return {
            'rank_correlation': rank_correlation,
            'mean_rank': mean_rank,
            'median_rank': median_rank,
            'top_quartile_hits': top_quartile_hits,
            'top_decile_hits': top_decile_hits,
            'selected_ranks': selected_ranks,
            'total_tickers': total_tickers
        }
    
    def calculate_forward_looking_accuracy(self, selected_tickers: List[str],
                                         start_date: str, end_date: str,
                                         lookforward_days: int = 30) -> Dict:
        """
        Calculate forward-looking accuracy by checking performance after selection.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Selection date
            end_date: End date for evaluation
            lookforward_days: Days to look forward for performance evaluation
            
        Returns:
            Dictionary with forward-looking accuracy metrics
        """
        if not selected_tickers:
            return {
                'forward_return': 0.0,
                'outperformance_rate': 0.0,
                'positive_return_rate': 0.0,
                'avg_forward_return': 0.0,
                'median_forward_return': 0.0
            }
        
        # Calculate forward returns for each ticker
        forward_returns = []
        outperforming_tickers = []
        positive_return_tickers = []
        
        for ticker in selected_tickers:
            try:
                # Get data for the lookforward period
                selection_date = datetime.strptime(start_date, '%Y-%m-%d')
                forward_start = selection_date + timedelta(days=1)
                forward_end = forward_start + timedelta(days=lookforward_days)
                
                # Fetch data
                data = yf.download(ticker, 
                                 start=forward_start.strftime('%Y-%m-%d'),
                                 end=forward_end.strftime('%Y-%m-%d'),
                                 progress=False)
                
                if not data.empty and 'Adj Close' in data.columns:
                    # Calculate forward return
                    initial_price = data['Adj Close'].iloc[0]
                    final_price = data['Adj Close'].iloc[-1]
                    forward_return = (final_price - initial_price) / initial_price
                    
                    forward_returns.append(forward_return)
                    
                    if forward_return > 0:
                        positive_return_tickers.append(ticker)
                    
                    # Check if outperformed market (simplified - would need benchmark)
                    if forward_return > 0.05:  # 5% threshold
                        outperforming_tickers.append(ticker)
                
            except Exception as e:
                logger.warning(f"Error calculating forward return for {ticker}: {str(e)}")
                continue
        
        if not forward_returns:
            return {
                'forward_return': 0.0,
                'outperformance_rate': 0.0,
                'positive_return_rate': 0.0,
                'avg_forward_return': 0.0,
                'median_forward_return': 0.0
            }
        
        # Calculate metrics
        avg_forward_return = np.mean(forward_returns)
        median_forward_return = np.median(forward_returns)
        positive_return_rate = len(positive_return_tickers) / len(selected_tickers)
        outperformance_rate = len(outperforming_tickers) / len(selected_tickers)
        
        return {
            'forward_return': avg_forward_return,
            'outperformance_rate': outperformance_rate,
            'positive_return_rate': positive_return_rate,
            'avg_forward_return': avg_forward_return,
            'median_forward_return': median_forward_return,
            'forward_returns': forward_returns,
            'positive_return_tickers': positive_return_tickers,
            'outperforming_tickers': outperforming_tickers
        }
    
    def _calculate_overlap_accuracy(self, selected_tickers: List[str], 
                                  benchmark_tickers: List[str]) -> Dict:
        """Calculate simple overlap accuracy."""
        if not selected_tickers or not benchmark_tickers:
            return {'overlap_accuracy': 0.0, 'overlap_count': 0}
        
        overlap = set(selected_tickers) & set(benchmark_tickers)
        overlap_accuracy = len(overlap) / len(selected_tickers)
        
        return {
            'overlap_accuracy': overlap_accuracy,
            'overlap_count': len(overlap),
            'overlap_tickers': list(overlap)
        }
    
    def _calculate_performance_accuracy(self, selected_tickers: List[str],
                                      benchmark_tickers: List[str],
                                      start_date: str, end_date: str) -> Dict:
        """Calculate performance-based accuracy."""
        # This would implement performance-based accuracy calculation
        # For now, return a simplified version
        return {
            'performance_accuracy': 0.0,
            'method': 'performance_based'
        }
    
    def _calculate_rank_correlation_accuracy(self, selected_tickers: List[str],
                                           benchmark_tickers: List[str],
                                           start_date: str, end_date: str) -> Dict:
        """Calculate rank correlation accuracy."""
        # This would implement rank correlation calculation
        # For now, return a simplified version
        return {
            'rank_correlation': 0.0,
            'method': 'rank_correlation'
        }
    
    def generate_accuracy_report(self, results: List[Dict]) -> pd.DataFrame:
        """
        Generate comprehensive accuracy report.
        
        Args:
            results: List of accuracy results
            
        Returns:
            DataFrame with accuracy report
        """
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Add summary statistics
        summary = {
            'total_evaluations': len(df),
            'avg_hit_rate': df['hit_rate'].mean() if 'hit_rate' in df.columns else 0,
            'avg_precision': df['precision'].mean() if 'precision' in df.columns else 0,
            'avg_recall': df['recall'].mean() if 'recall' in df.columns else 0,
            'avg_f1_score': df['f1_score'].mean() if 'f1_score' in df.columns else 0,
            'best_hit_rate': df['hit_rate'].max() if 'hit_rate' in df.columns else 0,
            'worst_hit_rate': df['hit_rate'].min() if 'hit_rate' in df.columns else 0
        }
        
        return df, summary
