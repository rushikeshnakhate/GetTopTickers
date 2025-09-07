"""
Performance Validator for validating selection performance against various criteria.

This module provides validation capabilities to ensure the ticker selection
algorithm meets performance standards and criteria.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceValidator:
    """Validate ticker selection performance against various criteria."""
    
    def __init__(self):
        """Initialize the PerformanceValidator."""
        self.validation_criteria = {
            'min_hit_rate': 0.6,  # Minimum 60% hit rate
            'min_sharpe_ratio': 1.0,  # Minimum Sharpe ratio of 1.0
            'max_drawdown_threshold': -0.2,  # Maximum 20% drawdown
            'min_annual_return': 0.1,  # Minimum 10% annual return
            'max_volatility': 0.3,  # Maximum 30% volatility
            'min_win_rate': 0.5  # Minimum 50% win rate
        }
        
    def validate_selection_performance(self, selected_tickers: List[str],
                                     performance_metrics: Dict,
                                     benchmark_metrics: Dict = None) -> Dict:
        """
        Validate selection performance against criteria.
        
        Args:
            selected_tickers: List of selected tickers
            performance_metrics: Performance metrics dictionary
            benchmark_metrics: Benchmark metrics for comparison
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'overall_valid': True,
            'criteria_checks': {},
            'recommendations': [],
            'performance_score': 0.0
        }
        
        # Check individual criteria
        criteria_results = {}
        for criterion, threshold in self.validation_criteria.items():
            result = self._check_criterion(criterion, performance_metrics, threshold)
            criteria_results[criterion] = result
            
            if not result['passed']:
                validation_results['overall_valid'] = False
                validation_results['recommendations'].append(result['recommendation'])
        
        validation_results['criteria_checks'] = criteria_results
        
        # Calculate overall performance score
        validation_results['performance_score'] = self._calculate_performance_score(
            criteria_results, performance_metrics
        )
        
        # Compare with benchmark if provided
        if benchmark_metrics:
            benchmark_comparison = self._compare_with_benchmark(
                performance_metrics, benchmark_metrics
            )
            validation_results['benchmark_comparison'] = benchmark_comparison
        
        return validation_results
    
    def validate_consistency(self, historical_results: List[Dict]) -> Dict:
        """
        Validate consistency of performance over time.
        
        Args:
            historical_results: List of historical performance results
            
        Returns:
            Dictionary with consistency validation results
        """
        if not historical_results:
            return {'error': 'No historical results provided'}
        
        # Extract performance metrics
        hit_rates = [result.get('hit_rate', 0) for result in historical_results]
        sharpe_ratios = [result.get('sharpe_ratio', 0) for result in historical_results]
        annual_returns = [result.get('annual_return', 0) for result in historical_results]
        
        consistency_results = {
            'hit_rate_consistency': self._calculate_consistency_score(hit_rates),
            'sharpe_consistency': self._calculate_consistency_score(sharpe_ratios),
            'return_consistency': self._calculate_consistency_score(annual_returns),
            'overall_consistency': 0.0,
            'trend_analysis': self._analyze_trends(historical_results)
        }
        
        # Calculate overall consistency
        consistency_scores = [
            consistency_results['hit_rate_consistency'],
            consistency_results['sharpe_consistency'],
            consistency_results['return_consistency']
        ]
        consistency_results['overall_consistency'] = np.mean(consistency_scores)
        
        return consistency_results
    
    def validate_risk_metrics(self, performance_metrics: Dict) -> Dict:
        """
        Validate risk-related metrics.
        
        Args:
            performance_metrics: Performance metrics dictionary
            
        Returns:
            Dictionary with risk validation results
        """
        risk_validation = {
            'max_drawdown_valid': True,
            'volatility_valid': True,
            'var_valid': True,
            'risk_score': 0.0,
            'recommendations': []
        }
        
        # Check maximum drawdown
        max_dd = performance_metrics.get('max_drawdown', 0)
        if max_dd < self.validation_criteria['max_drawdown_threshold']:
            risk_validation['max_drawdown_valid'] = False
            risk_validation['recommendations'].append(
                f"Maximum drawdown {max_dd:.2%} exceeds threshold {self.validation_criteria['max_drawdown_threshold']:.2%}"
            )
        
        # Check volatility
        volatility = performance_metrics.get('volatility', 0)
        if volatility > self.validation_criteria['max_volatility']:
            risk_validation['volatility_valid'] = False
            risk_validation['recommendations'].append(
                f"Volatility {volatility:.2%} exceeds threshold {self.validation_criteria['max_volatility']:.2%}"
            )
        
        # Calculate Value at Risk (VaR) if returns are available
        if 'returns' in performance_metrics:
            returns = performance_metrics['returns']
            var_95 = np.percentile(returns, 5)  # 95% VaR
            if var_95 < -0.1:  # 10% daily loss threshold
                risk_validation['var_valid'] = False
                risk_validation['recommendations'].append(
                    f"95% VaR {var_95:.2%} indicates high tail risk"
                )
        
        # Calculate risk score (lower is better)
        risk_score = 0
        if not risk_validation['max_drawdown_valid']:
            risk_score += 0.4
        if not risk_validation['volatility_valid']:
            risk_score += 0.3
        if not risk_validation['var_valid']:
            risk_score += 0.3
        
        risk_validation['risk_score'] = risk_score
        
        return risk_validation
    
    def _check_criterion(self, criterion: str, metrics: Dict, threshold: float) -> Dict:
        """
        Check a specific performance criterion.
        
        Args:
            criterion: Criterion name
            metrics: Performance metrics
            threshold: Threshold value
            
        Returns:
            Dictionary with criterion check results
        """
        metric_value = metrics.get(criterion.replace('min_', '').replace('max_', ''), 0)
        
        if criterion.startswith('min_'):
            passed = metric_value >= threshold
            comparison = '>='
        elif criterion.startswith('max_'):
            passed = metric_value <= threshold
            comparison = '<='
        else:
            passed = True
            comparison = '='
        
        recommendation = ""
        if not passed:
            if criterion.startswith('min_'):
                recommendation = f"Improve {criterion.replace('min_', '')} from {metric_value:.2%} to at least {threshold:.2%}"
            elif criterion.startswith('max_'):
                recommendation = f"Reduce {criterion.replace('max_', '')} from {metric_value:.2%} to at most {threshold:.2%}"
        
        return {
            'criterion': criterion,
            'threshold': threshold,
            'actual_value': metric_value,
            'passed': passed,
            'comparison': comparison,
            'recommendation': recommendation
        }
    
    def _calculate_performance_score(self, criteria_results: Dict, metrics: Dict) -> float:
        """
        Calculate overall performance score.
        
        Args:
            criteria_results: Results of criteria checks
            metrics: Performance metrics
            
        Returns:
            Performance score (0-100)
        """
        if not criteria_results:
            return 0.0
        
        # Weight different criteria
        weights = {
            'min_hit_rate': 0.25,
            'min_sharpe_ratio': 0.20,
            'max_drawdown_threshold': 0.20,
            'min_annual_return': 0.15,
            'max_volatility': 0.10,
            'min_win_rate': 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, result in criteria_results.items():
            weight = weights.get(criterion, 0.1)
            total_weight += weight
            
            if result['passed']:
                total_score += weight * 100
            else:
                # Partial credit based on how close to threshold
                threshold = result['threshold']
                actual = result['actual_value']
                
                if criterion.startswith('min_'):
                    # For minimum criteria, give partial credit if close
                    if actual > 0:
                        partial_score = min(100, (actual / threshold) * 100)
                    else:
                        partial_score = 0
                elif criterion.startswith('max_'):
                    # For maximum criteria, give partial credit if not too far over
                    if threshold > 0:
                        partial_score = max(0, 100 - ((actual - threshold) / threshold) * 100)
                    else:
                        partial_score = 0
                else:
                    partial_score = 0
                
                total_score += weight * partial_score
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _compare_with_benchmark(self, your_metrics: Dict, benchmark_metrics: Dict) -> Dict:
        """
        Compare performance with benchmark.
        
        Args:
            your_metrics: Your performance metrics
            benchmark_metrics: Benchmark performance metrics
            
        Returns:
            Dictionary with benchmark comparison
        """
        comparison = {
            'outperforms_benchmark': True,
            'outperformance_metrics': {},
            'areas_for_improvement': []
        }
        
        # Compare key metrics
        metrics_to_compare = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        
        for metric in metrics_to_compare:
            your_value = your_metrics.get(metric, 0)
            benchmark_value = benchmark_metrics.get(metric, 0)
            
            if metric in ['max_drawdown']:
                # For drawdown, lower is better
                outperforms = your_value > benchmark_value
            else:
                # For other metrics, higher is better
                outperforms = your_value > benchmark_value
            
            comparison['outperformance_metrics'][metric] = {
                'your_value': your_value,
                'benchmark_value': benchmark_value,
                'outperforms': outperforms,
                'difference': your_value - benchmark_value
            }
            
            if not outperforms:
                comparison['areas_for_improvement'].append(metric)
        
        # Overall outperformance
        outperformance_count = sum(1 for m in comparison['outperformance_metrics'].values() if m['outperforms'])
        comparison['outperforms_benchmark'] = outperformance_count >= len(metrics_to_compare) / 2
        
        return comparison
    
    def _calculate_consistency_score(self, values: List[float]) -> float:
        """
        Calculate consistency score based on coefficient of variation.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Consistency score (0-100, higher is more consistent)
        """
        if not values or len(values) < 2:
            return 0.0
        
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if mean_value == 0:
            return 0.0
        
        # Coefficient of variation (lower is more consistent)
        cv = std_value / abs(mean_value)
        
        # Convert to consistency score (0-100)
        consistency_score = max(0, 100 - (cv * 100))
        
        return consistency_score
    
    def _analyze_trends(self, historical_results: List[Dict]) -> Dict:
        """
        Analyze trends in historical performance.
        
        Args:
            historical_results: List of historical results
            
        Returns:
            Dictionary with trend analysis
        """
        if len(historical_results) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Extract metrics over time
        hit_rates = [result.get('hit_rate', 0) for result in historical_results]
        sharpe_ratios = [result.get('sharpe_ratio', 0) for result in historical_results]
        annual_returns = [result.get('annual_return', 0) for result in historical_results]
        
        trends = {}
        
        for metric_name, values in [('hit_rate', hit_rates), ('sharpe_ratio', sharpe_ratios), ('annual_return', annual_returns)]:
            if len(values) >= 2:
                # Calculate trend using linear regression
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                trends[metric_name] = {
                    'slope': slope,
                    'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                    'strength': abs(slope)
                }
        
        return trends
    
    def set_validation_criteria(self, criteria: Dict):
        """
        Set custom validation criteria.
        
        Args:
            criteria: Dictionary with custom criteria
        """
        self.validation_criteria.update(criteria)
        logger.info(f"Updated validation criteria: {criteria}")
    
    def get_validation_criteria(self) -> Dict:
        """
        Get current validation criteria.
        
        Returns:
            Dictionary with current criteria
        """
        return self.validation_criteria.copy()
