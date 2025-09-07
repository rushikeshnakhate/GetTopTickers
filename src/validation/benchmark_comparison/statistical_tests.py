"""
Statistical Tests module for comprehensive statistical analysis.

This module provides various statistical tests to validate the significance
and reliability of ticker selection performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, kstest, jarque_bera
import warnings

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Perform comprehensive statistical tests on ticker selection performance."""
    
    def __init__(self):
        """Initialize the StatisticalTests."""
        self.test_results = []
        
    def perform_comprehensive_tests(self, your_returns: pd.Series, 
                                  benchmark_returns: pd.Series,
                                  significance_level: float = 0.05) -> Dict:
        """
        Perform comprehensive statistical tests.
        
        Args:
            your_returns: Your portfolio returns
            benchmark_returns: Benchmark returns
            significance_level: Significance level for tests
            
        Returns:
            Dictionary with all test results
        """
        if your_returns.empty or benchmark_returns.empty:
            return {'error': 'Insufficient data for statistical tests'}
        
        # Align returns
        common_dates = your_returns.index.intersection(benchmark_returns.index)
        your_returns = your_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        if len(common_dates) < 30:
            logger.warning("Limited data points for statistical tests")
        
        test_results = {
            'data_info': {
                'sample_size': len(common_dates),
                'your_mean_return': your_returns.mean(),
                'benchmark_mean_return': benchmark_returns.mean(),
                'your_std': your_returns.std(),
                'benchmark_std': benchmark_returns.std()
            },
            'normality_tests': self._test_normality(your_returns, benchmark_returns),
            'mean_comparison_tests': self._test_mean_differences(your_returns, benchmark_returns, significance_level),
            'variance_tests': self._test_variance_differences(your_returns, benchmark_returns, significance_level),
            'correlation_tests': self._test_correlations(your_returns, benchmark_returns, significance_level),
            'distribution_tests': self._test_distributions(your_returns, benchmark_returns),
            'time_series_tests': self._test_time_series_properties(your_returns, benchmark_returns),
            'summary': {}
        }
        
        # Generate summary
        test_results['summary'] = self._generate_test_summary(test_results)
        
        # Store results
        self.test_results.append({
            'timestamp': pd.Timestamp.now(),
            'results': test_results
        })
        
        return test_results
    
    def test_outperformance_significance(self, your_returns: pd.Series,
                                       benchmark_returns: pd.Series,
                                       test_type: str = 'both') -> Dict:
        """
        Test if outperformance is statistically significant.
        
        Args:
            your_returns: Your portfolio returns
            benchmark_returns: Benchmark returns
            test_type: Type of test ('parametric', 'nonparametric', 'both')
            
        Returns:
            Dictionary with outperformance test results
        """
        if your_returns.empty or benchmark_returns.empty:
            return {'error': 'Insufficient data for outperformance tests'}
        
        # Align returns
        common_dates = your_returns.index.intersection(benchmark_returns.index)
        your_returns = your_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        results = {
            'sample_size': len(common_dates),
            'mean_difference': your_returns.mean() - benchmark_returns.mean(),
            'tests': {}
        }
        
        if test_type in ['parametric', 'both']:
            # T-test for mean difference
            try:
                t_stat, p_value = ttest_ind(your_returns, benchmark_returns)
                results['tests']['t_test'] = {
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_t_test(t_stat, p_value)
                }
            except Exception as e:
                results['tests']['t_test'] = {'error': str(e)}
        
        if test_type in ['nonparametric', 'both']:
            # Wilcoxon rank-sum test (Mann-Whitney U test)
            try:
                u_stat, p_value = mannwhitneyu(your_returns, benchmark_returns, 
                                             alternative='two-sided')
                results['tests']['mann_whitney'] = {
                    'statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_mann_whitney(u_stat, p_value)
                }
            except Exception as e:
                results['tests']['mann_whitney'] = {'error': str(e)}
        
        # Overall significance
        significant_tests = [test for test in results['tests'].values() 
                           if isinstance(test, dict) and test.get('significant', False)]
        results['overall_significant'] = len(significant_tests) > 0
        
        return results
    
    def test_risk_adjusted_performance(self, your_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     risk_free_rate: float = 0.05) -> Dict:
        """
        Test risk-adjusted performance metrics.
        
        Args:
            your_returns: Your portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Dictionary with risk-adjusted test results
        """
        if your_returns.empty or benchmark_returns.empty:
            return {'error': 'Insufficient data for risk-adjusted tests'}
        
        # Align returns
        common_dates = your_returns.index.intersection(benchmark_returns.index)
        your_returns = your_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate risk-adjusted metrics
        your_sharpe = (your_returns.mean() * 252 - risk_free_rate) / (your_returns.std() * np.sqrt(252))
        benchmark_sharpe = (benchmark_returns.mean() * 252 - risk_free_rate) / (benchmark_returns.std() * np.sqrt(252))
        
        # Information ratio
        excess_returns = your_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Beta calculation
        covariance = np.cov(your_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation
        alpha = (your_returns.mean() * 252) - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))
        
        results = {
            'your_sharpe_ratio': your_sharpe,
            'benchmark_sharpe_ratio': benchmark_sharpe,
            'sharpe_difference': your_sharpe - benchmark_sharpe,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'risk_metrics': {
                'your_volatility': your_returns.std() * np.sqrt(252),
                'benchmark_volatility': benchmark_returns.std() * np.sqrt(252),
                'tracking_error': excess_returns.std() * np.sqrt(252)
            }
        }
        
        # Test significance of alpha
        if len(excess_returns) > 1:
            alpha_t_stat = alpha / (excess_returns.std() / np.sqrt(len(excess_returns)))
            alpha_p_value = 2 * (1 - stats.t.cdf(abs(alpha_t_stat), len(excess_returns) - 1))
            
            results['alpha_significance'] = {
                't_statistic': alpha_t_stat,
                'p_value': alpha_p_value,
                'significant': alpha_p_value < 0.05
            }
        
        return results
    
    def _test_normality(self, your_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Test normality of return distributions."""
        normality_tests = {}
        
        # Jarque-Bera test
        try:
            jb_stat_your, jb_p_your = jarque_bera(your_returns)
            jb_stat_bench, jb_p_bench = jarque_bera(benchmark_returns)
            
            normality_tests['jarque_bera'] = {
                'your_returns': {
                    'statistic': jb_stat_your,
                    'p_value': jb_p_your,
                    'normal': jb_p_your > 0.05
                },
                'benchmark_returns': {
                    'statistic': jb_stat_bench,
                    'p_value': jb_p_bench,
                    'normal': jb_p_bench > 0.05
                }
            }
        except Exception as e:
            normality_tests['jarque_bera'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat_your, ks_p_your = kstest(your_returns, 'norm', 
                                           args=(your_returns.mean(), your_returns.std()))
            ks_stat_bench, ks_p_bench = kstest(benchmark_returns, 'norm',
                                             args=(benchmark_returns.mean(), benchmark_returns.std()))
            
            normality_tests['kolmogorov_smirnov'] = {
                'your_returns': {
                    'statistic': ks_stat_your,
                    'p_value': ks_p_your,
                    'normal': ks_p_your > 0.05
                },
                'benchmark_returns': {
                    'statistic': ks_stat_bench,
                    'p_value': ks_p_bench,
                    'normal': ks_p_bench > 0.05
                }
            }
        except Exception as e:
            normality_tests['kolmogorov_smirnov'] = {'error': str(e)}
        
        return normality_tests
    
    def _test_mean_differences(self, your_returns: pd.Series, benchmark_returns: pd.Series,
                             significance_level: float) -> Dict:
        """Test differences in means."""
        mean_tests = {}
        
        # Independent t-test
        try:
            t_stat, p_value = ttest_ind(your_returns, benchmark_returns)
            mean_tests['independent_t_test'] = {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'interpretation': self._interpret_t_test(t_stat, p_value)
            }
        except Exception as e:
            mean_tests['independent_t_test'] = {'error': str(e)}
        
        # Paired t-test (if same time periods)
        try:
            if len(your_returns) == len(benchmark_returns):
                t_stat, p_value = stats.ttest_rel(your_returns, benchmark_returns)
                mean_tests['paired_t_test'] = {
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < significance_level,
                    'interpretation': self._interpret_t_test(t_stat, p_value)
                }
        except Exception as e:
            mean_tests['paired_t_test'] = {'error': str(e)}
        
        return mean_tests
    
    def _test_variance_differences(self, your_returns: pd.Series, benchmark_returns: pd.Series,
                                 significance_level: float) -> Dict:
        """Test differences in variances."""
        variance_tests = {}
        
        # F-test for equal variances
        try:
            f_stat = your_returns.var() / benchmark_returns.var()
            df1 = len(your_returns) - 1
            df2 = len(benchmark_returns) - 1
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            
            variance_tests['f_test'] = {
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'interpretation': f"Variances are {'significantly different' if p_value < significance_level else 'not significantly different'}"
            }
        except Exception as e:
            variance_tests['f_test'] = {'error': str(e)}
        
        # Levene's test (more robust to non-normality)
        try:
            levene_stat, levene_p = stats.levene(your_returns, benchmark_returns)
            variance_tests['levene_test'] = {
                'statistic': levene_stat,
                'p_value': levene_p,
                'significant': levene_p < significance_level,
                'interpretation': f"Variances are {'significantly different' if levene_p < significance_level else 'not significantly different'}"
            }
        except Exception as e:
            variance_tests['levene_test'] = {'error': str(e)}
        
        return variance_tests
    
    def _test_correlations(self, your_returns: pd.Series, benchmark_returns: pd.Series,
                         significance_level: float) -> Dict:
        """Test correlations between returns."""
        correlation_tests = {}
        
        # Pearson correlation
        try:
            pearson_corr, pearson_p = stats.pearsonr(your_returns, benchmark_returns)
            correlation_tests['pearson'] = {
                'correlation': pearson_corr,
                'p_value': pearson_p,
                'significant': pearson_p < significance_level,
                'interpretation': f"Correlation is {'significant' if pearson_p < significance_level else 'not significant'}"
            }
        except Exception as e:
            correlation_tests['pearson'] = {'error': str(e)}
        
        # Spearman correlation
        try:
            spearman_corr, spearman_p = stats.spearmanr(your_returns, benchmark_returns)
            correlation_tests['spearman'] = {
                'correlation': spearman_corr,
                'p_value': spearman_p,
                'significant': spearman_p < significance_level,
                'interpretation': f"Rank correlation is {'significant' if spearman_p < significance_level else 'not significant'}"
            }
        except Exception as e:
            correlation_tests['spearman'] = {'error': str(e)}
        
        return correlation_tests
    
    def _test_distributions(self, your_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Test distribution properties."""
        distribution_tests = {}
        
        # Skewness and Kurtosis
        distribution_tests['moments'] = {
            'your_returns': {
                'skewness': stats.skew(your_returns),
                'kurtosis': stats.kurtosis(your_returns),
                'excess_kurtosis': stats.kurtosis(your_returns, fisher=True)
            },
            'benchmark_returns': {
                'skewness': stats.skew(benchmark_returns),
                'kurtosis': stats.kurtosis(benchmark_returns),
                'excess_kurtosis': stats.kurtosis(benchmark_returns, fisher=True)
            }
        }
        
        # Distribution comparison
        try:
            ks_stat, ks_p = kstest(your_returns, benchmark_returns)
            distribution_tests['distribution_comparison'] = {
                'ks_statistic': ks_stat,
                'p_value': ks_p,
                'same_distribution': ks_p > 0.05,
                'interpretation': f"Distributions are {'the same' if ks_p > 0.05 else 'significantly different'}"
            }
        except Exception as e:
            distribution_tests['distribution_comparison'] = {'error': str(e)}
        
        return distribution_tests
    
    def _test_time_series_properties(self, your_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Test time series properties."""
        time_series_tests = {}
        
        # Autocorrelation (Ljung-Box test)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            ljung_your = acorr_ljungbox(your_returns, lags=10, return_df=True)
            ljung_bench = acorr_ljungbox(benchmark_returns, lags=10, return_df=True)
            
            time_series_tests['autocorrelation'] = {
                'your_returns': {
                    'ljung_box_p': ljung_your['lb_pvalue'].iloc[-1],
                    'autocorrelated': ljung_your['lb_pvalue'].iloc[-1] < 0.05
                },
                'benchmark_returns': {
                    'ljung_box_p': ljung_bench['lb_pvalue'].iloc[-1],
                    'autocorrelated': ljung_bench['lb_pvalue'].iloc[-1] < 0.05
                }
            }
        except ImportError:
            time_series_tests['autocorrelation'] = {'error': 'statsmodels not available'}
        except Exception as e:
            time_series_tests['autocorrelation'] = {'error': str(e)}
        
        return time_series_tests
    
    def _interpret_t_test(self, t_stat: float, p_value: float) -> str:
        """Interpret t-test results."""
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.10:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        direction = "higher" if t_stat > 0 else "lower"
        
        return f"Your returns are {direction} than benchmark ({significance}, p={p_value:.4f})"
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float) -> str:
        """Interpret Mann-Whitney U test results."""
        if p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        return f"Distributions are {significance}ly different (p={p_value:.4f})"
    
    def _generate_test_summary(self, test_results: Dict) -> Dict:
        """Generate summary of all test results."""
        summary = {
            'total_tests_performed': 0,
            'significant_tests': 0,
            'key_findings': [],
            'recommendations': []
        }
        
        # Count tests and significant results
        for test_category, tests in test_results.items():
            if isinstance(tests, dict) and test_category != 'data_info':
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and 'significant' in test_result:
                        summary['total_tests_performed'] += 1
                        if test_result['significant']:
                            summary['significant_tests'] += 1
        
        # Generate key findings
        if 'mean_comparison_tests' in test_results:
            mean_tests = test_results['mean_comparison_tests']
            if 'independent_t_test' in mean_tests and 'significant' in mean_tests['independent_t_test']:
                if mean_tests['independent_t_test']['significant']:
                    summary['key_findings'].append("Significant difference in mean returns")
        
        if 'correlation_tests' in test_results:
            corr_tests = test_results['correlation_tests']
            if 'pearson' in corr_tests and 'significant' in corr_tests['pearson']:
                if corr_tests['pearson']['significant']:
                    summary['key_findings'].append("Significant correlation with benchmark")
        
        # Generate recommendations
        if summary['significant_tests'] / max(summary['total_tests_performed'], 1) < 0.3:
            summary['recommendations'].append("Consider increasing sample size for more reliable results")
        
        return summary
    
    def get_test_history(self) -> pd.DataFrame:
        """Get history of statistical tests."""
        if not self.test_results:
            return pd.DataFrame()
        
        history_data = []
        for result in self.test_results:
            history_data.append({
                'timestamp': result['timestamp'],
                'sample_size': result['results'].get('data_info', {}).get('sample_size', 0),
                'significant_tests': result['results'].get('summary', {}).get('significant_tests', 0),
                'total_tests': result['results'].get('summary', {}).get('total_tests_performed', 0)
            })
        
        return pd.DataFrame(history_data)
    
    def clear_history(self):
        """Clear test history."""
        self.test_results.clear()
        logger.info("Statistical test history cleared")
