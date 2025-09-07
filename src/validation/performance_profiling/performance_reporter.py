"""
Performance Reporter for generating comprehensive performance reports.

This module provides capabilities to generate detailed performance reports
combining execution time, memory usage, and accuracy metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    def __init__(self):
        """Initialize the PerformanceReporter."""
        self.reports = []
        
    def generate_comprehensive_report(self, 
                                    execution_stats: List[Dict],
                                    memory_stats: List[Dict],
                                    accuracy_stats: List[Dict],
                                    benchmark_results: Dict) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            execution_stats: List of execution statistics
            memory_stats: List of memory statistics
            accuracy_stats: List of accuracy statistics
            benchmark_results: Benchmark comparison results
            
        Returns:
            Dictionary with comprehensive report
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_performance',
                'version': '1.0'
            },
            'execution_performance': self._analyze_execution_performance(execution_stats),
            'memory_performance': self._analyze_memory_performance(memory_stats),
            'accuracy_performance': self._analyze_accuracy_performance(accuracy_stats),
            'benchmark_comparison': self._analyze_benchmark_results(benchmark_results),
            'summary': {}
        }
        
        # Generate executive summary
        report['summary'] = self._generate_executive_summary(report)
        
        return report
    
    def generate_execution_report(self, execution_stats: List[Dict]) -> Dict:
        """
        Generate execution performance report.
        
        Args:
            execution_stats: List of execution statistics
            
        Returns:
            Dictionary with execution report
        """
        if not execution_stats:
            return {'error': 'No execution statistics provided'}
        
        df = pd.DataFrame(execution_stats)
        
        # Function-level analysis
        function_analysis = df.groupby('function').agg({
            'execution_time': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'memory_used': ['mean', 'median', 'std', 'min', 'max'],
            'success': ['mean', 'sum']
        }).round(4)
        
        # Overall statistics
        overall_stats = {
            'total_functions_called': len(df),
            'successful_calls': df['success'].sum(),
            'failed_calls': (~df['success']).sum(),
            'success_rate': df['success'].mean(),
            'total_execution_time': df['execution_time'].sum(),
            'average_execution_time': df['execution_time'].mean(),
            'median_execution_time': df['execution_time'].median(),
            'max_execution_time': df['execution_time'].max(),
            'min_execution_time': df['execution_time'].min(),
            'execution_time_std': df['execution_time'].std(),
            'total_memory_used': df['memory_used'].sum(),
            'average_memory_used': df['memory_used'].mean(),
            'max_memory_used': df['memory_used'].max()
        }
        
        # Performance trends
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df_sorted = df.sort_values('timestamp')
        
        # Calculate performance trends
        if len(df_sorted) > 1:
            time_trend = np.polyfit(range(len(df_sorted)), df_sorted['execution_time'], 1)[0]
            memory_trend = np.polyfit(range(len(df_sorted)), df_sorted['memory_used'], 1)[0]
        else:
            time_trend = 0
            memory_trend = 0
        
        return {
            'overall_statistics': overall_stats,
            'function_analysis': function_analysis.to_dict(),
            'performance_trends': {
                'execution_time_trend': time_trend,
                'memory_usage_trend': memory_trend
            },
            'detailed_data': df.to_dict('records')
        }
    
    def generate_memory_report(self, memory_stats: List[Dict]) -> Dict:
        """
        Generate memory performance report.
        
        Args:
            memory_stats: List of memory statistics
            
        Returns:
            Dictionary with memory report
        """
        if not memory_stats:
            return {'error': 'No memory statistics provided'}
        
        df = pd.DataFrame(memory_stats)
        
        # Memory usage analysis
        memory_analysis = {
            'peak_memory_mb': df['rss_mb'].max() if 'rss_mb' in df.columns else 0,
            'min_memory_mb': df['rss_mb'].min() if 'rss_mb' in df.columns else 0,
            'avg_memory_mb': df['rss_mb'].mean() if 'rss_mb' in df.columns else 0,
            'memory_volatility': df['rss_mb'].std() if 'rss_mb' in df.columns else 0,
            'memory_growth_rate': self._calculate_memory_growth_rate(df),
            'memory_leak_indicators': self._detect_memory_leaks(df)
        }
        
        return {
            'memory_analysis': memory_analysis,
            'detailed_data': df.to_dict('records')
        }
    
    def generate_accuracy_report(self, accuracy_stats: List[Dict]) -> Dict:
        """
        Generate accuracy performance report.
        
        Args:
            accuracy_stats: List of accuracy statistics
            
        Returns:
            Dictionary with accuracy report
        """
        if not accuracy_stats:
            return {'error': 'No accuracy statistics provided'}
        
        df = pd.DataFrame(accuracy_stats)
        
        # Accuracy analysis
        accuracy_analysis = {
            'avg_hit_rate': df['hit_rate'].mean() if 'hit_rate' in df.columns else 0,
            'avg_precision': df['precision'].mean() if 'precision' in df.columns else 0,
            'avg_recall': df['recall'].mean() if 'recall' in df.columns else 0,
            'avg_f1_score': df['f1_score'].mean() if 'f1_score' in df.columns else 0,
            'best_hit_rate': df['hit_rate'].max() if 'hit_rate' in df.columns else 0,
            'worst_hit_rate': df['hit_rate'].min() if 'hit_rate' in df.columns else 0,
            'consistency_score': 1 - df['hit_rate'].std() if 'hit_rate' in df.columns else 0
        }
        
        return {
            'accuracy_analysis': accuracy_analysis,
            'detailed_data': df.to_dict('records')
        }
    
    def save_report_to_file(self, report: Dict, filename: str, 
                          format: str = 'json') -> bool:
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            filename: Output filename
            format: Output format ('json', 'csv', 'html')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format == 'csv':
                # Convert report to CSV format
                df = self._convert_report_to_dataframe(report)
                df.to_csv(filename, index=False)
            
            elif format == 'html':
                html_content = self._convert_report_to_html(report)
                with open(filename, 'w') as f:
                    f.write(html_content)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Report saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report to {filename}: {str(e)}")
            return False
    
    def _analyze_execution_performance(self, execution_stats: List[Dict]) -> Dict:
        """Analyze execution performance statistics."""
        if not execution_stats:
            return {}
        
        df = pd.DataFrame(execution_stats)
        
        return {
            'total_calls': len(df),
            'success_rate': df['success'].mean(),
            'avg_execution_time': df['execution_time'].mean(),
            'total_execution_time': df['execution_time'].sum(),
            'avg_memory_used': df['memory_used'].mean(),
            'function_breakdown': df.groupby('function').agg({
                'execution_time': ['count', 'mean', 'sum'],
                'success': 'mean'
            }).to_dict()
        }
    
    def _analyze_memory_performance(self, memory_stats: List[Dict]) -> Dict:
        """Analyze memory performance statistics."""
        if not memory_stats:
            return {}
        
        df = pd.DataFrame(memory_stats)
        
        return {
            'peak_memory': df['rss_mb'].max() if 'rss_mb' in df.columns else 0,
            'avg_memory': df['rss_mb'].mean() if 'rss_mb' in df.columns else 0,
            'memory_growth_rate': self._calculate_memory_growth_rate(df),
            'memory_leak_detected': self._detect_memory_leaks(df).get('leak_detected', False)
        }
    
    def _analyze_accuracy_performance(self, accuracy_stats: List[Dict]) -> Dict:
        """Analyze accuracy performance statistics."""
        if not accuracy_stats:
            return {}
        
        df = pd.DataFrame(accuracy_stats)
        
        return {
            'avg_hit_rate': df['hit_rate'].mean() if 'hit_rate' in df.columns else 0,
            'avg_precision': df['precision'].mean() if 'precision' in df.columns else 0,
            'avg_recall': df['recall'].mean() if 'recall' in df.columns else 0,
            'avg_f1_score': df['f1_score'].mean() if 'f1_score' in df.columns else 0,
            'consistency': 1 - df['hit_rate'].std() if 'hit_rate' in df.columns else 0
        }
    
    def _analyze_benchmark_results(self, benchmark_results: Dict) -> Dict:
        """Analyze benchmark comparison results."""
        if not benchmark_results:
            return {}
        
        summary = benchmark_results.get('summary', {})
        
        return {
            'total_periods_tested': summary.get('total_periods', 0),
            'your_library_wins': summary.get('your_library_wins', 0),
            'avg_speedup': summary.get('your_library_avg_speedup', 0),
            'avg_accuracy': summary.get('your_library_avg_accuracy', 0),
            'competitor_comparisons': summary.get('competitor_comparisons', {})
        }
    
    def _generate_executive_summary(self, report: Dict) -> Dict:
        """Generate executive summary of the report."""
        exec_perf = report.get('execution_performance', {})
        mem_perf = report.get('memory_performance', {})
        acc_perf = report.get('accuracy_performance', {})
        bench_comp = report.get('benchmark_comparison', {})
        
        return {
            'overall_performance_score': self._calculate_overall_score(exec_perf, mem_perf, acc_perf),
            'key_metrics': {
                'execution_success_rate': exec_perf.get('success_rate', 0),
                'avg_execution_time': exec_perf.get('avg_execution_time', 0),
                'peak_memory_usage': mem_perf.get('peak_memory', 0),
                'selection_accuracy': acc_perf.get('avg_hit_rate', 0),
                'benchmark_outperformance': bench_comp.get('avg_speedup', 0)
            },
            'recommendations': self._generate_recommendations(exec_perf, mem_perf, acc_perf, bench_comp),
            'performance_grade': self._calculate_performance_grade(exec_perf, mem_perf, acc_perf)
        }
    
    def _calculate_memory_growth_rate(self, df: pd.DataFrame) -> float:
        """Calculate memory growth rate."""
        if 'rss_mb' not in df.columns or len(df) < 2:
            return 0.0
        
        memory_series = df['rss_mb']
        x = np.arange(len(memory_series))
        y = memory_series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _detect_memory_leaks(self, df: pd.DataFrame) -> Dict:
        """Detect potential memory leaks."""
        if 'rss_mb' not in df.columns or len(df) < 10:
            return {'leak_detected': False, 'confidence': 0.0}
        
        memory_series = df['rss_mb']
        growth_rate = self._calculate_memory_growth_rate(df)
        
        leak_detected = growth_rate > 0.1
        confidence = min(1.0, abs(growth_rate) * 10)
        
        return {
            'leak_detected': leak_detected,
            'confidence': confidence,
            'growth_rate': growth_rate
        }
    
    def _convert_report_to_dataframe(self, report: Dict) -> pd.DataFrame:
        """Convert report to DataFrame for CSV export."""
        # This would implement conversion logic
        return pd.DataFrame()
    
    def _convert_report_to_html(self, report: Dict) -> str:
        """Convert report to HTML format."""
        # This would implement HTML conversion logic
        return "<html><body>Report</body></html>"
    
    def _calculate_overall_score(self, exec_perf: Dict, mem_perf: Dict, acc_perf: Dict) -> float:
        """Calculate overall performance score."""
        # Weighted combination of different metrics
        weights = {'execution': 0.3, 'memory': 0.2, 'accuracy': 0.5}
        
        exec_score = exec_perf.get('success_rate', 0) * 100
        mem_score = max(0, 100 - mem_perf.get('peak_memory', 0) / 10)  # Penalize high memory usage
        acc_score = acc_perf.get('avg_hit_rate', 0) * 100
        
        overall_score = (weights['execution'] * exec_score + 
                        weights['memory'] * mem_score + 
                        weights['accuracy'] * acc_score)
        
        return round(overall_score, 2)
    
    def _generate_recommendations(self, exec_perf: Dict, mem_perf: Dict, 
                                acc_perf: Dict, bench_comp: Dict) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if exec_perf.get('success_rate', 0) < 0.95:
            recommendations.append("Improve error handling to increase success rate")
        
        if exec_perf.get('avg_execution_time', 0) > 10:
            recommendations.append("Optimize execution time - consider caching or algorithm improvements")
        
        if mem_perf.get('peak_memory', 0) > 1000:
            recommendations.append("Reduce memory usage - consider streaming or batch processing")
        
        if acc_perf.get('avg_hit_rate', 0) < 0.7:
            recommendations.append("Improve selection accuracy - review algorithm parameters")
        
        if bench_comp.get('avg_speedup', 0) < 1.0:
            recommendations.append("Optimize performance to outperform competitors")
        
        return recommendations
    
    def _calculate_performance_grade(self, exec_perf: Dict, mem_perf: Dict, acc_perf: Dict) -> str:
        """Calculate performance grade (A-F)."""
        overall_score = self._calculate_overall_score(exec_perf, mem_perf, acc_perf)
        
        if overall_score >= 90:
            return 'A'
        elif overall_score >= 80:
            return 'B'
        elif overall_score >= 70:
            return 'C'
        elif overall_score >= 60:
            return 'D'
        else:
            return 'F'
