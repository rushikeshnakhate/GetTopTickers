"""
Performance Plots for visualizing validation results.

This module provides comprehensive plotting capabilities for performance
visualization, addressing reviewer concerns about data presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)


class PerformancePlotter:
    """Create comprehensive performance visualization plots."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the PerformancePlotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            logger.warning(f"Style {style} not found, using default style")
    
    def create_validation_plots(self, benchmark_comparison: Dict, 
                              walk_forward_results: Dict) -> Dict:
        """
        Create comprehensive validation plots.
        
        Args:
            benchmark_comparison: Benchmark comparison results
            walk_forward_results: Walk-forward analysis results
            
        Returns:
            Dictionary with plot information
        """
        plots = {}
        
        try:
            # Performance comparison plot
            plots['performance_comparison'] = self._plot_performance_comparison(benchmark_comparison)
            
            # Walk-forward analysis plot
            plots['walk_forward'] = self._plot_walk_forward_analysis(walk_forward_results)
            
            # Risk-return scatter plot
            plots['risk_return'] = self._plot_risk_return_scatter(benchmark_comparison)
            
            # Drawdown comparison plot
            plots['drawdown_comparison'] = self._plot_drawdown_comparison(benchmark_comparison)
            
            # Performance over time plot
            plots['performance_over_time'] = self._plot_performance_over_time(walk_forward_results)
            
        except Exception as e:
            logger.error(f"Error creating validation plots: {str(e)}")
            plots['error'] = str(e)
        
        return plots
    
    def plot_benchmark_comparison(self, comparison_results: Dict) -> plt.Figure:
        """
        Plot benchmark comparison results.
        
        Args:
            comparison_results: Benchmark comparison results
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Benchmark Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        benchmarks = list(comparison_results.get('benchmark_comparisons', {}).keys())
        if not benchmarks:
            logger.warning("No benchmark data available for plotting")
            return fig
        
        # 1. Returns comparison
        ax1 = axes[0, 0]
        your_return = comparison_results.get('your_performance', {}).get('annual_return', 0)
        benchmark_returns = [comparison_results['benchmark_comparisons'][b].get('benchmark_return', 0) 
                           for b in benchmarks]
        
        x_pos = np.arange(len(benchmarks) + 1)
        returns = [your_return] + benchmark_returns
        labels = ['Your Portfolio'] + benchmarks
        
        bars = ax1.bar(x_pos, returns, color=self.colors[:len(returns)])
        ax1.set_title('Annual Returns Comparison')
        ax1.set_ylabel('Annual Return')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        # 2. Sharpe ratio comparison
        ax2 = axes[0, 1]
        your_sharpe = comparison_results.get('your_performance', {}).get('sharpe_ratio', 0)
        benchmark_sharpes = [comparison_results['benchmark_comparisons'][b].get('benchmark_sharpe', 0) 
                           for b in benchmarks]
        
        sharpes = [your_sharpe] + benchmark_sharpes
        
        bars = ax2.bar(x_pos, sharpes, color=self.colors[:len(sharpes)])
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Outperformance
        ax3 = axes[1, 0]
        outperformance = [comparison_results['benchmark_comparisons'][b].get('outperformance', 0) 
                         for b in benchmarks]
        
        colors = ['green' if x > 0 else 'red' for x in outperformance]
        bars = ax3.bar(benchmarks, outperformance, color=colors, alpha=0.7)
        ax3.set_title('Outperformance vs Benchmarks')
        ax3.set_ylabel('Outperformance')
        ax3.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, outperformance):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 4. Risk metrics
        ax4 = axes[1, 1]
        your_vol = comparison_results.get('your_performance', {}).get('volatility', 0)
        benchmark_vols = [comparison_results['benchmark_comparisons'][b].get('benchmark_volatility', 0) 
                         for b in benchmarks]
        
        volatilities = [your_vol] + benchmark_vols
        
        bars = ax4.bar(x_pos, volatilities, color=self.colors[:len(volatilities)])
        ax4.set_title('Volatility Comparison')
        ax4.set_ylabel('Volatility')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, volatilities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_walk_forward_analysis(self, walk_forward_results: Dict) -> plt.Figure:
        """
        Plot walk-forward analysis results.
        
        Args:
            walk_forward_results: Walk-forward analysis results
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
        
        detailed_results = walk_forward_results.get('detailed_results', [])
        if not detailed_results:
            logger.warning("No walk-forward data available for plotting")
            return fig
        
        # Extract data
        iterations = [r['iteration'] for r in detailed_results]
        test_returns = [r['test_performance'].get('annual_return', 0) for r in detailed_results]
        test_sharpes = [r['test_performance'].get('sharpe_ratio', 0) for r in detailed_results]
        train_returns = [r['train_performance'].get('annual_return', 0) for r in detailed_results]
        degradations = [r['performance_degradation'].get('annual_return_degradation', 0) 
                       for r in detailed_results]
        
        # 1. Test vs Train Returns
        ax1 = axes[0, 0]
        ax1.plot(iterations, train_returns, 'o-', label='Train Returns', color=self.colors[0])
        ax1.plot(iterations, test_returns, 's-', label='Test Returns', color=self.colors[1])
        ax1.set_title('Train vs Test Returns')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Annual Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Degradation
        ax2 = axes[0, 1]
        colors = ['red' if x < 0 else 'green' for x in degradations]
        bars = ax2.bar(iterations, degradations, color=colors, alpha=0.7)
        ax2.set_title('Performance Degradation (Test vs Train)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Degradation')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio Over Time
        ax3 = axes[1, 0]
        ax3.plot(iterations, test_sharpes, 'o-', color=self.colors[2])
        ax3.set_title('Test Sharpe Ratio Over Time')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # 4. Consistency Analysis
        ax4 = axes[1, 1]
        positive_periods = [1 if x > 0 else 0 for x in test_returns]
        cumulative_positive = np.cumsum(positive_periods)
        win_rate = cumulative_positive / np.arange(1, len(positive_periods) + 1)
        
        ax4.plot(iterations, win_rate, 'o-', color=self.colors[3])
        ax4.set_title('Cumulative Win Rate')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Win Rate')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_return_scatter(self, comparison_results: Dict) -> plt.Figure:
        """
        Plot risk-return scatter plot.
        
        Args:
            comparison_results: Comparison results
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        your_return = comparison_results.get('your_performance', {}).get('annual_return', 0)
        your_risk = comparison_results.get('your_performance', {}).get('volatility', 0)
        
        benchmark_comparisons = comparison_results.get('benchmark_comparisons', {})
        benchmark_returns = []
        benchmark_risks = []
        benchmark_names = []
        
        for name, comparison in benchmark_comparisons.items():
            benchmark_returns.append(comparison.get('benchmark_return', 0))
            benchmark_risks.append(comparison.get('benchmark_volatility', 0))
            benchmark_names.append(name)
        
        # Plot benchmarks
        ax.scatter(benchmark_risks, benchmark_returns, 
                  c=self.colors[1:len(benchmark_names)+1], 
                  s=100, alpha=0.7, label='Benchmarks')
        
        # Plot your portfolio
        ax.scatter(your_risk, your_return, 
                  c=self.colors[0], s=150, alpha=0.8, 
                  marker='*', label='Your Portfolio')
        
        # Add labels
        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Annual Return')
        ax.set_title('Risk-Return Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add benchmark labels
        for i, name in enumerate(benchmark_names):
            ax.annotate(name, (benchmark_risks[i], benchmark_returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add your portfolio label
        ax.annotate('Your Portfolio', (your_risk, your_return), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        return fig
    
    def plot_drawdown_comparison(self, comparison_results: Dict) -> plt.Figure:
        """
        Plot drawdown comparison.
        
        Args:
            comparison_results: Comparison results
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        your_dd = comparison_results.get('your_performance', {}).get('max_drawdown', 0)
        
        benchmark_comparisons = comparison_results.get('benchmark_comparisons', {})
        benchmark_dds = []
        benchmark_names = []
        
        for name, comparison in benchmark_comparisons.items():
            benchmark_dds.append(comparison.get('benchmark_max_dd', 0))
            benchmark_names.append(name)
        
        # Create data for plotting
        all_names = ['Your Portfolio'] + benchmark_names
        all_dds = [your_dd] + benchmark_dds
        
        # Plot bars
        colors = [self.colors[0]] + self.colors[1:len(benchmark_names)+1]
        bars = ax.bar(all_names, all_dds, color=colors, alpha=0.7)
        
        ax.set_title('Maximum Drawdown Comparison')
        ax.set_ylabel('Maximum Drawdown')
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, all_dds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2%}', ha='center', va='top')
        
        return fig
    
    def plot_performance_over_time(self, walk_forward_results: Dict) -> plt.Figure:
        """
        Plot performance over time.
        
        Args:
            walk_forward_results: Walk-forward results
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        detailed_results = walk_forward_results.get('detailed_results', [])
        if not detailed_results:
            logger.warning("No walk-forward data available for plotting")
            return fig
        
        # Extract data
        test_dates = [r['test_start'] for r in detailed_results]
        test_returns = [r['test_performance'].get('annual_return', 0) for r in detailed_results]
        
        # Plot performance over time
        ax.plot(test_dates, test_returns, 'o-', color=self.colors[0], linewidth=2, markersize=6)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Out-of-Sample Performance Over Time')
        ax.set_xlabel('Test Period Start Date')
        ax.set_ylabel('Annual Return')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(test_returns) > 1:
            x_numeric = np.arange(len(test_returns))
            z = np.polyfit(x_numeric, test_returns, 1)
            p = np.poly1d(z)
            ax.plot(test_dates, p(x_numeric), '--', color='red', alpha=0.7, label='Trend')
            ax.legend()
        
        return fig
    
    def _plot_performance_comparison(self, benchmark_comparison: Dict) -> str:
        """Create performance comparison plot (internal method)."""
        try:
            fig = self.plot_benchmark_comparison(benchmark_comparison)
            filename = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as e:
            logger.error(f"Error creating performance comparison plot: {str(e)}")
            return None
    
    def _plot_walk_forward_analysis(self, walk_forward_results: Dict) -> str:
        """Create walk-forward analysis plot (internal method)."""
        try:
            fig = self.plot_walk_forward_analysis(walk_forward_results)
            filename = f"walk_forward_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as e:
            logger.error(f"Error creating walk-forward analysis plot: {str(e)}")
            return None
    
    def _plot_risk_return_scatter(self, benchmark_comparison: Dict) -> str:
        """Create risk-return scatter plot (internal method)."""
        try:
            fig = self.plot_risk_return_scatter(benchmark_comparison)
            filename = f"risk_return_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as e:
            logger.error(f"Error creating risk-return scatter plot: {str(e)}")
            return None
    
    def _plot_drawdown_comparison(self, benchmark_comparison: Dict) -> str:
        """Create drawdown comparison plot (internal method)."""
        try:
            fig = self.plot_drawdown_comparison(benchmark_comparison)
            filename = f"drawdown_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as e:
            logger.error(f"Error creating drawdown comparison plot: {str(e)}")
            return None
    
    def _plot_performance_over_time(self, walk_forward_results: Dict) -> str:
        """Create performance over time plot (internal method)."""
        try:
            fig = self.plot_performance_over_time(walk_forward_results)
            filename = f"performance_over_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as e:
            logger.error(f"Error creating performance over time plot: {str(e)}")
            return None
    
    def save_all_plots(self, plots: Dict, output_dir: str = "plots") -> Dict:
        """
        Save all plots to files.
        
        Args:
            plots: Dictionary of plot information
            output_dir: Output directory for plots
            
        Returns:
            Dictionary with saved file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for plot_name, plot_info in plots.items():
            if isinstance(plot_info, str) and plot_info.endswith('.png'):
                # Plot already saved
                saved_files[plot_name] = plot_info
            elif hasattr(plot_info, 'savefig'):
                # Matplotlib figure object
                filename = f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(output_dir, filename)
                plot_info.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(plot_info)
                saved_files[plot_name] = filepath
        
        return saved_files
