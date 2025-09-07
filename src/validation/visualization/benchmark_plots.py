"""
Benchmark Plots for visualizing benchmark comparison results.

This module provides specialized plotting capabilities for benchmark
comparison and market analysis.
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


class BenchmarkPlotter:
    """Create plots for benchmark comparison and market analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the BenchmarkPlotter.
        
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
    
    def plot_benchmark_comparison_detailed(self, benchmark_data: Dict) -> plt.Figure:
        """
        Plot detailed benchmark comparison.
        
        Args:
            benchmark_data: Dictionary with benchmark comparison data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Detailed Benchmark Comparison Analysis', fontsize=18, fontweight='bold')
        
        if not benchmark_data:
            logger.warning("No benchmark data available for plotting")
            return fig
        
        # Extract data
        your_performance = benchmark_data.get('your_performance', {})
        benchmark_comparisons = benchmark_data.get('benchmark_comparisons', {})
        
        if not benchmark_comparisons:
            return fig
        
        benchmarks = list(benchmark_comparisons.keys())
        
        # 1. Returns comparison with error bars
        ax1 = axes[0, 0]
        your_return = your_performance.get('annual_return', 0)
        benchmark_returns = [benchmark_comparisons[b].get('benchmark_return', 0) for b in benchmarks]
        
        x_pos = np.arange(len(benchmarks) + 1)
        returns = [your_return] + benchmark_returns
        labels = ['Your Portfolio'] + benchmarks
        
        bars = ax1.bar(x_pos, returns, color=self.colors[:len(returns)], alpha=0.8)
        ax1.set_title('Annual Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Annual Return')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Risk-adjusted returns (Sharpe ratios)
        ax2 = axes[0, 1]
        your_sharpe = your_performance.get('sharpe_ratio', 0)
        benchmark_sharpes = [benchmark_comparisons[b].get('benchmark_sharpe', 0) for b in benchmarks]
        
        sharpes = [your_sharpe] + benchmark_sharpes
        
        bars = ax2.bar(x_pos, sharpes, color=self.colors[:len(sharpes)], alpha=0.8)
        ax2.set_title('Risk-Adjusted Returns (Sharpe Ratios)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sharpes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Volatility comparison
        ax3 = axes[1, 0]
        your_vol = your_performance.get('volatility', 0)
        benchmark_vols = [benchmark_comparisons[b].get('benchmark_volatility', 0) for b in benchmarks]
        
        volatilities = [your_vol] + benchmark_vols
        
        bars = ax3.bar(x_pos, volatilities, color=self.colors[:len(volatilities)], alpha=0.8)
        ax3.set_title('Volatility Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volatility')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, volatilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Maximum drawdown comparison
        ax4 = axes[1, 1]
        your_dd = your_performance.get('max_drawdown', 0)
        benchmark_dds = [benchmark_comparisons[b].get('benchmark_max_dd', 0) for b in benchmarks]
        
        max_drawdowns = [your_dd] + benchmark_dds
        
        bars = ax4.bar(x_pos, max_drawdowns, color=self.colors[:len(max_drawdowns)], alpha=0.8)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Maximum Drawdown')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, max_drawdowns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='top', fontweight='bold')
        
        # 5. Outperformance analysis
        ax5 = axes[2, 0]
        outperformance = [benchmark_comparisons[b].get('outperformance', 0) for b in benchmarks]
        
        colors = ['green' if x > 0 else 'red' for x in outperformance]
        bars = ax5.bar(benchmarks, outperformance, color=colors, alpha=0.7)
        ax5.set_title('Outperformance vs Benchmarks', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Outperformance')
        ax5.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, outperformance):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 6. Performance ranking
        ax6 = axes[2, 1]
        # Calculate composite performance score
        performance_scores = []
        for i, benchmark in enumerate(benchmarks):
            comp = benchmark_comparisons[benchmark]
            # Simple composite score: return - volatility + sharpe
            score = (comp.get('benchmark_return', 0) - 
                    comp.get('benchmark_volatility', 0) + 
                    comp.get('benchmark_sharpe', 0))
            performance_scores.append(score)
        
        # Add your portfolio score
        your_score = (your_performance.get('annual_return', 0) - 
                     your_performance.get('volatility', 0) + 
                     your_performance.get('sharpe_ratio', 0))
        all_scores = [your_score] + performance_scores
        all_labels = ['Your Portfolio'] + benchmarks
        
        # Sort by score
        sorted_data = sorted(zip(all_labels, all_scores), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_scores = zip(*sorted_data)
        
        bars = ax6.bar(range(len(sorted_labels)), sorted_scores, 
                      color=self.colors[:len(sorted_labels)], alpha=0.8)
        ax6.set_title('Performance Ranking (Composite Score)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Composite Score')
        ax6.set_xticks(range(len(sorted_labels)))
        ax6.set_xticklabels(sorted_labels, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sorted_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_benchmark_correlation_heatmap(self, correlation_data: pd.DataFrame) -> plt.Figure:
        """
        Plot correlation heatmap between your portfolio and benchmarks.
        
        Args:
            correlation_data: DataFrame with correlation data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8},
                   fmt='.3f')
        
        ax.set_title('Portfolio-Benchmark Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_rolling_performance(self, rolling_data: Dict) -> plt.Figure:
        """
        Plot rolling performance comparison.
        
        Args:
            rolling_data: Dictionary with rolling performance data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Rolling Performance Comparison', fontsize=16, fontweight='bold')
        
        if not rolling_data:
            logger.warning("No rolling data available for plotting")
            return fig
        
        # Extract data
        dates = rolling_data.get('dates', [])
        your_returns = rolling_data.get('your_returns', [])
        benchmark_returns = rolling_data.get('benchmark_returns', {})
        
        if not dates or not your_returns:
            return fig
        
        # 1. Rolling returns
        ax1 = axes[0]
        ax1.plot(dates, your_returns, label='Your Portfolio', color=self.colors[0], linewidth=2)
        
        for i, (benchmark, returns) in enumerate(benchmark_returns.items()):
            ax1.plot(dates, returns, label=benchmark, color=self.colors[i+1], alpha=0.7)
        
        ax1.set_title('Rolling Returns Comparison')
        ax1.set_ylabel('Rolling Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe ratios
        ax2 = axes[1]
        your_sharpes = rolling_data.get('your_sharpes', [])
        benchmark_sharpes = rolling_data.get('benchmark_sharpes', {})
        
        if your_sharpes:
            ax2.plot(dates, your_sharpes, label='Your Portfolio', color=self.colors[0], linewidth=2)
        
        for i, (benchmark, sharpes) in enumerate(benchmark_sharpes.items()):
            ax2.plot(dates, sharpes, label=benchmark, color=self.colors[i+1], alpha=0.7)
        
        ax2.set_title('Rolling Sharpe Ratios')
        ax2.set_ylabel('Rolling Sharpe Ratio')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_benchmark_attribution(self, attribution_data: Dict) -> plt.Figure:
        """
        Plot performance attribution analysis.
        
        Args:
            attribution_data: Dictionary with attribution data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Attribution Analysis', fontsize=16, fontweight='bold')
        
        if not attribution_data:
            logger.warning("No attribution data available for plotting")
            return fig
        
        # 1. Sector attribution
        ax1 = axes[0, 0]
        sector_attribution = attribution_data.get('sector_attribution', {})
        if sector_attribution:
            sectors = list(sector_attribution.keys())
            contributions = list(sector_attribution.values())
            
            bars = ax1.bar(sectors, contributions, color=self.colors[:len(sectors)])
            ax1.set_title('Sector Attribution')
            ax1.set_ylabel('Contribution to Return')
            ax1.set_xticklabels(sectors, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
        
        # 2. Factor attribution
        ax2 = axes[0, 1]
        factor_attribution = attribution_data.get('factor_attribution', {})
        if factor_attribution:
            factors = list(factor_attribution.keys())
            contributions = list(factor_attribution.values())
            
            bars = ax2.bar(factors, contributions, color=self.colors[:len(factors)])
            ax2.set_title('Factor Attribution')
            ax2.set_ylabel('Contribution to Return')
            ax2.set_xticklabels(factors, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # 3. Stock selection vs allocation
        ax3 = axes[1, 0]
        selection_effect = attribution_data.get('selection_effect', 0)
        allocation_effect = attribution_data.get('allocation_effect', 0)
        
        effects = ['Selection Effect', 'Allocation Effect']
        values = [selection_effect, allocation_effect]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax3.bar(effects, values, color=colors, alpha=0.7)
        ax3.set_title('Selection vs Allocation Effect')
        ax3.set_ylabel('Contribution to Return')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk attribution
        ax4 = axes[1, 1]
        risk_attribution = attribution_data.get('risk_attribution', {})
        if risk_attribution:
            risk_sources = list(risk_attribution.keys())
            risk_contributions = list(risk_attribution.values())
            
            bars = ax4.bar(risk_sources, risk_contributions, color=self.colors[:len(risk_sources)])
            ax4.set_title('Risk Attribution')
            ax4.set_ylabel('Contribution to Risk')
            ax4.set_xticklabels(risk_sources, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_benchmark_consistency(self, consistency_data: Dict) -> plt.Figure:
        """
        Plot benchmark consistency analysis.
        
        Args:
            consistency_data: Dictionary with consistency data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Benchmark Consistency Analysis', fontsize=16, fontweight='bold')
        
        if not consistency_data:
            logger.warning("No consistency data available for plotting")
            return fig
        
        # 1. Win rate over time
        ax1 = axes[0, 0]
        win_rates = consistency_data.get('win_rates', {})
        if win_rates:
            periods = list(win_rates.keys())
            rates = list(win_rates.values())
            
            ax1.plot(periods, rates, 'o-', color=self.colors[0], linewidth=2, markersize=6)
            ax1.set_title('Win Rate Over Time')
            ax1.set_ylabel('Win Rate')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
        
        # 2. Performance consistency
        ax2 = axes[0, 1]
        performance_std = consistency_data.get('performance_std', {})
        if performance_std:
            benchmarks = list(performance_std.keys())
            stds = list(performance_std.values())
            
            bars = ax2.bar(benchmarks, stds, color=self.colors[:len(benchmarks)])
            ax2.set_title('Performance Consistency (Lower is Better)')
            ax2.set_ylabel('Standard Deviation of Returns')
            ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # 3. Outperformance frequency
        ax3 = axes[1, 0]
        outperformance_freq = consistency_data.get('outperformance_frequency', {})
        if outperformance_freq:
            benchmarks = list(outperformance_freq.keys())
            frequencies = list(outperformance_freq.values())
            
            bars = ax3.bar(benchmarks, frequencies, color=self.colors[:len(benchmarks)])
            ax3.set_title('Outperformance Frequency')
            ax3.set_ylabel('Frequency of Outperformance')
            ax3.set_xticklabels(benchmarks, rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
        
        # 4. Risk-adjusted consistency
        ax4 = axes[1, 1]
        risk_adjusted_consistency = consistency_data.get('risk_adjusted_consistency', {})
        if risk_adjusted_consistency:
            benchmarks = list(risk_adjusted_consistency.keys())
            consistency_scores = list(risk_adjusted_consistency.values())
            
            bars = ax4.bar(benchmarks, consistency_scores, color=self.colors[:len(benchmarks)])
            ax4.set_title('Risk-Adjusted Consistency')
            ax4.set_ylabel('Consistency Score')
            ax4.set_xticklabels(benchmarks, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_benchmark_plots(self, plots: Dict, output_dir: str = "benchmark_plots") -> Dict:
        """
        Save all benchmark plots to files.
        
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
        
        for plot_name, plot_fig in plots.items():
            if hasattr(plot_fig, 'savefig'):
                filename = f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(output_dir, filename)
                plot_fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(plot_fig)
                saved_files[plot_name] = filepath
        
        return saved_files
