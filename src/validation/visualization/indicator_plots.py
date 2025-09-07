"""
Indicator Plots for visualizing technical indicator results.

This module provides plotting capabilities for technical indicators
used in the ticker selection process.
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


class IndicatorPlotter:
    """Create plots for technical indicators and their results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the IndicatorPlotter.
        
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
    
    def plot_indicator_performance(self, indicator_results: Dict) -> plt.Figure:
        """
        Plot performance of different indicators.
        
        Args:
            indicator_results: Dictionary with indicator performance results
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Technical Indicator Performance Analysis', fontsize=16, fontweight='bold')
        
        if not indicator_results:
            logger.warning("No indicator results available for plotting")
            return fig
        
        # Extract data
        indicators = list(indicator_results.keys())
        if not indicators:
            return fig
        
        # 1. Indicator hit rates
        ax1 = axes[0, 0]
        hit_rates = [indicator_results[ind].get('hit_rate', 0) for ind in indicators]
        
        bars = ax1.bar(indicators, hit_rates, color=self.colors[:len(indicators)])
        ax1.set_title('Indicator Hit Rates')
        ax1.set_ylabel('Hit Rate')
        ax1.set_xticklabels(indicators, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, hit_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        # 2. Indicator accuracy
        ax2 = axes[0, 1]
        accuracies = [indicator_results[ind].get('accuracy', 0) for ind in indicators]
        
        bars = ax2.bar(indicators, accuracies, color=self.colors[:len(indicators)])
        ax2.set_title('Indicator Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticklabels(indicators, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        # 3. Indicator frequency of use
        ax3 = axes[1, 0]
        frequencies = [indicator_results[ind].get('frequency', 0) for ind in indicators]
        
        bars = ax3.bar(indicators, frequencies, color=self.colors[:len(indicators)])
        ax3.set_title('Indicator Usage Frequency')
        ax3.set_ylabel('Frequency')
        ax3.set_xticklabels(indicators, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, frequencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 4. Indicator performance score
        ax4 = axes[1, 1]
        scores = [indicator_results[ind].get('performance_score', 0) for ind in indicators]
        
        bars = ax4.bar(indicators, scores, color=self.colors[:len(indicators)])
        ax4.set_title('Overall Performance Score')
        ax4.set_ylabel('Score')
        ax4.set_xticklabels(indicators, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_indicator_correlation(self, indicator_correlations: pd.DataFrame) -> plt.Figure:
        """
        Plot correlation matrix of indicators.
        
        Args:
            indicator_correlations: DataFrame with indicator correlations
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(indicator_correlations, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Indicator Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def plot_indicator_signals(self, price_data: pd.DataFrame, 
                              indicator_data: Dict, 
                              ticker: str) -> plt.Figure:
        """
        Plot price data with indicator signals.
        
        Args:
            price_data: DataFrame with price data
            indicator_data: Dictionary with indicator data
            ticker: Ticker symbol
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{ticker} - Price and Indicator Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price chart with moving averages
        ax1 = axes[0]
        ax1.plot(price_data.index, price_data['Close'], label='Close Price', color='black', linewidth=1)
        
        # Add moving averages if available
        if 'SMA_20' in indicator_data:
            ax1.plot(price_data.index, indicator_data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in indicator_data:
            ax1.plot(price_data.index, indicator_data['SMA_50'], label='SMA 50', alpha=0.7)
        
        ax1.set_title('Price Chart with Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[1]
        if 'RSI' in indicator_data:
            ax2.plot(price_data.index, indicator_data['RSI'], label='RSI', color=self.colors[0])
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax2.set_ylim(0, 100)
        
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axes[2]
        if 'MACD' in indicator_data and 'MACD_Signal' in indicator_data:
            ax3.plot(price_data.index, indicator_data['MACD'], label='MACD', color=self.colors[1])
            ax3.plot(price_data.index, indicator_data['MACD_Signal'], label='Signal', color=self.colors[2])
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax3.set_title('MACD Indicator')
        ax3.set_ylabel('MACD')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_strategy_performance(self, strategy_results: Dict) -> plt.Figure:
        """
        Plot strategy performance comparison.
        
        Args:
            strategy_results: Dictionary with strategy results
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        if not strategy_results:
            logger.warning("No strategy results available for plotting")
            return fig
        
        strategies = list(strategy_results.keys())
        if not strategies:
            return fig
        
        # 1. Strategy returns
        ax1 = axes[0, 0]
        returns = [strategy_results[strategy].get('total_return', 0) for strategy in strategies]
        
        bars = ax1.bar(strategies, returns, color=self.colors[:len(strategies)])
        ax1.set_title('Strategy Total Returns')
        ax1.set_ylabel('Total Return')
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        # 2. Strategy Sharpe ratios
        ax2 = axes[0, 1]
        sharpes = [strategy_results[strategy].get('sharpe_ratio', 0) for strategy in strategies]
        
        bars = ax2.bar(strategies, sharpes, color=self.colors[:len(strategies)])
        ax2.set_title('Strategy Sharpe Ratios')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sharpes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Strategy win rates
        ax3 = axes[1, 0]
        win_rates = [strategy_results[strategy].get('win_rate', 0) for strategy in strategies]
        
        bars = ax3.bar(strategies, win_rates, color=self.colors[:len(strategies)])
        ax3.set_title('Strategy Win Rates')
        ax3.set_ylabel('Win Rate')
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, win_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        # 4. Strategy max drawdowns
        ax4 = axes[1, 1]
        max_dds = [strategy_results[strategy].get('max_drawdown', 0) for strategy in strategies]
        
        bars = ax4.bar(strategies, max_dds, color=self.colors[:len(strategies)])
        ax4.set_title('Strategy Maximum Drawdowns')
        ax4.set_ylabel('Max Drawdown')
        ax4.set_xticklabels(strategies, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, max_dds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='top')
        
        plt.tight_layout()
        return fig
    
    def plot_indicator_distribution(self, indicator_values: Dict) -> plt.Figure:
        """
        Plot distribution of indicator values.
        
        Args:
            indicator_values: Dictionary with indicator values
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Indicator Value Distributions', fontsize=16, fontweight='bold')
        
        indicators = list(indicator_values.keys())
        if len(indicators) == 0:
            return fig
        
        # Plot distributions for up to 4 indicators
        for i, indicator in enumerate(indicators[:4]):
            ax = axes[i//2, i%2]
            
            values = indicator_values[indicator]
            if isinstance(values, (list, np.ndarray)):
                ax.hist(values, bins=30, alpha=0.7, color=self.colors[i])
                ax.set_title(f'{indicator} Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_signal_analysis(self, signal_data: Dict) -> plt.Figure:
        """
        Plot signal analysis results.
        
        Args:
            signal_data: Dictionary with signal analysis data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Signal Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Signal frequency
        ax1 = axes[0, 0]
        if 'signal_frequency' in signal_data:
            signals = list(signal_data['signal_frequency'].keys())
            frequencies = list(signal_data['signal_frequency'].values())
            
            bars = ax1.bar(signals, frequencies, color=self.colors[:len(signals)])
            ax1.set_title('Signal Frequency')
            ax1.set_ylabel('Frequency')
            ax1.set_xticklabels(signals, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
        
        # 2. Signal accuracy
        ax2 = axes[0, 1]
        if 'signal_accuracy' in signal_data:
            signals = list(signal_data['signal_accuracy'].keys())
            accuracies = list(signal_data['signal_accuracy'].values())
            
            bars = ax2.bar(signals, accuracies, color=self.colors[:len(signals)])
            ax2.set_title('Signal Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticklabels(signals, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # 3. Signal timing
        ax3 = axes[1, 0]
        if 'signal_timing' in signal_data:
            timing_data = signal_data['signal_timing']
            # This would plot timing analysis
            ax3.text(0.5, 0.5, 'Signal Timing Analysis', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Signal Timing Analysis')
        
        # 4. Signal correlation
        ax4 = axes[1, 1]
        if 'signal_correlation' in signal_data:
            correlation_data = signal_data['signal_correlation']
            # This would plot correlation analysis
            ax4.text(0.5, 0.5, 'Signal Correlation Analysis', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Signal Correlation Analysis')
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, plots: Dict, output_dir: str = "indicator_plots") -> Dict:
        """
        Save all indicator plots to files.
        
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
