import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.cache.cache_factory import CacheFactory
# Import all performance metric classes
from src.performance_matrix.benchmark_relative_metrics.active_return import ActiveReturn
from src.performance_matrix.benchmark_relative_metrics.alpha import Alpha
from src.performance_matrix.benchmark_relative_metrics.beta import Beta
from src.performance_matrix.benchmark_relative_metrics.downside_capture_ratio import DownsideCaptureRatio
from src.performance_matrix.benchmark_relative_metrics.information_ratio import InformationRatio
from src.performance_matrix.benchmark_relative_metrics.rsquared import RSquared
from src.performance_matrix.benchmark_relative_metrics.tracking_rrror import TrackingError
from src.performance_matrix.benchmark_relative_metrics.upside_capture_ratio import UpsideCaptureRatio
from src.performance_matrix.distribution_metrics.kurtosis import Kurtosis
from src.performance_matrix.distribution_metrics.skewness import Skewness
from src.performance_matrix.distribution_metrics.tail_ratio import TailRatio
from src.performance_matrix.return_matrix.annualized_return import AnnualizedReturn
from src.performance_matrix.return_matrix.average_daily_return import AverageDailyReturn
from src.performance_matrix.return_matrix.cumulative_return import CumulativeReturn
from src.performance_matrix.return_matrix.gain import Gain
from src.performance_matrix.return_matrix.loss import Loss
from src.performance_matrix.return_matrix.percentage_change import PercentageChange
from src.performance_matrix.risk_adjusted_return_metrics.calmar_ratio import CalmarRatio
from src.performance_matrix.risk_adjusted_return_metrics.omega_ratio import OmegaRatio
from src.performance_matrix.risk_adjusted_return_metrics.sharpe_ratio import SharpeRatio
from src.performance_matrix.risk_adjusted_return_metrics.sortino_ratio import SortinoRatio
from src.performance_matrix.risk_adjusted_return_metrics.sterling_ratio import SterlingRatio
from src.performance_matrix.risk_adjusted_return_metrics.treynor_ratio import TreynorRatio
from src.performance_matrix.risk_metrics.conditional_value_at_risk import ConditionalValueAtRisk
from src.performance_matrix.risk_metrics.maximum_drawdown import MaximumDrawdown
from src.performance_matrix.risk_metrics.risk_of_ruin import RiskOfRuin
from src.performance_matrix.risk_metrics.value_at_risk import ValueAtRisk
from src.performance_matrix.risk_metrics.volatility import Volatility
from src.performance_matrix.trade_metrics.gain_to_pain_ratio import GainToPainRatio
from src.performance_matrix.trade_metrics.loss_rate import LossRate
from src.performance_matrix.trade_metrics.profit_factor import ProfitFactor
from src.performance_matrix.trade_metrics.ulcer_index import UlcerIndex
from src.performance_matrix.trade_metrics.win_rate import WinRate
from src.service.main import dataFetcher
from src.utils.constants import CacheType


class PerformanceMatrixFactory:
    """Factory class to calculate multiple performance metrics."""

    # Group metrics into categories
    RETURN_METRICS = {
        'PercentageChange': PercentageChange,
        'Gain': Gain,
        'Loss': Loss,
        'CumulativeReturn': CumulativeReturn,
        'AnnualizedReturn': AnnualizedReturn,
        'AverageDailyReturn': AverageDailyReturn,
    }

    RISK_METRICS = {
        # 'Volatility': Volatility,
        # 'MaximumDrawdown': MaximumDrawdown,
        # 'ValueAtRisk': ValueAtRisk,
        # 'ConditionalValueAtRisk': ConditionalValueAtRisk,
        # 'RiskOfRuin': RiskOfRuin,
    }

    RISK_ADJUSTED_METRICS = {
        # 'SharpeRatio': SharpeRatio,
        # 'SortinoRatio': SortinoRatio,
        # 'CalmarRatio': CalmarRatio,
        # 'OmegaRatio': OmegaRatio,
        # 'SterlingRatio': SterlingRatio,
        # 'TreynorRatio': TreynorRatio,
    }

    BENCHMARK_RELATIVE_METRICS = {
        # 'Beta': Beta,
        # 'Alpha': Alpha,
        # 'RSquared': RSquared,
        # 'InformationRatio': InformationRatio,
        # 'TrackingError': TrackingError,
        # 'ActiveReturn': ActiveReturn,
        # 'UpsideCaptureRatio': UpsideCaptureRatio,
        # 'DownsideCaptureRatio': DownsideCaptureRatio,
    }

    DISTRIBUTION_METRICS = {
        # 'Skewness': Skewness,
        # 'Kurtosis': Kurtosis,
        # 'TailRatio': TailRatio,
    }

    TRADE_METRICS = {
        # 'WinRate': WinRate,
        # 'LossRate': LossRate,
        # 'ProfitFactor': ProfitFactor,
        # 'GainToPainRatio': GainToPainRatio,
        # 'UlcerIndex': UlcerIndex,
    }

    # Map group names to their respective metric dictionaries
    METRIC_GROUPS = {
        'return': RETURN_METRICS,
        'risk': RISK_METRICS,
        'risk_adjusted': RISK_ADJUSTED_METRICS,
        'benchmark_relative': BENCHMARK_RELATIVE_METRICS,
        'distribution': DISTRIBUTION_METRICS,
        'trade': TRADE_METRICS,
    }

    def __init__(self, stock_data: pd.Series, risk_free_rate: float = 0.0):
        """
        Initialize with stock data and risk-free rate.
        """
        self.stock_data = stock_data
        self.risk_free_rate = risk_free_rate
        self.cache = CacheFactory.get_cache(CacheType.PANDAS)

    def calculate(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            group_name: Optional[str] = None,
            selected_metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for the given stock data.
        :param ticker: Stock ticker symbol (for caching).
        :param start_date: Start date of the data (for caching).
        :param end_date: End date of the data (for caching).
        :param group_name: Name of the metric group to calculate (e.g., 'return', 'risk').
                          If None, calculate all groups.
        :param selected_metrics: List of metric names to calculate. If None, calculate all metrics in the group.
        :return: Dictionary of metric names and their values.
        """
        # Filter data based on start_date and end_date
        filtered_data = self.stock_data.loc[start_date:end_date]

        # Determine which groups to calculate
        if group_name:
            if group_name not in self.METRIC_GROUPS:
                raise ValueError(f"Unknown group: {group_name}")
            groups = {group_name: self.METRIC_GROUPS[group_name]}
        else:
            groups = self.METRIC_GROUPS

        results = {}
        for group, metrics in groups.items():
            if selected_metrics is None:
                selected_metrics = list(metrics.keys())

            for metric_name in selected_metrics:
                if metric_name in metrics:
                    metric_class = metrics[metric_name]
                    metric = metric_class(filtered_data)
                    results[metric_name] = metric.calculate()
                else:
                    raise ValueError(f"Unknown metric: {metric_name}, available metrics={metrics} for group={group}")
            selected_metrics = None  # Reset to avoid reusing old selected metrics

        for key, value in results.items():
            if isinstance(value, list):
                if len(value) == 0:
                    results[key] = np.nan  # Replace empty lists with NaN
                elif len(value) == 1:
                    results[key] = value[0]  # Convert single-element lists to scalars

        results["Ticker"] = ticker
        df = pd.DataFrame([results])
        df = df[["Ticker"] + [col for col in df.columns if col != "Ticker"]]
        return df


def get_performance_metrics(
        ticker: str,
        start_date: str,
        end_date: str,
        group_name: Optional[str] = None,
        selected_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch stock data and calculate performance metrics.
    :param ticker: Stock ticker symbol.
    :param start_date: Start date of the data.
    :param end_date: End date of the data.
    :param group_name: Name of the metric group to calculate (e.g., 'return', 'risk').
    :param selected_metrics: List of metric names to calculate. If None, calculate all metrics in the group.
    :return: DataFrame containing the performance metrics.
    """
    # Generate cache key
    cache_key = f"performance_{ticker}_{start_date}_{end_date}"
    cache = CacheFactory.get_cache(CacheType.PANDAS)

    # Check if data is cached
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logging.info(f"Returning cached performance metrics for {ticker} ({start_date} to {end_date})")
        return cached_data

    # Fetch stock data
    stock_data = dataFetcher.get_close_price_service(ticker=ticker, start_date=start_date, end_date=end_date)

    # Calculate performance metrics
    factory = PerformanceMatrixFactory(stock_data, risk_free_rate=0.02)
    df = factory.calculate(ticker, start_date, end_date, group_name, selected_metrics)

    # Cache the result
    cache.set(cache_key, df)
    return df
