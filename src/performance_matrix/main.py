import logging
from typing import Optional, List, Dict, Any

import pandas as pd

from src.performance_matrix.benchmark_relative_metrics.benchmark_relative_metrics_group import \
    BenchmarkRelativeMetricsGroup
from src.performance_matrix.distribution_metrics.distribution_metrics_group import DistributionMetricsGroup
from src.performance_matrix.return_matrix.return_metrics_group import ReturnMetricsGroup
from src.performance_matrix.risk_adjusted_return_metrics.risk_adjusted_metrics_group import RiskAdjustedMetricsGroup
from src.performance_matrix.risk_metrics.return_metrics_group import RiskMetricsGroup
from src.performance_matrix.trade_metrics.trade_metrics_group import TradeMetricsGroup
from src.service.main import dataFetcher
from typing import Optional, List, Dict

import pandas as pd

from src.performance_matrix.benchmark_relative_metrics.benchmark_relative_metrics_group import \
    BenchmarkRelativeMetricsGroup
from src.performance_matrix.distribution_metrics.distribution_metrics_group import DistributionMetricsGroup
from src.performance_matrix.return_matrix.return_metrics_group import ReturnMetricsGroup
from src.performance_matrix.risk_adjusted_return_metrics.risk_adjusted_metrics_group import RiskAdjustedMetricsGroup
from src.performance_matrix.risk_metrics.return_metrics_group import RiskMetricsGroup
from src.performance_matrix.trade_metrics.trade_metrics_group import TradeMetricsGroup
from src.service.main import dataFetcher
from src.utils.constants import GLobalColumnName
from src.utils.utils import to_dataframe


class PerformanceMatrixFactory:
    """Factory class to calculate multiple performance metrics."""

    def __init__(self, stock_data: pd.Series, market_data: pd.Series = None, risk_free_rate: float = 0.0):
        """Initialize with stock data, market data, and risk-free rate."""
        self.stock_data = stock_data
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate

        # Store metric groups in a dictionary
        self.metric_groups = {
            "benchmark_relative": BenchmarkRelativeMetricsGroup(stock_data, market_data),
            "distribution": DistributionMetricsGroup(stock_data),
            "return": ReturnMetricsGroup(stock_data),
            "risk": RiskMetricsGroup(stock_data),
            "risk_adjusted": RiskAdjustedMetricsGroup(stock_data, market_data),
            "trade": TradeMetricsGroup(stock_data),
        }

    def calculate(self, key: str, group_name: Optional[str] = None,
                  selected_metrics: Optional[List[str]] = None) -> dict[Any, Any]:
        """
        Calculate performance metrics.

        :param key: Unique key for caching (e.g., ticker_start_end).
        :param group_name: Optional, specifies a metric group to calculate.
        :param selected_metrics: Optional, list of specific metrics.
        :return: DataFrame with metric results.
        """

        # Determine which groups to calculate
        groups_to_calculate = (
            {group_name: self.metric_groups[group_name]}
            if group_name and group_name in self.metric_groups
            else self.metric_groups
        )

        results = {}
        for name, group in groups_to_calculate.items():
            group_results = group.calculate(cache_key=key, selected_metrics=selected_metrics)
            results.update(group_results)
        return results


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
    cache_key = f"{ticker}_{start_date}_{end_date}"
    # Fetch stock data
    stock_data = dataFetcher.get_close_price_service(ticker=ticker, start_date=start_date, end_date=end_date)
    market_data = dataFetcher.get_close_price_service(ticker=GLobalColumnName.ticker_nifty50, start_date=start_date,
                                                      end_date=end_date)

    if stock_data.empty or market_data.empty:
        logging.error(
            "get performance matrix failed for cache_key={},"
            "market_data={},stock_data={} is None".format(cache_key, market_data.shape, stock_data.shape))
        # Calculate performance metrics
        return pd.DataFrame
    factory = PerformanceMatrixFactory(stock_data, market_data=market_data, risk_free_rate=0.02)
    results = factory.calculate(cache_key, group_name, selected_metrics)
    return to_dataframe(column_name=GLobalColumnName.Ticker, column_value=ticker, results=results)
