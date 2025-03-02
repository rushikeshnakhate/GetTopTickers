import logging
from typing import Any
from typing import Optional, List

import pandas as pd

from src.cache.cache_factory import CacheFactory
from src.performance_matrix.benchmark_relative_metrics.benchmark_relative_metrics_group import \
    BenchmarkRelativeMetricsGroup
from src.performance_matrix.distribution_metrics.distribution_metrics_group import DistributionMetricsGroup
from src.performance_matrix.return_matrix.return_metrics_group import ReturnMetricsGroup
from src.performance_matrix.risk_adjusted_return_metrics.risk_adjusted_metrics_group import RiskAdjustedMetricsGroup
from src.performance_matrix.risk_metrics.return_metrics_group import RiskMetricsGroup
from src.performance_matrix.trade_metrics.trade_metrics_group import TradeMetricsGroup
from src.utils.constants import GLobalColumnName, CacheType, GlobalStockData
from src.utils.utils import to_dataframe


class PerformanceMatrixFactory:
    """Factory class to calculate multiple performance metrics."""

    def __init__(self, ticker_data: pd.DataFrame, market_data: pd.DataFrame = None, risk_free_rate: float = 0.0):
        """Initialize with stock data, market data, and risk-free rate."""
        self.ticker_data = ticker_data
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate
        self.cache = CacheFactory.get_cache(CacheType.PANDAS)

        # Store metric groups in a dictionary
        self.metric_groups = {
            "benchmark_relative": BenchmarkRelativeMetricsGroup(ticker_data, market_data),
            "distribution": DistributionMetricsGroup(ticker_data, market_data),
            "return": ReturnMetricsGroup(ticker_data),
            "risk": RiskMetricsGroup(ticker_data),
            "risk_adjusted": RiskAdjustedMetricsGroup(ticker_data, market_data),
            "trade": TradeMetricsGroup(ticker_data),
        }

    def calculate(self,
                  key: str,
                  group_name: Optional[str] = None,
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
            group_results = group.calculate(cache_key=key,
                                            selected_metrics=selected_metrics)
            results.update(group_results)
        return results


def get_performance_metrics(
        ticker: str,
        ticker_data: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        group_name: Optional[str] = None,
        selected_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch stock data and calculate performance metrics.
    :param ticker_data: 
    :param market_data:
    :param ticker: Stock ticker symbol.
    :param start_date: Start date of the data.
    :param end_date: End date of the data.
    :param group_name: Name of the metric group to calculate (e.g., 'return', 'risk').
    :param selected_metrics: List of metric names to calculate. If None, calculate all metrics in the group.
    :return: DataFrame containing the performance metrics.
    """
    # Generate cache key
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if ticker_data.empty or market_data.empty:
        logging.error(
            "get performance matrix failed for cache_key={},"
            "market_data={},ticker_data={} is None".format(cache_key, market_data.shape, ticker_data.shape))
        # Calculate performance metrics
        return pd.DataFrame

    factory = PerformanceMatrixFactory(ticker_data=ticker_data,
                                       market_data=market_data,
                                       risk_free_rate=0.02)
    results = factory.calculate(cache_key, group_name, selected_metrics)
    return to_dataframe(column_name=GLobalColumnName.TICKER, column_value=ticker, results=results)


def get_performance_metrics_bulk(
        ticker_list: Optional[List[str]],
        ticker_data_df: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        group_name: Optional[str] = None,
        selected_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    :param ticker_list: 
    :param ticker_data_df: 
    :param market_data: 
    :param start_date: 
    :param end_date: 
    :param group_name: 
    :param selected_metrics: 
    :return: 
    """
    ticker_len = len(ticker_list)
    cache_key = "all_performance_matrix_{ticker_len}_{start_date}_{end_date}".format(ticker_len=ticker_len,
                                                                                     start_date=start_date,
                                                                                     end_date=end_date)
    cache = CacheFactory.get_cache(CacheType.PANDAS)
    cached_results = cache.get(cache_key)
    if cached_results is not None:
        logging.info("Returning cached data to performance_metrics_bulk for all stocks key={}".format(cache_key))
        return cached_results

    # Initialize the indicator factory and compute indicators for each ticker
    performance_list = []
    for ticker in ticker_list:
        ticker_data = ticker_data_df[ticker_data_df[GLobalColumnName.TICKER] == ticker]
        if ticker_data.empty:
            logging.warning(f"get_performance_metrics_bulk No data found for ticker={ticker}. Skipping.")
            continue

        performance_df = get_performance_metrics(ticker=ticker,
                                                 ticker_data=ticker_data,
                                                 start_date=start_date,
                                                 end_date=end_date,
                                                 market_data=market_data,
                                                 selected_metrics=selected_metrics)
        performance_list.append(performance_df)

    # Concatenate results if available
    if performance_list:
        df = pd.concat(performance_list, ignore_index=True)
    else:
        logging.warning("No performance_metrics_bulk were computed for the provided tickers.")
        df = pd.DataFrame()
    cache.set(cache_key, df)
    return df
