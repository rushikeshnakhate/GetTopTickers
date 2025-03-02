import logging

import pandas as pd
from typing import List, Dict, Optional, Type, Any

from src.cache.cache_factory import CacheFactory
from src.utils.constants import CacheType


class BaseMetricsGroup:
    """Base class for handling different groups of performance metrics."""

    def __init__(self, stock_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.0, metrics: Dict[str, Type] = None):
        self.stock_data = stock_data
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate
        self.metrics = metrics or {}
        self.cache = CacheFactory.get_cache(CacheType.PANDAS)

    def calculate(self, cache_key: str, selected_metrics: Optional[List[str]] = None) -> Dict[str | Any, Any]:
        """
        Calculate the selected performance metrics.

        :param cache_key: Unique key for caching results.
        :param selected_metrics: List of metric names to calculate.
        :return: Dictionary with calculated metric values.
        """
        # Append the derived class name to the cache_key for differentiation
        cache_key = f"{self.__class__.__name__}_{cache_key}"
        # Check if result is cached
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            return cached_results

        results = {}
        selected_metrics = selected_metrics or list(self.metrics.keys())

        for metric_name in selected_metrics:
            if metric_name in self.metrics:
                metric_class = self.metrics[metric_name]

                # Determine required arguments dynamically
                metric_args = {"stock_data": self.stock_data}
                if "market_data" in metric_class.__init__.__code__.co_varnames and self.market_data is not None:
                    metric_args["market_data"] = self.market_data
                if "risk_free_rate" in metric_class.__init__.__code__.co_varnames:
                    metric_args["risk_free_rate"] = self.risk_free_rate

                # Compute metric
                metric = metric_class(**metric_args)
                results[metric_name] = metric.calculate()
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

        # Cache the results before returning
        self.cache.set(cache_key, results)
        return results
