import pandas as pd
from src.performance_matrix.distribution_metrics.skewness import Skewness
from src.performance_matrix.distribution_metrics.kurtosis import Kurtosis
from src.performance_matrix.distribution_metrics.tail_ratio import TailRatio
from src.performance_matrix.base_metrics_group import BaseMetricsGroup


class DistributionMetricsGroup(BaseMetricsGroup):
    """Handles distribution-related performance metrics."""

    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame, risk_free_rate: float = 0.0):
        metrics = {
            'Skewness': Skewness,
            'Kurtosis': Kurtosis,
            'TailRatio': TailRatio,
        }
        super().__init__(stock_data=stock_data, market_data=market_data, risk_free_rate=risk_free_rate, metrics=metrics)
