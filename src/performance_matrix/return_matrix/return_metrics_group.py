import pandas as pd

from src.performance_matrix.base_metrics_group import BaseMetricsGroup
from src.performance_matrix.return_matrix.annualized_return import AnnualizedReturn
from src.performance_matrix.return_matrix.average_daily_return import AverageDailyReturn
from src.performance_matrix.return_matrix.cumulative_return import CumulativeReturn
from src.performance_matrix.return_matrix.gain import Gain
from src.performance_matrix.return_matrix.loss import Loss
from src.performance_matrix.return_matrix.percentage_change_by_method import PercentageChangeByMethod


class ReturnMetricsGroup(BaseMetricsGroup):
    """Class to handle return-related performance metrics."""

    def __init__(self, stock_data: pd.Series, market_data: pd.Series = None, risk_free_rate: float = 0.0):
        self.stock_data = stock_data
        metrics = {
            'PercentageChange': PercentageChangeByMethod,
            'Gain': Gain,
            'Loss': Loss,
            'CumulativeReturn': CumulativeReturn,
            'AnnualizedReturn': AnnualizedReturn,
            'AverageDailyReturn': AverageDailyReturn,
        }
        super().__init__(stock_data=stock_data, market_data=market_data, risk_free_rate=risk_free_rate, metrics=metrics)
