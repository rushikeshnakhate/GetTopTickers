import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class ActiveReturn(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.benchmark_data = market_data

    def calculate(self):
        stock_returns = PercentageChange(self.stock_data).calculate()
        benchmark_returns = PercentageChange(self.benchmark_data).calculate()
        return stock_returns.mean() - benchmark_returns.mean()
