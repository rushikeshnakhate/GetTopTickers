import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange
from src.performance_matrix.risk_metrics.volatility import Volatility


class SharpeRatio(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, market_data: pd.Series = None, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate

    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()
        excess_returns = returns - self.risk_free_rate
        return excess_returns.mean() / Volatility(self.stock_data).calculate()
