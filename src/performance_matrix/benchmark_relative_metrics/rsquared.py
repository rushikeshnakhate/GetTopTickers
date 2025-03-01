import pandas as pd

from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class RSquared(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, market_data: pd.Series):
        super().__init__(stock_data)
        self.market_data = market_data

    def calculate(self):
        stock_returns = PercentageChange(self.stock_data).calculate()
        market_returns = PercentageChange(self.market_data).calculate()
        correlation = stock_returns.corr(market_returns)
        return correlation ** 2
