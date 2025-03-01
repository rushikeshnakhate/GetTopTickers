import pandas as pd

from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Beta(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, market_data: pd.Series):
        super().__init__(stock_data)
        self.market_data = market_data

    def calculate(self):
        stock_returns = PercentageChange(self.stock_data).calculate()
        market_returns = PercentageChange(self.market_data).calculate()
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance
