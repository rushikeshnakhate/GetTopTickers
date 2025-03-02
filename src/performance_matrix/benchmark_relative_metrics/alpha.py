import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.benchmark_relative_metrics.beta import Beta
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Alpha(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate

    def calculate(self):
        stock_returns = PercentageChange(self.stock_data).calculate()
        market_returns = PercentageChange(self.market_data).calculate()
        beta = Beta(self.stock_data, self.market_data).calculate()
        excess_stock_returns = stock_returns.mean() - self.risk_free_rate
        excess_market_returns = market_returns.mean() - self.risk_free_rate
        return excess_stock_returns - (beta * excess_market_returns)
