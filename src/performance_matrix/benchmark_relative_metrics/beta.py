import logging

import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Beta(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.risk_free_rate = risk_free_rate
        self.market_data = market_data

    def calculate(self):
        try:
            stock_returns = PercentageChange(self.stock_data).calculate()
            market_returns = PercentageChange(self.market_data).calculate()
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()
            return covariance / market_variance
        except Exception as ex:
            logging.error("Beta Exception{}".format(str(ex)))
            return str(ex)
