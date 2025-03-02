import logging

import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class DownsideCaptureRatio(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, market_data: pd.Series, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.market_data = market_data

    def calculate(self):
        try:
            stock_returns = PercentageChange(self.stock_data).calculate()
            market_returns = PercentageChange(self.market_data).calculate()
            negative_market_returns = market_returns[market_returns < 0]
            stock_negative_returns = stock_returns[market_returns < 0]
            return (stock_negative_returns.mean() / negative_market_returns.mean()) * 100

        except Exception as e:
            logging.info("UpsideCaptureRatio Exception: " + str(e))
            return str(e)
