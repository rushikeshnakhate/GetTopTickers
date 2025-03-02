import logging

import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class SortinoRatio(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, market_data: pd.Series = None, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate

    def calculate(self):
        try:
            returns = PercentageChange(self.stock_data).calculate()
            downside_returns = returns[returns < 0]
            downside_risk = downside_returns.std()
            excess_returns = returns.mean() - self.risk_free_rate
            return excess_returns / downside_risk
        except Exception as e:
            logging.error("SortinoRatio failed with error={}".format(e))
            return str(e)
