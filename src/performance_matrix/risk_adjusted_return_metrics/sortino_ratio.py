import pandas as pd

from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class SortinoRatio(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, risk_free_rate: float = 0.0):
        super().__init__(stock_data)
        self.risk_free_rate = risk_free_rate

    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()
        downside_returns = returns[returns < 0]
        downside_risk = downside_returns.std()
        excess_returns = returns.mean() - self.risk_free_rate
        return excess_returns / downside_risk
