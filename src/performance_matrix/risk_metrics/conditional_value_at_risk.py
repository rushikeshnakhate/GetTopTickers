import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class ConditionalValueAtRisk(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.DataFrame, confidence_level: float = 0.95):
        super().__init__(stock_data)
        self.confidence_level = confidence_level

    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()
        var = returns.quantile(1 - self.confidence_level)
        return returns[returns <= var].mean()
