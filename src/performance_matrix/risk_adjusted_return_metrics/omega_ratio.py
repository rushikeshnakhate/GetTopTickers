import pandas as pd

from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class OmegaRatio(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, threshold: float = 0.0):
        super().__init__(stock_data)
        self.threshold = threshold

    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()
        gains = returns[returns > self.threshold] - self.threshold
        losses = self.threshold - returns[returns < self.threshold]
        return gains.sum() / losses.sum()
