from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Gain(BasePerformanceMatrix):
    def calculate(self):
        pct_change = PercentageChange(self.stock_data).calculate()
        return pct_change[pct_change > 0].max()
