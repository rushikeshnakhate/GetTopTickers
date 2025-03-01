from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Loss(BasePerformanceMatrix):
    def calculate(self):
        pct_change = PercentageChange(self.stock_data).calculate()
        return pct_change if pct_change < 0 else 0
