from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Kurtosis(BasePerformanceMatrix):
    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()
        return returns.kurtosis()
