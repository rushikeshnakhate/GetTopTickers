from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Volatility(BasePerformanceMatrix):
    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()
        return returns.std()
