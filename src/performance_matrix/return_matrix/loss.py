from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class Loss(BasePerformanceMatrix):
    def calculate(self):
        pct_change = PercentageChange(self.stock_data).calculate()
        # Calculate the loss for each period; if positive, return 0, else return the loss
        return pct_change[pct_change < 0].min()  # Maximum negative return (largest loss)
