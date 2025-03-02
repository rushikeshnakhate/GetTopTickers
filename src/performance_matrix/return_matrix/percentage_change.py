from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix


class PercentageChange(BasePerformanceMatrix):
    def calculate(self):
        return self.stock_data['Close'].pct_change().dropna()
