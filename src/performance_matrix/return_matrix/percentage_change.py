from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix

from src.utils.constants import GlobalStockData


class PercentageChange(BasePerformanceMatrix):
    def calculate(self):
        return self.stock_data[GlobalStockData.CLOSE].pct_change().dropna()
