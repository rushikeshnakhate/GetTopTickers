import numpy as np
import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class AnnualizedReturn(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.DataFrame, periods_per_year: int = 252):
        super().__init__(stock_data)
        self.periods_per_year = periods_per_year

    import numpy as np

    def calculate(self):
        returns = PercentageChange(self.stock_data).calculate()

        # Number of days in data
        num_days = len(self.stock_data)

        # Cumulative return instead of mean percentage change
        cumulative_return = (1 + returns).prod() - 1

        # Correct annualization formula
        annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1

        return annualized_return
