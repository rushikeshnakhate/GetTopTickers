import pandas as pd

from src.performance_matrix.return_matrix.annualized_return import AnnualizedReturn
from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.risk_metrics.maximum_drawdown import MaximumDrawdown


class CalmarRatio(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.Series, periods_per_year: int = 252):
        super().__init__(stock_data)
        self.periods_per_year = periods_per_year

    def calculate(self):
        annualized_return = AnnualizedReturn(self.stock_data, self.periods_per_year).calculate()
        max_drawdown = MaximumDrawdown(self.stock_data).calculate()
        return annualized_return / abs(max_drawdown)
