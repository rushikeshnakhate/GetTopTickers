import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.percentage_change import PercentageChange


class RiskOfRuin(BasePerformanceMatrix):
    def __init__(self, stock_data: pd.DataFrame, ruin_level: float = 0.2):
        super().__init__(stock_data)
        self.ruin_level = ruin_level

    def calculate(self):
        cumulative_returns = (1 + PercentageChange(self.stock_data).calculate()).cumprod()
        drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
        return (drawdown <= -self.ruin_level).mean()
