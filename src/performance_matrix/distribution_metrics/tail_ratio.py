# src/performance_matrix/tail_ratio.py
import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.gain import Gain
from src.performance_matrix.return_matrix.loss import Loss


class TailRatio(BasePerformanceMatrix):
    """Calculates the Tail Ratio."""

    def calculate(self):
        """
        Calculate the Tail Ratio.
        :return: Tail Ratio as a float.
        """
        gains = Gain(self.stock_data).calculate()
        losses = Loss(self.stock_data).calculate()

        avg_gain = gains.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return float('inf')  # Avoid division by zero
        return avg_gain / avg_loss
