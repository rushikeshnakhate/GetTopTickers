# src/performance_matrix/sterling_ratio.py
import logging

import pandas as pd

from src.performance_matrix.base_performance_matrix import BasePerformanceMatrix
from src.performance_matrix.return_matrix.annualized_return import AnnualizedReturn
from src.performance_matrix.risk_metrics.maximum_drawdown import MaximumDrawdown


class SterlingRatio(BasePerformanceMatrix):
    """Calculates the Sterling Ratio."""

    def calculate(self):
        """
        Calculate the Sterling Ratio.
        :return: Sterling Ratio as a float.
        """
        try:
            annualized_return = AnnualizedReturn(self.stock_data).calculate()
            max_drawdown = MaximumDrawdown(self.stock_data).calculate()

            if max_drawdown == 0:
                return float('inf')  # Avoid division by zero
            return annualized_return / abs(max_drawdown)
        except Exception as e:
            logging.error("SterlingRatio failed with error={}".format(e))
            return str(e)
