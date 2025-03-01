import logging

import numpy as np

from src.indicators.base_indicator import BaseIndicator


class MovingAverageIndicator(BaseIndicator):
    """Calculates the Simple Moving Average (SMA)."""

    def __init__(self, period=None):
        super().__init__(period)

    def calculate(self, data):
        if data is None or data.empty:
            logging.warn("MovingAverageIndicator no data for stock.")
            # raise ValueError("Data cannot be empty.")
            return None

        period = self.period if self.period else len(data)  # Default: full length
        if len(data) < period:
            logging.warn(
                "MovingAverageIndicatornot enough data for stock len(data)={}, period={}".format(len(data), period))
            return None

        return np.mean(data[-period:])  # SMA of last `period` values
