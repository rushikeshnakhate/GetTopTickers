import logging

import pandas as pd

from src.indicators.base_indicator import BaseIndicator


class ExponentialMovingAverage(BaseIndicator):
    """Calculates the Exponential Moving Average (EMA)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.Series):
        if data is None or data.empty:
            # raise ValueError("Data cannot be empty.")
            logging.warn("ExponentialMovingAverage no data for stock.")
            return None
        return data.ewm(span=self.period, adjust=False).mean().iloc[-1]
