import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class MovingAverageConvergenceDivergence(BaseIndicator):
    """Calculates the Moving Average Convergence Divergence (MACD)."""

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__(fast_period)
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.Series):
        if data is None or data.empty:
            logging.warn("MovingAverageConvergenceDivergence no data for stock.")
            return None

        # Calculate the fast and slow EMAs
        fast_ema = data.ewm(span=self.period, adjust=False).mean()
        slow_ema = data.ewm(span=self.slow_period, adjust=False).mean()

        # Calculate the MACD and Signal Line
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()

        # Check if we have more than one value
        macd_value = macd.iloc[-1]
        signal_line_value = signal_line.iloc[-1]

        # Ensure we only get scalar values (if the result is a Series)
        if isinstance(macd_value, pd.Series):
            macd_value = macd_value.values[-1]  # Get the last value as scalar
        if isinstance(signal_line_value, pd.Series):
            signal_line_value = signal_line_value.values[-1]  # Get the last value as scalar

        # Return as a dictionary with the latest values
        result = {
            'MACD': macd_value.item() if isinstance(macd_value, np.ndarray) else macd_value,
            'Signal Line': signal_line_value.item() if isinstance(signal_line_value, np.ndarray) else signal_line_value
        }

        return result
