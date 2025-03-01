import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class AverageTrueRange(BaseIndicator):
    """Calculates the Average True Range (ATR)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("AverageTrueRange no data for stock.")
            return None
            # raise ValueError("Data cannot be empty.")

        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()

        return atr.iloc[-1]
