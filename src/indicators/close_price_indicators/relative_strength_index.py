import logging
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class RelativeStrengthIndex(BaseIndicator):
    """Calculates the Relative Strength Index (RSI)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.Series):
        if data is None or data.empty:
            # raise ValueError("Data cannot be empty.")
            logging.warn("RelativeStrengthIndex no data for stock.")
            return None

        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))

        return rsi_series.iloc[-1]
