import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class KeltnerChannel(BaseIndicator):
    """Calculates the Keltner Channel."""

    def __init__(self, period=14, multiplier=2):
        super().__init__(period)
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("KeltnerChannel no data for stock.")
            return None

        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        moving_avg = typical_price.rolling(window=self.period).mean()
        atr = data['Close'].diff().abs().rolling(window=self.period).mean()

        upper_band = moving_avg + (self.multiplier * atr)
        lower_band = moving_avg - (self.multiplier * atr)

        return upper_band.iloc[-1], lower_band.iloc[-1]
