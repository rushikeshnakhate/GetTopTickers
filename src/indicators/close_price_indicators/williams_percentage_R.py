import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class WilliamsR(BaseIndicator):
    """Calculates the Williams %R indicator."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("ForceIndex no data for stock.")
            return None

        highest_high = data['High'].rolling(window=self.period).max()
        lowest_low = data['Low'].rolling(window=self.period).min()

        williams_r = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))

        return williams_r.iloc[-1]
