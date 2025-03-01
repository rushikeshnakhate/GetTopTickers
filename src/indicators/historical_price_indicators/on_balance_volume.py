import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class OnBalanceVolume(BaseIndicator):
    """Calculates the On-Balance Volume (OBV)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("OnBalanceVolume no data for stock.")
            return None

        obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                       np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))

        return obv.cumsum()[-1]
