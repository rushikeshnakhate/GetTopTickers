import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class ForceIndex(BaseIndicator):
    """Calculates the Force Index."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("ForceIndex no data for stock.")
            return None

        force_index = data['Close'].diff() * data['Volume']
        force_index_ma = force_index.rolling(window=self.period).mean()

        return force_index_ma.iloc[-1]
