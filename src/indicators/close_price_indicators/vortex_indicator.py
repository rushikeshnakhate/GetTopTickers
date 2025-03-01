import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class VortexIndicator(BaseIndicator):
    """Calculates the Vortex Indicator."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("ForceIndex no data for stock.")
            return None

        tr = pd.concat([data['High'] - data['Low'],
                        (data['High'] - data['Close'].shift()).abs(),
                        (data['Low'] - data['Close'].shift()).abs()], axis=1).max(axis=1)

        vm_plus = (data['High'] - data['High'].shift()).rolling(window=self.period).sum()
        vm_minus = (data['Low'].shift() - data['Low']).rolling(window=self.period).sum()

        vi_plus = vm_plus / tr.rolling(window=self.period).sum()
        vi_minus = vm_minus / tr.rolling(window=self.period).sum()

        return vi_plus.iloc[-1], vi_minus.iloc[-1]
