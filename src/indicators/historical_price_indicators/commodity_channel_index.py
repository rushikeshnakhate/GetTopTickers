import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class CommodityChannelIndex(BaseIndicator):
    """Calculates the Commodity Channel Index (CCI)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("CommodityChannelIndex no data for stock.")
            return None

        tp = (data['High'] + data['Low'] + data['Close']) / 3
        sma = tp.rolling(window=self.period).mean()
        mad = tp.rolling(window=self.period).apply(lambda x: np.fabs(x - x.mean()).mean())

        cci = (tp - sma) / (0.015 * mad)

        return cci.iloc[-1]
