import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class ADLine(BaseIndicator):
    """Calculates the Accumulation/Distribution Line."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            # raise ValueError("Data cannot be empty.")
            logging.warn("ADLine no data for stock.")
            return None

        money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (
                data['High'] - data['Low'])
        money_flow_volume = money_flow_multiplier * data['Volume']
        ad_line = money_flow_volume.cumsum()

        return ad_line.iloc[-1]
