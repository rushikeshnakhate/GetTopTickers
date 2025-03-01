import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class EaseOfMovement(BaseIndicator):
    """Calculates the Ease of Movement indicator."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("EaseOfMovement no data for stock.")
            return None

        box_ratio = (data['High'] - data['Low']) / (data['Volume'] / 100000)
        ease_of_movement = box_ratio.rolling(window=self.period).mean()

        return ease_of_movement.iloc[-1]
