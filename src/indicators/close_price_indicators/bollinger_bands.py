import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class BollingerBands(BaseIndicator):
    """Calculates Bollinger Bands."""

    def __init__(self, period=14, num_std=2):
        super().__init__(period)
        self.num_std = num_std

    def calculate(self, data: pd.DataFrame) -> dict:
        if data is None or data.empty:
            # raise ValueError("Data cannot be empty.")
            logging.warn("BollingerBands no data for stock.")
            return None

        # Calculate moving average and rolling standard deviation
        moving_avg = data.rolling(window=self.period).mean()
        rolling_std = data.rolling(window=self.period).std()

        # Calculate the upper and lower bands
        upper_band = moving_avg + (rolling_std * self.num_std)
        lower_band = moving_avg - (rolling_std * self.num_std)

        # Return a dictionary with "Upper Band" and "Lower Band"
        return {
            "Upper Band": upper_band.iloc[-1],  # Last value of upper band
            "Lower Band": lower_band.iloc[-1],  # Last value of lower band
        }
